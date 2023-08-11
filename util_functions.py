import torch
import functorch as ft
import numpy as np
from torch.autograd.functional import jvp, vjp
from numpy import isnan
import matplotlib.pyplot as plt
from numpy import isnan

def Dh(x):
    dhx = x[:,range(1,x.shape[1])] - x[:,range(x.shape[1]-1)]
    dhx = torch.cat((torch.zeros(x.shape[0],1), dhx), 1)
    return dhx

def Dv(x):
    dvx = x[range(1,x.shape[0]),:] - x[range(x.shape[0]-1),:]
    dvx = torch.cat((torch.zeros(1,x.shape[1]), dvx), 0)
    return dvx

def DhT(x):
    dhx = x[:,range(x.shape[1]-1)] - x[:,range(1,x.shape[1])]
    dhx = torch.cat((dhx,torch.zeros(x.shape[0],1)), 1)
    return dhx

def DvT(x):
    dvx = x[range(x.shape[0]-1),:] - x[range(1,x.shape[0]),:]
    dvx = torch.cat((dvx,torch.zeros(1,x.shape[1])), 0)
    return dvx

def Dhc(x):
    dhx = x[:,range(1,x.shape[1])] - x[:,range(x.shape[1]-1)]
    dhx = torch.cat((torch.zeros(x.shape[0],1).cuda(), dhx), 1)
    return dhx

def DhTc(x):
    dhx = x[:,range(1,x.shape[1]-1)] - x[:,range(2,x.shape[1])]
    dhx = torch.cat((-x[:,1].reshape(x.shape[0],1), dhx, x[:,-1].reshape(x.shape[0],1)), 1)
    return dhx

def DvTc(x):
    dvx = x[range(1,x.shape[0]-1),:] - x[range(2,x.shape[0]),:]
    dvx = torch.cat((-x[1,:].reshape(1,x.shape[1]), dvx, x[-1,:].reshape(1,x.shape[1])), 0)
    return dvx

def Dvc(x):
    dvx = x[range(1,x.shape[0]),:] - x[range(x.shape[0]-1),:]
    dvx = torch.cat((torch.zeros(1,x.shape[1]).cuda(), dvx), 0)
    return dvx

    
def A_diff(x,dt,K):
    xk = x
    for i in range(K):
        xk = xk - dt*(DhT(Dh(xk)) + DvT(Dv(xk)))
    return xk

def A_diffc(x,dt,K):
    xk = x
    for i in range(K):
        xk = xk - dt*(DhTc(Dhc(xk)) + DvTc(Dvc(xk)))
    return xk


def A_nldiffc(x,dt,K,kappa):
    xk = x
    for i in range(K):
        dhx = Dhc(xk)
        dvx = Dvc(xk)
        diffusivity = 1/(1 + (dhx**2 + dvx**2)/kappa**2)
        xk = xk - dt*(DhTc(dhx*diffusivity) + DvTc(dvx*diffusivity))
    return xk

def A_nldiff(x,dt,K,kappa):
    xk = x
    for i in range(K):
        dhx = Dh(xk)
        dvx = Dv(xk)
        diffusivity = 1/(1 + (dhx**2 + dvx**2)/kappa**2)
        xk = xk - dt*(DhT(dhx*diffusivity) + DvT(dvx*diffusivity))
    return xk


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def A_curv_flowc(x,dt,K,tol):
    xk = x
    for i in range(K):
        dhx = Dhc(xk)
        dvx = Dvc(xk)
        gg = torch.sqrt(dhx**2 + dvx**2 + tol)
        kappa = DhTc(dhx/gg) + DvTc(dvx/gg)
        xk = xk - dt*gg*kappa    
    return xk

def A_curv_flow(x,dt,K,tol):
    xk = x
    for i in range(K):
        dhx = Dh(xk)
        dvx = Dv(xk)
        gg = torch.sqrt(dhx**2 + dvx**2 + tol)
        kappa = DhT(dhx/gg) + DvTc(dvx/gg)
        xk = xk - dt*gg*kappa    
    return xk


def power(A,x,uk):
    xk = x
    ss = []
    las = 1
    while True:
        _, t1 = jvp(A,uk,xk)
        t1 = t1.cpu().numpy()
        t1[isnan(t1)] = 0
        t1 = torch.tensor(t1).cuda()
        _, t2 = vjp(A,uk,t1)
        t2 = t2.cpu().numpy()
        t2[isnan(t2)] = 0
        t2 = torch.tensor(t2).cuda()
        xk = t2 + DhTc(Dhc(xk)) + DvTc(Dvc(xk))
        xk = xk/torch.sqrt(torch.sum(xk**2))
        _, t1 = jvp(A,uk,xk)
        t1 = t1.cpu().numpy()
        t1[isnan(t1)] = 0
        t1 = torch.tensor(t1).cuda()
        s = torch.sqrt(torch.sum(t1**2) + torch.sum(Dhc(xk)**2 + Dvc(xk)**2)).item()
        ss.append(s)
        if las > 1:
            if (ss[-1] - ss[-2])**2/ss[-2]**2 < 1e-6:
                break
        las += 1
    
    return s

def aempower(A,x,L,n,noise):
    xk = x.clone()
    ss = []
    las = 1
    while True:
        t1 = A(xk)
        t1 = torch.linalg.solve_triangular(L,t1.cpu().reshape(n*n,1),upper=False)*noise
        t11 = t1.clone().cuda()
        t1 = torch.linalg.solve_triangular(L.mT,t1,upper=True)
        t1 = torch.reshape(t1,(n,n)).cuda()
        t1 = A(t1)*noise
        xk = t1 + DhTc(Dhc(xk)) + DvTc(Dvc(xk))
        xk = xk/torch.sqrt(torch.sum(xk**2))
        s = torch.sqrt(torch.sum(t11**2) + torch.sum(Dhc(xk)**2 + Dvc(xk)**2)).item()
        ss.append(s)
        if las > 1:
            if (ss[-1] - ss[-2])**2/ss[-2]**2 < 1e-6:
                break
        las += 1
    
    return s


def l1_obj_func(A,x,y,lam):
    t1 = torch.sum(torch.abs(A(x) - y))
    t2 = lam*torch.sum(torch.abs(Dhc(x)) + torch.abs(Dvc(x)))
    return t1# + t2

def l2_obj_func(A,x,y,lam):
    t1 = 0.5*torch.sum((A(x) - y)**2)
    t2 = lam*torch.sum(torch.abs(Dhc(x)) + torch.abs(Dvc(x)))
    return t1# + t2
