import util_functions
import torch
from torch.autograd.functional import jvp, vjp
from numpy import isnan
import matplotlib.pyplot as plt


#Fixed approximation with primal dual
def pd_fixed(y,A,ATilde,ATildeT,init,lam,normtype):
    '''
    
    Parameters
    ----------
    y : Measurement data.
    A : Exact model.
    ATilde : Approximate linear model.
    ATildeT : Adjoint of ATilde.
    init : Initialization.
    lam : Regularization parameter.
    normtype : Type of data fidelity. 'l1' or 'l2'.

    Returns
    -------
    Reconstructed image, data fidelity over sequence iterations.

    '''
    if normtype != 'l1' and normtype != 'l2':
        print("Wrong normtype")
        return 0
    if normtype == 'l1':
        errfun = lambda x: util_functions.l1_obj_func(A,x,y,lam)
    elif normtype == 'l2':
        errfun = lambda x: util_functions.l2_obj_func(A,x,y,lam)
    n = y.size()[0]
    uk = init.clone()
    ub = init.clone()
    ubo = ub.clone()
    theta = 1
    lask = 1
    delta = torch.linspace(0,1,20)
    ld = torch.zeros(20)
    errk = [errfun(ub).item()]
    pk = 0
    qkh = 0
    qkv = 0  
    Lp = util_functions.power(ATilde,torch.randn(n,n).cuda(),ub)
    tau = 1/Lp
    sigma = 1/Lp  
    while True:       
        las = 1
        eps = A(ub) - ATilde(ub)
        ye = y - eps
        uboi = ub.clone()
        while True:
            if normtype == 'l2':
                pk = (pk + sigma*(ATilde(ub) - ye))/(1 + sigma)
            elif normtype == 'l1':
                temp = pk + sigma*(ATilde(ub) - ye)
                atemp = torch.abs(temp)
                atemp[atemp<1] = 1
                pk = temp/atemp
                
            Duh = util_functions.Dhc(ub)
            Duv = util_functions.Dvc(ub)
            temph = qkh + sigma*Duh
            atemph = torch.abs(temph)
            atemph[atemph<lam] = lam
            tempv = qkv + sigma*Duv
            atempv = torch.abs(tempv)
            atempv[atempv<lam] = lam
            qkh = lam*temph/atemph
            qkv = lam*tempv/atempv
            uold = uk.clone()
            uk = uold - tau*ATildeT(pk) - tau*(util_functions.DhTc(qkh) + util_functions.DvTc(qkv))
            uk[uk<0] = 0
            uk[uk>1] = 1
            ub = uk + theta*(uk - uold) 
            err2 = torch.sum((ub - uboi)**2)/torch.sum(uboi**2)
            print(err2)
            
            if err2 < 1e-6:
                break       
            
            uboi = ub.clone()
                
            las += 1
            
        for i in range(len(delta)):
            ukii = (1 - delta[i])*ubo + delta[i]*ub
            ld[i] = errfun(ukii) + lam*(torch.sum(torch.sqrt(util_functions.Dhc(ukii)**2 + util_functions.Dvc(ukii)**2)))
            
        stepsize = delta[ld==torch.min(ld)][0].item()   
            
        ub = (1 - stepsize)*ubo + stepsize*ub
        err3 = torch.sum((ub - ubo)**2)/torch.sum(ubo**2)
        
        errk.append(errfun(ub).item()) 
        if stepsize > 0:
            if err3 < 1e-6:
                break
        else:
            break
        
        ubo = ub.clone()
        uk = ub.clone()
        
        lask += 1
        
    return ub, errk

#Adaptive approximation with primal dual
def pd_adaptive(y,A,init,lam,normtype):
    '''

    Parameters
    ----------
    y : Measurement data.
    A : Exact model.
    init : Initialization.
    lam : Regularization parameter.
    normtype : Type of data fidelity. 'l1' or 'l2'. 

    Returns
    -------
    Reconstructed image, data fidelity over sequence iterations.

    '''
    if normtype != 'l1' and normtype != 'l2':
        print("Wrong normtype")
        return 0
    if normtype == 'l1':
        errfun = lambda x: util_functions.l1_obj_func(A,x,y,lam)
    elif normtype == 'l2':
        errfun = lambda x: util_functions.l2_obj_func(A,x,y,lam)
    uk = init.clone()
    ub = init.clone()
    u0 = init.clone()
    ubo = init.clone()
    theta = 1
    delta = torch.linspace(0,1,20)
    ld = torch.zeros(20)
    lask = 1
    errk = [errfun(ub).item()]
    pk = 0
    qkh = 0
    qkv = 0  
    pdg = []
    while True:          
        Au0 = A(u0)
        Lp = util_functions.power(A,u0,u0)
        tau = 1/Lp
        sigma = 1/Lp
        las = 1
        uboi = ub.clone()
        while True:
            t1 = jvp(A,u0,ub-u0)[1]
            t1 = t1.cpu().numpy()
            t1[isnan(t1)] = 0
            t1 = torch.tensor(t1).cuda()
            if normtype == 'l2':
                pk = (pk + sigma*(Au0 + t1 - y))/(1 + sigma)
            elif normtype == 'l1':
                temp = pk + sigma*(Au0 + t1 - y)
                atemp = torch.abs(temp)
                atemp[atemp<1] = 1
                pk = temp/atemp                
            
            Duh = util_functions.Dhc(ub)
            Duv = util_functions.Dvc(ub)
            temph = qkh + sigma*Duh
            atemph = torch.abs(temph)
            atemph[atemph<lam] = lam
            tempv = qkv + sigma*Duv
            atempv = torch.abs(tempv)
            atempv[atempv<lam] = lam
            qkh = lam*temph/atemph
            qkv = lam*tempv/atempv
            uold = uk.clone()
            t1 = vjp(A,u0,pk)[1]
            t1 = t1.cpu().numpy()
            t1[isnan(t1)] = 0
            t1 = torch.tensor(t1).cuda()
            uk = uold - tau*t1 - tau*(util_functions.DhTc(qkh) + util_functions.DvTc(qkv))
            uk[uk<0] = 0
            uk[uk>1] = 1
            ub = uk + theta*(uk - uold)   
            err2 = torch.sum((ub - uboi)**2)/torch.sum(uboi**2)  
            print(err2)
            
            if err2 < 1e-6:
                break
            
            uboi = ub.clone()
                
            las += 1
            
        for i in range(len(delta)):
            ukii = (1 - delta[i])*ubo + delta[i]*ub
            ld[i] = errfun(ukii) + lam*(torch.sum(torch.sqrt(util_functions.Dhc(ukii)**2 + util_functions.Dvc(ukii)**2)))
            
        stepsize = delta[ld==torch.min(ld)][0].item()    
        
        ub = stepsize*ub + (1 - stepsize)*ubo
        err3 = torch.sum((ub - ubo)**2)/torch.sum(ubo**2)
        uk = ub.clone()  
        u0 = ub.clone()      
        
        errk.append(errfun(ub).item())  
        if stepsize > 0:
            if err3 < 1e-6:
                    break
        else:
            break

        ubo = ub.clone()
        
        lask += 1
        
    return ub, errk

def pd_nocor(y,ATilde,ATildeT,init,lam,normtype,A):
    '''

    Parameters
    ----------
    y : Measurement data.
    ATilde : Approximate linear model.
    ATildeT : Adjoint of ATilde.
    init : Initialization.
    lam : Regularization parameter.
    normtype : Type of data fidelity. 'l1' or 'l2'.
    A : Exact model.

    Returns
    -------
    Reconstructed image, data fidelity over primal-dual iterations.

    '''
    if normtype != 'l1' and normtype != 'l2':
        print("Wrong normtype")
        return 0
    if normtype == 'l1':
        errfun = lambda x: util_functions.l1_obj_func(A,x,y,lam)
    elif normtype == 'l2':
        errfun = lambda x: util_functions.l2_obj_func(A,x,y,lam)
    
    n = y.size()[0]
    uk = init.clone()
    ub = init.clone()
    ubo = init.clone()
    err = [errfun(ub).item()]
    theta = 1
    lask = 1
    pk = 0
    qkh = 0
    qkv = 0  
    Lp = util_functions.power(ATilde,torch.randn(n,n).cuda(),ub)
    tau = 1/Lp
    sigma = 1/Lp  
    while True:          
        if normtype == 'l2':
            pk = (pk + sigma*(ATilde(ub) - y))/(1 + sigma)
        elif normtype == 'l1':
            temp = pk + sigma*(ATilde(ub) - y)
            atemp = torch.abs(temp)
            atemp[atemp<1] = 1
            pk = temp/atemp
            
        Duh = util_functions.Dhc(ub)
        Duv = util_functions.Dvc(ub)
        temph = qkh + sigma*Duh
        atemph = torch.abs(temph)
        atemph[atemph<lam] = lam
        tempv = qkv + sigma*Duv
        atempv = torch.abs(tempv)
        atempv[atempv<lam] = lam
        qkh = lam*temph/atemph
        qkv = lam*tempv/atempv
        uold = uk.clone()
        uk = uold - tau*ATildeT(pk) - tau*(util_functions.DhTc(qkh) + util_functions.DvTc(qkv))
        uk[uk<0] = 0
        uk[uk>1] = 1
        ub = uk + theta*(uk - uold)
        err2 = torch.sum((ub - ubo)**2)/torch.sum(ubo**2) 
        err.append(errfun(ub).item())
        
        if err2 < 1e-6:
            break   
        
        ubo = ub.clone()
        
        lask += 1
        
    return ub, err