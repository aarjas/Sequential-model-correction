import torch
import util_functions
import optimization_functions
import matplotlib.pyplot as plt
from skimage import data
from skimage.metrics import structural_similarity



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Load image
caller = getattr(data, 'camera')
x = caller()
x = x[0:x.shape[0]:2,0:x.shape[1]:2]
x = torch.tensor(x/255,dtype=torch.float).cuda()


#Define true and approximative models
A = lambda x: util_functions.A_nldiffc(x,0.1,15,0.1)        #Nonlinear diffusion
#A = lambda x: util_functions.A_curv_flowc(x,0.1,15,1e-3)   #Curvature flow
ATilde = lambda x: util_functions.A_diffc(x,0.1,15)


#Create data
y = A(x)
y = y + 1e-2*torch.randn(y.size()).cuda() #Gaussian noise
#y[torch.bernoulli(0.02*torch.ones(y.size())).bool()] = 1  
#y[torch.bernoulli(0.02*torch.ones(y.size())).bool()] = 0  #Salt and pepper noise


init = 0.01*torch.randn(x.size()).cuda()
lam = 1e-3
data_fid = 'l2'

#Reconstruction with approximative model with no correction
xhat_nocor, err_nocor = optimization_functions.pd_nocor(y,ATilde,ATilde,init,lam,data_fid,A)

#Reconstruction with approximative model with fixed approximation
xhat_fixed, err_fixed = optimization_functions.pd_fixed(y,A,ATilde,ATilde,init,lam,data_fid)

#Reconstruction with approximative model with adaptive approximation
xhat_adaptive, err_adaptive = optimization_functions.pd_adaptive(y,A,init,lam,data_fid)


#Compare reconstructions
plt.figure(figsize=(12, 12), dpi=300)
plt.subplot(2,2,1)
plt.imshow(xhat_nocor.cpu(),cmap='gray')
plt.axis('off')
plt.title("No correction ({}, {})".format(round(util_functions.psnr(xhat_nocor,x).item(),2), round(structural_similarity(xhat_nocor.cpu().numpy(),x.cpu().numpy(), full=True)[0],2)))
plt.subplot(2,2,2)
plt.imshow(xhat_fixed.cpu(),cmap='gray')
plt.axis('off')
plt.title("Fixed approximation ({}, {})".format(round(util_functions.psnr(xhat_fixed,x).item(),2), round(structural_similarity(xhat_fixed.cpu().numpy(),x.cpu().numpy(), full=True)[0],2)))
plt.subplot(2,2,3)
plt.imshow(xhat_adaptive.cpu(),cmap='gray')
plt.axis('off')
plt.title("Adaptive approximation ({}, {})".format(round(util_functions.psnr(xhat_adaptive,x).item(),2), round(structural_similarity(xhat_adaptive.cpu().numpy(),x.cpu().numpy(), full=True)[0],2)))
plt.subplot(2,2,4)
plt.imshow(x.cpu(),cmap='gray')
plt.axis('off')
plt.title("Ground truth")


#Compare losses
plt.figure(figsize=(12, 4), dpi=300)
plt.semilogy(err_fixed,c="r",ls='-.',label="Fixed approximation")
plt.semilogy(err_adaptive,c="k",ls='--',label="Adaptive approximation")
plt.axhline(y=err_nocor[-1],label="No correction",c="b",ls='-')
plt.legend(loc="upper right",prop={'size': 15})
plt.xlabel("Iteration")
plt.ylabel("Data fidelity")

