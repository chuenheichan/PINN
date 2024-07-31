##test github pinn
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:18:46 2022

In this script, we use PINN to discover pulse that is symmetric,
and u(0) = u(T) = kappa_a = 0
and u'(0) = u'(T) = 0


@author: Bikun Li
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from itertools import chain
import argparse

parser = argparse.ArgumentParser('2driving_helix')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_iters', type=int, default=100)
parser.add_argument('--learning_rate',type=int,default=1) #multiplied by 1e-5 below
args = parser.parse_args()

torch.manual_seed(args.seed) #fixed the random seed # can be removed
# torch.set_default_dtype(torch.float32) # use float32 so the loss can go under 1e-8

print("GPU availability check: \n")
print("Is cuda available?: "+str(torch.cuda.is_available()))
print("Cuda device count: "+str(torch.cuda.device_count()))
if torch.cuda.is_available():
    print("Current device: "+str(torch.cuda.current_device())) 
print("\n")
print("\n")

USE_CUDA = True

mydevice = torch.device("cuda:0" if USE_CUDA else "cpu")
if USE_CUDA:
    print("\n**Running with GPU**\n")
else:
    print("\n**Running with CPU**\n")

pre_trained_model_path = './newtrain_T13p50_gpumodel_dict.pt'
filename = 'newtrain_2driving_helix_hadamard_seed'+str(args.seed)+'lr'+str(args.learning_rate)+'eneg5' # the name of the subfolder that saves the data
mypath = './data/'+filename # customize path for storing data

start_time = time.time() # start the clock

# try to solve the ODE for 3D Dubins curve with CONSTANT torsion:

    
# the admissible curvature interval: (kappa_a, kappa_b)
#kappa_a, kappa_b = (0, 2.0)
#kappa=1
#kappa=torch.nn.Parameter(torch.tensor(1.0, device=mydevice))
kappa=torch.tensor([1.0], device=mydevice, requires_grad=True)
# the admissible driving field interval: (psi_a,psi_b)
#phi_a,phi_b = (0,2*pi)
tau_magnitude=torch.tensor([1.0], device=mydevice, requires_grad=True)

# angles_f = torch.tensor([0,0,0],dtype = torch.float32)
c_0 = torch.zeros(3, 3)
c_f = torch.zeros(3, 3)
rot_0 = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],dtype = torch.float32) #identity
rot_f = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]],dtype = torch.float32) #Hadamard

# fixed the arc length?:
T = 10

# Domain and Sampling
def sampling(n, mydevice):
    # normalization: -1<s<1#
    # s = torch.linspace(-1, 1, n, requires_grad=True).view(-1,1).to(device=mydevice) 
    s = 2*(torch.rand(n,1,requires_grad = True).view(-1,1) - 0.5).to(device=mydevice) 
    return s

# define the class of Neural Network
class Pinn(torch.nn.Module):
    
    def __init__(self, nodes_num=10, feature_num=1):
        super(Pinn, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,nodes_num),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes_num,nodes_num),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes_num,nodes_num),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes_num,nodes_num),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes_num,nodes_num),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes_num,nodes_num),
            torch.nn.Tanh(),
	    # torch.nn.Linear(nodes_num,nodes_num),
            # torch.nn.Tanh(),
            torch.nn.Linear(nodes_num, feature_num) # the outputs are x, y, z, alpha, beta, gamma, omega
        )
    def forward(self, s):
        return self.net(s).cuda() if USE_CUDA else self.net(s) # attempt to use cuda
    
# Loss function: MSE type
mse = torch.nn.MSELoss()

# define the derivative of non-negative order
def deriv(f, x, order = 1):
    if order == 1:
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                                   create_graph=True, only_inputs=True, )[0]
    else:
        return deriv(deriv(f,x), x, order=order-1)

def l_eom(f, T, n_eom, mydevice): # loss function for equation of motion
    s = sampling(n_eom, mydevice)
    
    C = f(s)[:,0:9]
    dC = torch.zeros_like(C).to(device=mydevice)
    ddC = torch.zeros_like(C).to(device=mydevice)
    
    for k in range(9):
        dC[:,k] = 2/T * deriv(C[:,k], s, 1).view(-1) # dC/dt at t_i ##flattened row vector?
        ddC[:,k] = ((2/T)**2) * deriv(C[:,k], s, 2).view(-1) # (d/dt)dC/dt at t_i
    flip_partition = torch.logical_and((s > -1 / 3), (s < 1 / 3))
    flip_fn=2*flip_partition-1 #if -1/3<s<=1/3,flip_fn=-1;otherwise flip_fn=+1  #need grad here?
    tau=tau_magnitude*flip_fn
    cond_fs = torch.zeros_like(ddC).to(device=mydevice)
    cond_fs[:,0:3] = kappa*dC[:,3:6]
    cond_fs[:,3:6] = -kappa*dC[:,0:3] + tau.repeat(1,3)*dC[:,6:9]
    cond_fs[:,6:9] = -tau.repeat(1,3)*dC[:,3:6]
    
    # cond_0 = torch.zeros_like(s, requires_grad=False).view(-1).to(device=mydevice)
    cond_1 = torch.ones_like(s, requires_grad=False).view(-1).to(device=mydevice)
    
    # chirality:
    t_cross_n = torch.linalg.cross(dC[:,0:3], dC[:,3:6]).to(device=mydevice)
    
    # norms and inner products of dC/dt, which should follow orthonormal condition
    norm_t = torch.linalg.norm(dC[:,0:3],dim=1).to(device=mydevice)
    norm_n = torch.linalg.norm(dC[:,3:6],dim=1).to(device=mydevice)
    norm_b = torch.linalg.norm(dC[:,6:9],dim=1).to(device=mydevice)
    
    # ip_tn = torch.sum(dC[:,0:3]*dC[:,3:6], dim=1).to(device=mydevice)
    # ip_tb = torch.sum(dC[:,0:3]*dC[:,6:9], dim=1).to(device=mydevice)
    # ip_nb = torch.sum(dC[:,3:6]*dC[:,6:9], dim=1).to(device=mydevice)
    
    l = mse(ddC,cond_fs) + mse(t_cross_n, dC[:,6:9])\
        + mse(norm_t, cond_1) + mse(norm_n, cond_1) + mse(norm_b, cond_1)
        # \
        # + mse(ip_tn, cond_0) + mse(ip_tb, cond_0) + mse(ip_nb, cond_0)

    return l

def l_bc(f, c_0, c_f, rot_0, rot_f, T, mydevice): # loss function for boundary conditions
    
    s_0 = torch.tensor([-1], dtype=torch.float32, requires_grad=True).to(device=mydevice)
    s_f = torch.tensor([1], dtype=torch.float32, requires_grad=True).to(device=mydevice)
    
    cond_c_0 = c_0.view(-1).to(device=mydevice)
    cond_c_f = c_f.view(-1).to(device=mydevice)
    
    cond_rot_0 = rot_0.view(-1).to(device=mydevice)
    cond_rot_f = rot_f.view(-1).to(device=mydevice)
    
    c_0_out = f(s_0)
    c_f_out = f(s_f)
    dc_0_out = torch.zeros_like(cond_rot_0).to(device=mydevice)
    dc_f_out = torch.zeros_like(cond_rot_f).to(device=mydevice)
    
    for k in range(9):
        dc_0_out[k] = 2/T*deriv(c_0_out[k],s_0,1)
        dc_f_out[k] = 2/T*deriv(c_f_out[k],s_f,1)
    
    l_closed_curve=mse(cond_c_0[[0,1,2]], c_0_out[[0,1,2]]) + mse(cond_c_f[[0,1,2]], c_f_out[[0,1,2]])
    l_bc=mse(cond_rot_0, dc_0_out) + mse(cond_rot_f, dc_f_out) #
    return l_closed_curve,l_bc



def check_optimizer_param():
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ')
                for vp in v:
                    outputs += (str(vp.shape).ljust(30) + ' ')
        print(outputs)

# create model: f
f = Pinn(nodes_num=256,feature_num=9).to(device=mydevice)

# create optimizer: (ADAM)
#learning_rate = 1e-5
learning_rate = args.learning_rate*1e-5
optimizer = torch.optim.Adam(params=chain(f.parameters(),(kappa,tau_magnitude)), lr = learning_rate, eps=1e-16)
#check_optimizer_param()


# schedule the decay of lr by factor gamma every step_size:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)


# Start the training loop
n_eom = 128
n_iters = 100
steps_array = []
loss_history = []
loss_history_separate=[]
for i in range(1,args.n_iters+1):
    # reset learning rate every 10000 step
    if (i % 10000 == 0 and i != args.n_iters):
        optimizer = torch.optim.Adam(params=chain(f.parameters(),(kappa,tau_magnitude)), lr=learning_rate, eps=1e-16)
        #optimizer = torch.optim.Adam(params=f.parameters(), lr=scheduler.get_last_lr()[-1], eps=1e-16)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    # empty the gradient:
    optimizer.zero_grad()
    
    _l_eom = l_eom(f, T, n_eom, mydevice)
    _l_closed_curve, _l_bc = l_bc(f, c_0, c_f, rot_0, rot_f, T, mydevice)
    l = _l_eom+_l_closed_curve+_l_bc

    # backward pass:
    l.backward()
    
    # update the weights
    optimizer.step()

    # # schedule the decay of lr by factor gamma every step_size:
    scheduler.step()

    if (i==1 or i % 20 == 0) : # record the training loss every specific steps
        steps_array = np.append(steps_array, i)
        loss_history = np.append(loss_history, l.item())
        loss_history_separate = np.append(loss_history_separate, [_l_eom.item(),_l_closed_curve.item(),_l_bc.item()])

    if i % 10 == 0:
        print("Step = %2d, loss = %.8f" % (i, l.item()))

# end the clock and print the elasped time
print("--- %s seconds ---" % (time.time() - start_time))
print('kappa=',kappa)
print('tau_magnitude=',tau_magnitude)

# inference
# transfer the result from GPU to CPU
USE_CUDA = False
f.cpu()

s_test = torch.linspace(-1, 1, 1024).view(-1,1)
t_test = T*(s_test + 1)/2
# Data for a three-dimensional line
out_pred = f(s_test)
x_pred = out_pred[:,0].detach().numpy()
y_pred = out_pred[:,1].detach().numpy()
z_pred = out_pred[:,2].detach().numpy()
flip_fn=2*(s_test>0)-1 #If -1<=s<=0,flip_fn=+1; if 0<s<=1,flip_fn=-1  #need grad here?
tau_pred=tau_magnitude.item()*flip_fn
#u_pred = omega_to_ctrl(f(s_test)[:,9], kappa_a, kappa_b).detach().numpy()
#phi_pred=neuron_to_phi(f(s_test)[:,10], phi_a, phi_b).detach().numpy()

# plot
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(3, 3, 1, projection='3d')
ax.plot3D(x_pred, y_pred, z_pred, 'navy')
ax.view_init(30, 20)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_box_aspect((np.ptp(x_pred), np.ptp(y_pred), np.ptp(z_pred)))

ax = fig.add_subplot(3, 3, 2, projection='3d')
ax.plot3D(x_pred, y_pred, z_pred, 'navy')
ax.view_init(30, 80)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_box_aspect((np.ptp(x_pred), np.ptp(y_pred), np.ptp(z_pred)))

ax = fig.add_subplot(3, 3, 3, projection='3d')
ax.plot3D(x_pred, y_pred, z_pred, 'navy')
ax.view_init(30, 140)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_box_aspect((np.ptp(x_pred), np.ptp(y_pred), np.ptp(z_pred)))

ax = fig.add_subplot(3, 3, 4)
ax.plot(t_test, kappa.item()*torch.ones_like(t_test),'r',ms=2, label="predicted kappa")
# ax.set_title("predicted control field")
#ax.set_ylim(kappa_a-0.1,kappa_b+0.1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\Omega(t)$")
ax.set_title(r"$T={:.3f}$".format(T))

ax = fig.add_subplot(3, 3, 5)
ax.plot(steps_array, loss_history,'g',linewidth=1, label="loss")
#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("loss")
#ax.set_yticks([1e0,1e-2,1e-4,1e-6,1e-8],['$1$','$10^{-2}$','$10^{-4}$','$10^{-6}$','$10^{-8}$'])

ax = fig.add_subplot(3, 3, 6)
ax.plot(t_test,x_pred,'-r',linewidth=1, label="x")
ax.plot(t_test,y_pred,'-g',linewidth=1, label="y")
ax.plot(t_test,z_pred,'-b',linewidth=1, label="z")
ax.legend(loc=2, prop={'size': 7})

ax = fig.add_subplot(3, 3, 7)
ax.plot(t_test, tau_pred,'r',ms=2, label="predicted tau")
# ax.set_title("predicted control field")
#ax.set_ylim(phi_a-0.1,phi_b+0.1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$d\Phi(t)/dt$")
ax.set_title(r"$T={:.3f}$".format(T))
ax.set_xlabel(r"$t$")

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.7, 
                    hspace=0.5)

# saving:
if not os.path.exists(mypath):
    os.makedirs(mypath) # create the directory if it does not exist

# save the plotting    
plt.savefig(mypath+'/plot_'+filename+'.pdf', dpi='figure')

plt.show()
# save the model
torch.save(f.state_dict(),mypath+'/'+filename+'model_dict.pt')


# save the loss history
#np.savetxt(mypath+"/loss.txt", np.transpose([steps_array, loss_history]))
np.savetxt(mypath+"/loss.txt", np.hstack((np.transpose([steps_array, loss_history]),np.reshape(loss_history_separate,(-1,3)))))#start load env and run python
