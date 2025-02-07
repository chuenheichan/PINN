
# -*- coding: utf-8 -*-
"""
In this script, we use physics-informed neural network to find a 8-dimensional space curve
which encodes a quantum control pulse in a 3-level system.
This is based on the SCQC formalism in PRXQuantum.2.010341

@author: Chuen Hei Chan @Feb 2024
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import time
import os
import torch.nn.utils.prune as prune
import math
from sympy import *
from math import pi
from simpy_diff_geometry import cal_evolution_A,subs_Control_by_ts,cal_evec_bc
import argparse
#from simplify import simplify


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--n_eom', type=int, default=101)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--Adam_niters', type=int, default=100)
parser.add_argument('--LBFGS_niters', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--BFGS_tol_change',type=float ,default=1e-9)
parser.add_argument('--BFGS_tol_grad',type=float ,default=1e-7)
parser.add_argument('--leom_weight',type=float ,default=0.1)
args = parser.parse_args()

torch.manual_seed(args.seed) #fixed the random seed # can be removed

#dtype = torch.float64
dtype = torch.float32
torch.set_default_dtype(dtype) # use float32 so the loss can go under 1e-8
#torch.set_printoptions(precision=16)

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
filename = '3level_simframe_HFromHybrid_seed'+ str(args.seed)+'leom_weight'+str(args.leom_weight) # the name of the subfolder that saves the data
mypath = './data/'+filename # customize path for storing data

initial_start_time = time.time() # start the clock

##Follow all parameters from the paper of hybrid approach
Omega_a, Omega_b = (0, 2.0)
delta_max = 2.0

dG=[None]*8
dG_coeff= [[None for i in range(8)] for j in range(8)]
dG_coeff_ts= [[None for i in range(8)] for j in range(8)]
dG_ts=[None]*8
Evec=[None]*8
Evec_coeff = [[None for i in range(8)] for j in range(8)]
Evec_coeff_ts = [[None for i in range(8)] for j in range(8)]
Evec_ts=[None]*8
Enormsq=[None]*8
Enormsq_eval=[None]*8
Enormsq_ts=[None]*8
Curv=[None]*7
Curv_eval=[None]*7
Curv_ts=[None]*7
SD=[[None for i in range(8)] for j in range(8)]
SD_eval=[[None for i in range(8)] for j in range(8)]
SD_ts=[[None for i in range(8)] for j in range(8)]
Esubs_list=[None]*8
Esubs_lambdify_list=[None]*8
Esubs_ts_list=[None]*8
kappa=[None]*7
dOmega=[None]*6
dnOmega=[None]*6
A = [[None for i in range(8)] for j in range(8)]
A_ts= [[None for i in range(8)] for j in range(8)]

# boundary conditions: 
# angles_f = torch.tensor([0,0,0],dtype = dtype)
c_0 = torch.zeros(8, 8)
c_f = torch.zeros(8, 8)
#rot_0 = torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],dtype = dtype)
rot_0=torch.eye(8)
rot_f = rot_0
U_target = Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

# fixed the arc length?:
T = 10
#T=110

# Domain and Sampling
def sampling(n, mydevice,include_bc=False):
    # normalization: -1<s<1#
    if include_bc:
        s_0 = torch.tensor([[-1.0]], requires_grad=True).to(device=mydevice)
        s_int = 2 * (torch.rand(n-2, 1, requires_grad=True).view(-1, 1) - 0.5).to(device=mydevice)
        s_f = torch.tensor([[1.0]], requires_grad=True).to(device=mydevice)
        s = torch.cat((s_0, s_int, s_f), 0)
    else:
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

def l_eom_old(f,s, kappa, T, n_eom, mydevice): # loss function for equation of motion
    C = f(s)[:,0:64] #n_eomX64 ts
    for i in range(7):
        kappa[i]=f(s)[:,64+i].view(-1,1)
    #print('kappa[0]',kappa[0])

    dC = torch.zeros_like(C).to(device=mydevice)
    ddC = torch.zeros_like(C).to(device=mydevice)
    
    for k in range(64):
        dC[:,k] = 2/T * deriv(C[:,k], s, 1).view(-1) # dC/dt at t_i ##flattened row vector?
        ddC[:,k] = ((2/T)**2) * deriv(C[:,k], s, 2).view(-1) # (d/dt)dC/dt at t_i
    
    cond_fs = torch.zeros_like(ddC).to(device=mydevice)
    cond_fs[:,0:8] = kappa[0].repeat(1,8)*dC[:,8:16]
    #cond_fs[:,8:16] = -kappa[0].repeat(1,8)*dC[:,0:8] + kappa[1].repeat(1,8)*dC[:,8:16]
    for k in range(1,7):
        cond_fs[:,8*k:8*(k+1)]=-kappa[k-1].repeat(1,8)*dC[:,8*(k-1):8*k]+kappa[k].repeat(1,8)*dC[:,8*k:8*(k+1)]
    cond_fs[:,56:64] = -kappa[6].repeat(1,8)*dC[:,56:64]
    
    # cond_0 = torch.zeros_like(s, requires_grad=False).view(-1).to(device=mydevice)
    cond_1 = torch.ones_like(s, requires_grad=False).view(-1).to(device=mydevice)
    
    # chirality: #This enforces the orthogonal condition in 3d case. How to enforce in 8d???
    #t_cross_n = torch.linalg.cross(dC[:,0:3], dC[:,3:6]).to(device=mydevice)

    l = mse(ddC, cond_fs)  # + mse(t_cross_n, dC[:, 6:9]) \
    # + mse(norm_t, cond_1) + mse(norm_n, cond_1) + mse(norm_b, cond_1)

    # norms and inner products of dC/dt, which should follow orthonormal condition
    norm = [None] * 8
    #norm_t = torch.linalg.norm(dC[:,0:3],dim=1).to(device=mydevice)
    #norm_n = torch.linalg.norm(dC[:,3:6],dim=1).to(device=mydevice)
    #norm_b = torch.linalg.norm(dC[:,6:9],dim=1).to(device=mydevice)

    for i in range(8):
        norm[i]=torch.linalg.norm(dC[:,8*i:8*(i+1)],dim=1).to(device=mydevice)
        l+=mse(norm[i],cond_1)

    return l

def l_eom(f,s,A_ts,T,rot_0,rot_T,n_eom, mydevice):
    #s is n_eomX1 tensor
    #assume e_a,e_b are 64Xn_eomX1 tensor, initial and final condition for e_vec
    #A_ts[n][l] is n_eomX1 tensor,e_vec_ls[n] is 8Xn_eomX1 tensor
    e_a=torch.mul(rot_0.reshape(64,1),torch.ones(1,n_eom)).reshape(64,n_eom,1).to(device=mydevice)
    e_b=torch.mul(rot_T.reshape(64,1),torch.ones(1,n_eom)).reshape(64,n_eom,1).to(device=mydevice)
    h = f(s)[:, 0:64].permute(1,0).reshape(64,n_eom,1) #n_eomX64 ts becomes 64Xn_eomX1 ts
    one=torch.ones_like(s).to(device=mydevice)
    e_vec=(one-s)/2*e_a+(one+s)/2*e_b+(one-torch.square(s))/4*h  #e_a
    e_vec_ls=[None]*8
    for n in range(8):
        e_vec_ls[n]=e_vec[8*n:8*(n+1)]
    #e_vec_ls=[e_vec[8*n:8*(n+1)] for n in range(8)]

    #Impose equation of motion for the 8 frame vectors
    de_vec = torch.zeros_like(e_vec).to(device=mydevice)
    cond_de = torch.zeros_like(e_vec).to(device=mydevice)
    for i in range(64):
        de_vec[i]=2/T*deriv(e_vec[i],s,1)
    for n in range(8):
        for l in range(8):
            cond_de[8*n:8*(n+1)]+=A_ts[n][l]*e_vec_ls[l]
    #print('A_ts',A_ts)
    #print('cond_de',cond_de)
    #print('de_vec',de_vec)
    #Impose norm=1 condition
    norm = [None] * 8
    l_norm=0
    for n in range(8):
        norm[n]=torch.linalg.norm(e_vec_ls[n],dim=0).to(device=mydevice) #norm along dim=0 since e_vec_ls[n] is 8Xn_eomX1 tensor, return n_eomX1 tensor
        l_norm+=mse(norm[n],one)

    #Impose orthogonality condition
    l_dot=0
    cond_0 = torch.zeros_like(s, requires_grad=False).to(device=mydevice)
    for n in range(8):
        for m in range(n):
            l_dot+=mse(torch.sum(e_vec_ls[n]*e_vec_ls[m],dim=0),cond_0)


    #Impose dr(t)/dt=H1=c*e_3+d*e_8
    #r_vec=(torch.ones_like(s)+s)/2*f(s)[:, 64:72].permute(1,0).reshape(8,n_eom,1)
    r_vec=(one-torch.square(s))/4*f(s)[:, 64:72].permute(1,0).reshape(8,n_eom,1)  #Hard constraint on r_vec(0)=0=r_vec(T)
    
    dr_vec=torch.zeros_like(r_vec).to(device=mydevice)
    for i in range(8):
        dr_vec[i]=2/T*deriv(r_vec[i],s,1)
    #c=torch.tensor([c_eval],dtype=dtype).to(device=mydevice)
    #d=torch.tensor([d_eval],dtype=dtype).to(device=mydevice)
    c=torch.tensor([-1/2],dtype=dtype).to(device=mydevice)
    d=torch.tensor([-math.sqrt(3)/2],dtype=dtype).to(device=mydevice)
    cond_dr=c*e_vec_ls[2]+d*e_vec_ls[7]

    #print('shape',de_vec.size(),cond_de.size(),dr_vec.size(),cond_dr.size())
    return mse(de_vec,cond_de),mse(dr_vec,cond_dr),l_norm,l_dot

def get_omega_neuron(f,mydevice):
    s_0 = torch.tensor([-1], dtype=dtype, requires_grad=True).to(device=mydevice)
    omega_neuron=f(s_0)[[71]]
    return omega_neuron

def l_bc(f, c_0, c_f, rot_0, rot_f, T, mydevice): # loss function for boundary conditions
    
    s_0 = torch.tensor([-1], dtype=dtype, requires_grad=True).to(device=mydevice)
    s_f = torch.tensor([1], dtype=dtype, requires_grad=True).to(device=mydevice)
    
    cond_c_0 = c_0.view(-1).to(device=mydevice)
    cond_c_f = c_f.view(-1).to(device=mydevice)
    
    cond_rot_0 = rot_0.view(-1).to(device=mydevice)
    cond_rot_f = rot_f.view(-1).to(device=mydevice)
    
    c_0_out = f(s_0)
    c_f_out = f(s_f)
    dc_0_out = torch.zeros_like(cond_rot_0).to(device=mydevice)
    dc_f_out = torch.zeros_like(cond_rot_f).to(device=mydevice)
    
    #for k in range(8):
    for k in range(64):
        dc_0_out[k] = 2/T*deriv(c_0_out[k],s_0,1)
        dc_f_out[k] = 2/T*deriv(c_f_out[k],s_f,1)

    # closed curve starts/ends at origin
    #return mse(cond_c_0[0:8], c_0_out[0:8]) + mse(cond_c_f[0:8], c_f_out[0:8])\
    #    + mse(cond_rot_0, dc_0_out) + mse(cond_rot_f, dc_f_out)
    return mse(cond_rot_0, dc_0_out) + mse(cond_rot_f, dc_f_out)+mse(cond_c_0[0:8], c_0_out[0:8]) + mse(cond_c_f[0:8], c_f_out[0:8])

def l_p_old(f,Omega_a,Omega_b, mydevice): # loss function on the constraints of pulse shape
    s_ends = torch.tensor([-1,1], dtype=dtype, requires_grad=True).view(-1, 1).to(device=mydevice)
    Omega_neuron_ends = f(s_ends)[:, 72]
    Omega_ends = neuron_to_Omega(Omega_neuron_ends, Omega_a, Omega_b)
    cond_ends = Omega_a * torch.ones_like(Omega_ends, requires_grad=False).to(device=mydevice)
    #print('Omega_ends',Omega_ends,'cond_ends', cond_ends)
    return mse(Omega_ends, cond_ends)

def l_p(f, Omega_a, Omega_b, n_eom, mydevice):  # loss function on the constraints of pulse shape
    # the following code performs the constraint of symmetric pulse
    s = sampling(n_eom, mydevice)
    Omega_neuron = f(s)[:, 72].view(-1, 1)
    Omega_neuron_mirrored = f(-s)[:, 72].view(-1, 1)  # mirrored omega

    Omega = envelope_Omega(f, s, Omega_a, Omega_b)
    Omega_mirrored = envelope_Omega(f, -s, Omega_a, Omega_b)  # mirrored pulse

    # additional bc for control field at two ends:
    # we require:
    # u(0) = u(T) = u_a = 0
    # u'(0) = u'(T) = 0
    s_ends = torch.tensor([-1, 1], dtype=torch.float32, requires_grad=True).view(-1, 1).to(device=mydevice)
    Omega_neuron_ends = f(s_ends)[:, 72]
    Omega_ends = neuron_to_Omega(Omega_neuron_ends, Omega_a, Omega_b)
    dOmega_ends = deriv(Omega_ends, s_ends, 1)
    cond_ends = Omega_a * torch.ones_like(Omega_ends, requires_grad=False).to(device=mydevice)
    cond_ends_0 = torch.zeros_like(dOmega_ends, requires_grad=False).to(device=mydevice)
    return mse(Omega, Omega_mirrored) + mse(Omega_ends, cond_ends) + mse(dOmega_ends, cond_ends_0)

steepness = 2*T
def envelope(t): #smooth square envelope
    return 1/torch.tanh(torch.tensor(steepness/4).to(device=mydevice))*(torch.tanh(steepness*t/(4*T)) - torch.tanh(steepness*(t-T)/(4*T))) - 1


def envelope_Omega(f,s,Omega_a,Omega_b):
    #Omega_neuron = f(s)[:, 72].view(-1, 1)
    #Omega = (1-torch.exp(-(s+1)/2*T))*Omega_neuron
    #Omega = (1-torch.exp(-(s+1)/2*T))*(Omega_a + (Omega_b - Omega_a) * (torch.tanh(Omega_neuron)+1)/2)
    #Omega = Omega_a + (Omega_b - Omega_a) * torch.sin(Omega_neuron) ** 2  ##used in Bikun's version
    Omega = Omega_b * (2 / pi) * envelope(T/2*(s+1).view(-1,1)) * torch.atan(f(s)[:,72].view(-1,1)) * torch.sin(f(s)[:,73].view(-1,1))
    return Omega



def neuron_to_Omega(Omega_neuron,Omega_a,Omega_b):
    # Omega = (1-torch.exp(-(s+1)/2*T))*Omega_neuron
    # Omega = (1-torch.exp(-(s+1)/2*T))*(Omega_a + (Omega_b - Omega_a) * (torch.tanh(Omega_neuron)+1)/2)
    #Omega = Omega_a + (Omega_b - Omega_a) * torch.sin(Omega_neuron) ** 2  ##used in Bikun's version
    #Omega = Omega_a + (Omega_b - Omega_a) * torch.sigmoid(Omega_neuron)
    Omega= torch.sin(Omega_neuron) ** 2
    return Omega

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
f = Pinn(nodes_num=256,feature_num=78).to(device=mydevice)
# load model from pre-trained model
#f.load_state_dict(torch.load(pre_trained_model_path))
#f.eval()
# f.double()


# create optimizer: (ADAM)
learning_rate = 1e-5
#learning_rate = 1e-8
optimizer = torch.optim.Adam(params=f.parameters(), lr = learning_rate, eps=1e-16)
#check_optimizer_param()


# schedule the decay of lr by factor gamma every step_size:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

# Start the training loop
n_eom = 4096
#n_iters = 100
steps_array = []
loss_history = []
loss_history_separate=[]
start_time = time.time()  # start the clock

A=cal_evolution_A(A)
rot_0,rot_T=cal_evec_bc(U_target,T) #omega is a float from simpy_diff_geometry.py
"""for n in range(8):
    print('evec',n,':',rot_T[n])
    print('norm of evec',n,':',torch.linalg.norm(rot_T[n],dim=0))
    for m in range(n):
        print('evec', m, ':', rot_T[m])
        print('evec',n,'dot evec',m,':',torch.sum(rot_T[n]*rot_T[m],dim=0))"""

for i in range(1,args.Adam_niters+1):
    # reset learning rate every 10000 step
    #print("iter",i)
    if (i % 10000 == 0 and i != args.Adam_niters):
        optimizer = torch.optim.Adam(params=f.parameters(), lr=learning_rate, eps=1e-16)
        #optimizer = torch.optim.Adam(params=f.parameters(), lr=scheduler.get_last_lr()[-1], eps=1e-16)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    # empty the gradient:
    optimizer.zero_grad()
    s = sampling(n_eom, mydevice, include_bc=True)
    #Omega_ts=envelope_Omega(f, s, Omega_a, Omega_b)
    Omega1_ts = Omega_b * (2 / pi) * envelope(T / 2 * (s + 1).view(-1, 1)) * torch.atan(f(s)[:, 72].view(-1, 1)) * torch.sin(f(s)[:, 73].view(-1, 1))
    Omega2_ts = Omega_b * (2 / pi) * envelope(T / 2 * (s + 1).view(-1, 1)) * torch.atan(f(s)[:, 74].view(-1, 1)) * torch.sin(f(s)[:, 75].view(-1, 1))
    delta_ts = delta_max * (2 / pi) * envelope(T / 2 * (s + 1).view(-1, 1)) * torch.atan(f(s)[:, 76].view(-1, 1)) * torch.sin(f(s)[:, 77].view(-1, 1))
    ##sub A to be A_ts_
    for n in range(8):
        for l in range(8):
            #A_ts[n][l]=subs_Omega_by_ts(A[n][l],Omega_ts)
            A_ts[n][l] = subs_Control_by_ts(A[n][l], [Omega1_ts, Omega2_ts, delta_ts])
            #print('A_ts[',n,'][',l,']',A_ts[n][l])

    _l_eom,_l_dr,_l_norm,_l_dot = l_eom(f,s,A_ts,T,rot_0,rot_T, n_eom, mydevice)
    #_l_p=l_p(f, Omega_a, Omega_b, n_eom, mydevice)
    #_l_Omega=mse(Omega_ts,torch.zeros_like(Omega_ts).to(device=mydevice)) #penalize Omega of being too large
    _l_eom=_l_eom*args.leom_weight
    #print('loss',_l_eom,_l_dr,_l_norm,_l_dot,_l_p)
    l = _l_eom+_l_dr+_l_norm+_l_dot
    torch.set_printoptions(edgeitems=50)
    l.backward()
    # update the weights
    optimizer.step()
    # # schedule the decay of lr by factor gamma every step_size:
    scheduler.step()
    if (i==1 or i % 20 == 1) : # record the training loss every specific steps
        steps_array = np.append(steps_array, i)
        loss_history = np.append(loss_history, l.item())
        loss_history_separate = np.append(loss_history_separate, [_l_eom.item(),_l_dr.item(),_l_norm.item(),_l_dot.item()])
        #loss_history_separate = np.append(loss_history_separate, [_l_curv_test.item()])

    if (i==1 or i % 20 == 0):
        print("Step = %2d, loss = %.8f" % (i, l.item()))
        print('loss separate=',[l.item(),_l_eom.item(), _l_dr.item(), _l_norm.item(), _l_dot.item()])


# end the clock and print the elasped time
print("--- %s seconds ---" % (time.time() - initial_start_time))
 

# inference
# transfer the result from GPU to CPU
USE_CUDA = False
f.cpu()

s_test = torch.linspace(-1, 1, 1024).view(-1,1)
t_test = T*(s_test + 1)/2
# Data for a three-dimensional line
out_pred = f(s_test)
r_vec=(torch.ones_like(s_test)+s_test)/2*f(s_test)[:, 64:72]
#r_vec=(torch.ones_like(s_test)-torch.square(s_test))/4*f(s_test)[:, 64:72]
x_pred=r_vec[:,0].detach().numpy()
y_pred=r_vec[:,1].detach().numpy()
z_pred=r_vec[:,2].detach().numpy()
#Omega_pred = neuron_to_Omega(f(s_test)[:,72], Omega_a, Omega_b).detach().numpy()
mydevice = torch.device("cpu")
#Omega_pred=envelope_Omega(f,s_test,Omega_a,Omega_b).detach().numpy()
Omega1_pred = (Omega_b * (2 / pi) * envelope(T / 2 * (s_test + 1).view(-1, 1)) * torch.atan(
    f(s_test)[:, 72].view(-1, 1)) * torch.sin(f(s_test)[:, 73].view(-1, 1))).detach().numpy()
Omega2_pred = (Omega_b * (2 / pi) * envelope(T / 2 * (s_test + 1).view(-1, 1)) * torch.atan(
    f(s_test)[:, 74].view(-1, 1)) * torch.sin(f(s_test)[:, 75].view(-1, 1))).detach().numpy()
delta_pred = (delta_max * (2 / pi) * envelope(T / 2 * (s_test + 1).view(-1, 1)) * torch.atan(
    f(s_test)[:, 76].view(-1, 1)) * torch.sin(f(s_test)[:, 77].view(-1, 1))).detach().numpy()

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
ax.plot(t_test, Omega1_pred,'r',ms=2, label="Omega1")
ax.plot(t_test, Omega2_pred,'g',ms=2, label="Omega2")
ax.plot(t_test, delta_pred,'b',ms=2, label="delta")
# ax.set_title("predicted control field")
ax.legend(loc=3, prop={'size': 6})
#ax.set_ylim(Omega_a-0.1,0.2)
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

"""ax = fig.add_subplot(3, 3, 6)
ax.plot(t_test,x_pred,'-r',linewidth=1, label="x")
ax.plot(t_test,y_pred,'-g',linewidth=1, label="y")
ax.plot(t_test,z_pred,'-b',linewidth=1, label="z")
ax.legend(loc=2, prop={'size': 7})"""

"""ax = fig.add_subplot(3, 3, 7)
ax.plot(t_test, phi_pred,'r',ms=2, label="predicted control field Phi")
# ax.set_title("predicted control field")
ax.set_ylim(phi_a-0.1,phi_b+0.1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\Phi(t)$")
ax.set_title(r"$T={:.3f}$".format(T))
ax.set_xlabel(r"$t$")"""

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9, 
                    top=0.9,
                    wspace=0.7, 
                    hspace=1.5)

# saving:
if not os.path.exists(mypath):
    os.makedirs(mypath) # create the directory if it does not exist

# save the plotting    
plt.savefig(mypath+'/Plot_'+filename+'.pdf', dpi='figure')

plt.show()
# save the model
torch.save(f.state_dict(),mypath+'/'+filename+'model_dict.pt')


# save the loss history
np.savetxt(mypath+"/loss.txt", np.transpose([steps_array, loss_history]))
np.savetxt(mypath+"/loss.txt", np.hstack((np.transpose([steps_array, loss_history]),np.reshape(loss_history_separate,(-1,4)))))#start load env and run python
