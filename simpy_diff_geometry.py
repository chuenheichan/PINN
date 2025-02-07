from sympy import *
from sympy.core.numbers import Integer as SympyInteger
from sympy.physics.quantum import Commutator as Commu
import time
#import numpy as np
import torch
import math
init_printing(wrap_line=False)

init_start_time = time.time() # start the clock

ld1,ld2,ld3,ld4,ld5,ld6,ld7,ld8=symbols('ld1 ld2 ld3 ld4 ld5 ld6 ld7 ld8',commutative=False) #Gell Mann matrices
ld_sym_list=[ld1,ld2,ld3,ld4,ld5,ld6,ld7,ld8]
lambda_sub_list=[(ld1,Matrix([[0, 1, 0],[1, 0, 0],[0, 0, 0]])),(ld2,Matrix([[0, -I, 0],[I, 0, 0],[0, 0, 0]])),(ld3,Matrix([[1, 0, 0],[0, -1, 0],[0, 0, 0]])),(ld4,Matrix([[0, 0, 1],[0, 0, 0],[1, 0, 0]])),(ld5,Matrix([[0, 0, -I],[0, 0, 0],[I, 0, 0]])),(ld6,Matrix([[0, 0, 0],[0, 0, 1],[0, 1, 0]])),(ld7,Matrix([[0, 0, 0],[0, 0, -I],[0, I, 0]])),(ld8,1/sqrt(3)*Matrix([[1, 0, 0],[0, 1, 0],[0, 0, -2]]))]

Enormsq_sym=list(symbols('E:8normsq_sym')) #Symbols to be used during intermediate steps before evaluation
SD_sym = symbols(('SD0(:2)','SD1(:2)','SD2(:3)','SD3(:4)','SD4(:5)','SD5(:6)','SD6(:7)','SD7(:8)')) #Symbols to be used during intermediate steps before evaluation
t,a,b,c,d = symbols("t a b c d", real=True)
Omega1 = Function('Omega1', positive=True) #can be negative??
Omega2 = Function('Omega2', positive=True) #can be negative??
delta= Function('delta', positive=True) #can be negative??
Delta=-2*pi*0.2 #anharmonicity
H0= Omega1(t)/2*ld1 + sqrt(2)/2*Omega1(t)*ld6 +Omega2(t)/2*ld2 + sqrt(2)/2*Omega2(t)*ld7-delta(t)/2*ld3 -sqrt(3)/2*(delta(t)+2*Delta/3)*ld8
H1= -1/2*ld3 + -sqrt(3)/2*ld8 #H1=\deltaH in PRX

#EC=2.2g
#EJtoEC=100
#omega=(math.sqrt(8*EJtoEC)-1)*EC #assuming omega=energy splitting between 0 and 1st level
#H0=H0.subs([(a, (-1/2*delta(t)).evalf()), (b, (-sqrt(3)/2*(sqrt(8*EJtoEC)*EC)-20/12*EC-omega).evalf())]) ##need to be updated to possibly include small omega and the correct const value for transmon
#H1=H1.subs([(c, -1/2),(d,-sqrt(3)/2)])
print('H0',H0)
print('H1',H1)

#Structure constants for commulation relations of Gell Mann matrices
commu_list=[(Commu(ld1,ld2),2*I*ld3),(Commu(ld2,ld3),2*I*ld1),(Commu(ld1,ld3),-2*I*ld2),
(Commu(ld1,ld4),I*ld7),(Commu(ld4,ld7),I*ld1),(Commu(ld1,ld7),-I*ld4),
(Commu(ld1,ld5),-I*ld6),(Commu(ld5,ld6),-I*ld1),(Commu(ld1,ld6),I*ld5),
(Commu(ld2,ld4),I*ld6),(Commu(ld4,ld6),I*ld2),(Commu(ld2,ld6),-I*ld4),
(Commu(ld2,ld5),I*ld7),(Commu(ld5,ld7),I*ld2),(Commu(ld2,ld7),-I*ld5),
(Commu(ld3,ld4),I*ld5),(Commu(ld4,ld5),I*ld3+sqrt(3)*I*ld8),(Commu(ld3,ld5),-I*ld4),
(Commu(ld3,ld6),-I*ld7),(Commu(ld6,ld7),-I*ld3+sqrt(3)*I*ld8),(Commu(ld3,ld7),I*ld6),
(Commu(ld5,ld8),sqrt(3)*I*ld4),(Commu(ld4,ld8),-sqrt(3)*I*ld5),
(Commu(ld7,ld8),sqrt(3)*I*ld6),(Commu(ld6,ld8),-sqrt(3)*I*ld7),
(Commu(ld1,ld8),0),(Commu(ld2,ld8),0),(Commu(ld3,ld8),0)]

#Values of Omega(t) and its derivatives for evaluation
#omega_list=[(diff(Omega(t), t,6),1),(diff(Omega(t), t,5),1),(diff(Omega(t), t,4),1),(diff(Omega(t), t,3),1),(diff(Omega(t), t,2),1),(diff(Omega(t), t),1),(Omega(t),1)]


def ex(A):
    """Expansion of expression, only expand distributive law of basic multiplication"""
    return expand(A,multinomial=False,power_base=False, power_exp=False,log=False)


def SDot(A,B,expand=False):
    """Matrix dot product, defined by SDot(A,B)=1/2Tr(A*B) and used the relation Tr(\lambda_i*\lambda_j)=2\delta_ij
    Require A and B to be in the expanded form for coeff() to work"""
    if expand:
        A = ex(A)
        B = ex(B)
    return A.coeff(ld1)*B.coeff(ld1)+A.coeff(ld2)*B.coeff(ld2)+A.coeff(ld3)*B.coeff(ld3)+A.coeff(ld4)*B.coeff(ld4)\
    +A.coeff(ld5)*B.coeff(ld5)+A.coeff(ld6)*B.coeff(ld6)+A.coeff(ld7)*B.coeff(ld7)+A.coeff(ld8)*B.coeff(ld8)

def SNormsq(A,expand=False):
    """Norm Squared of matrix, defined by SNormsq(A)=1/2Tr(A*A)
    Require A to be in the expanded form for coeff() to work"""
    if expand:
        A = ex(A)
    return A.coeff(ld1)**2+A.coeff(ld2)**2+A.coeff(ld3)**2+A.coeff(ld4)**2\
    +A.coeff(ld5)**2+A.coeff(ld6)**2+A.coeff(ld7)**2+A.coeff(ld8)**2

def SCommu(A,B):
    """Evaluate commutation of two expressions involving Gell Mann matrices"""
    return Commu(A,B).expand(commutator=True,multinomial=False).expand(commutator=True,multinomial=False).subs(commu_list)

def subs_sym_by_eval(expr,i,omega_list,Esubs_list):
    """Substitute symbols in expr by omega_list and Esubs_list[k] for all k=1 to i
    In general, for an expr involving Evec[n], we need to set argument i= n for all its symbols to be substituted"""
    expr=expr.subs(omega_list)
    for k in range(1,i+1):
        expr=expr.subs(Esubs_list[k])
    return expr.evalf()

def subs_sym_by_ts(expr,i,Omega_lambdify_list,Omega_ts_list,Esubs_lambdify_list,Esubs_ts_list,lamb_module="math"):
    if len(expr.free_symbols)==0: ##check if there is any free symbols
        if isinstance(expr,SympyInteger):
            return int(expr)*torch.ones_like(Omega_ts_list[0])
        else:
            return float(expr)*torch.ones_like(Omega_ts_list[0])
    else:
        total_lambdify_list = Omega_lambdify_list.copy()
        total_Esubs_ts_list = Omega_ts_list.copy()
        for k in range(1, i + 1):
            total_lambdify_list.extend(Esubs_lambdify_list[k])
            total_Esubs_ts_list.extend(Esubs_ts_list[k])
        if lamb_module == "math":
            expr_lambdify = lambdify([total_lambdify_list], expr, "math")
        elif lamb_module == "sympy":
            expr_lambdify = lambdify([total_lambdify_list], expr, "sympy")
        return expr_lambdify(total_Esubs_ts_list)

def subs_Omega_by_ts(expr,Omega_ts):
    if len(expr.free_symbols)==0: ##check if there is any free symbols
        if isinstance(expr,SympyInteger):
            return int(expr)*torch.ones_like(Omega_ts)
        else:
            return float(expr)*torch.ones_like(Omega_ts)
    else:
        expr_lambdify = lambdify([Omega(t)], expr, "math")
    return expr_lambdify(Omega_ts)

def subs_Control_by_ts(expr,Control_ts_list):  ##subs Omega1,Omega2 and delta by ts, Control_ts_list need to be list of ts?
    if len(expr.free_symbols)==0: ##check if there is any free symbols
        # if expr contains no symbol, just return the const value with the shape of the first ts in Control_ts_list
        if isinstance(expr,SympyInteger):
            return int(expr)*torch.ones_like(Control_ts_list[0])
        else:
            return float(expr)*torch.ones_like(Control_ts_list[0]) #need to change
    else:
        expr_lambdify = lambdify([[Omega1(t),Omega2(t),delta(t)]], expr, "math")
    return expr_lambdify(Control_ts_list) #check if this needs to be list



def subs_sym_by_float(expr,i,Omega_lambdify_list,Omega_ft_list,Esubs_lambdify_list,Esubs_ft_list,lamb_module="math"):
    total_lambdify_list=Omega_lambdify_list.copy()
    total_Esubs_ft_list=Omega_ft_list.copy()
    for k in range(1,i+1):
        total_lambdify_list.extend(Esubs_lambdify_list[k])
        total_Esubs_ft_list.extend(Esubs_ft_list[k])
    if lamb_module=="math":
        expr_lambdify = lambdify([total_lambdify_list], expr, "math")
    elif lamb_module=="sympy":
        expr_lambdify = lambdify([total_lambdify_list], expr, "sympy")
    return expr_lambdify(total_Esubs_ft_list)

dG=[None]*8
dG_coeff=[[None for i in range(8)] for j in range(8)]
dGnormsq=[None]*8
dGnormsq_ts=[None]*8
Evec=[None]*8
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

dG[0] = H1/sqrt(SNormsq(H1))
Evec[0]=ex(dG[0])
Enormsq[0]=SNormsq(Evec[0]).cancel()
Enormsq_sym[0]=Enormsq[0] #Since Enormsq[0]=1 here, subs in the beginning to speed up
Enormsq_eval[0]=Enormsq[0] #Since Enormsq[0]=1 here, subs in the beginning to speed up

def cal_curv_sym(dG,dG_coeff,Evec,Enormsq,Curv,SD):
    dG[0] = ex(H1 / sqrt(SNormsq(H1)))
    Evec[0] = dG[0]
    Enormsq[0] = SNormsq(Evec[0]).cancel()
    for i in range(1,8):
        """PRX Eq10-13
        Keeping Enormsq and SD as symbols"""
        dG[i]=ex(I*SCommu(H0,dG[i-1])+diff(dG[i-1],t))
        Evec[i] = dG[i]
        for j in range(i):
            Evec[i]-= SD_sym[i][j]/Enormsq_sym[j] * Evec[j]
        Evec[i] = ex(Evec[i])
        Enormsq[i]=SNormsq(Evec[i])
        Curv[i-1]=SD_sym[i][i]/sqrt(Enormsq_sym[i-1]*Enormsq_sym[i])
        for j in range(i+1):
            SD[i][j]=SDot(dG[i], Evec[j])
    dG_coeff = cal_vec_coeff(dG,dG_coeff)
    print('subpage dG',dG)
    print('subpage dG_coeff',dG_coeff)
    return dG,dG_coeff,Evec,Enormsq,Curv,SD


#testing example
torch.manual_seed(4) #fixed the random seed # can be removed
Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
dOmega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
d2Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
d3Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
d4Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
d5Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)
d6Omega_ts = torch.rand(5,1, dtype=torch.float64, requires_grad=True)

#ls_ts=[Omega_ts,dOmega_ts]
#Omega_lambdify_list=[Omega(t),diff(Omega(t), t),diff(Omega(t), t,2),diff(Omega(t), t,3),diff(Omega(t), t,4),diff(Omega(t), t,5),diff(Omega(t), t,6)]
#Omega_ts_list=[Omega_ts,dOmega_ts,d2Omega_ts,d3Omega_ts,d4Omega_ts,d5Omega_ts,d6Omega_ts]

def cal_curv_value(dGnormsq,dGnormsq_ts,Enormsq,Enormsq_ts,Curv,Curv_ts,SD,SD_ts,Esubs_list,Esubs_lambdify_list,Esubs_ts_list,Omega_lambdify_list,Omega_ts_list):
    for i in range(1, 8):
        """Substitute Enormsq and SD by evaluated values"""
        Esubs_list[i] = []
        Esubs_lambdify_list[i] = []
        Esubs_ts_list[i] = []
        for j in range(i):
            SD_ts[i][j] = subs_sym_by_ts(SD[i][j], j, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,Esubs_ts_list)
            Esubs_lambdify_list[i].append(SD_sym[i][j])
            Esubs_ts_list[i].append(SD_ts[i][j])
        if i > 1:
            Esubs_lambdify_list[i].append(Enormsq_sym[i - 1])
            Esubs_ts_list[i].append(Enormsq_ts[i - 1])

        dGnormsq_ts[i]=subs_sym_by_ts(dGnormsq[i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
                                       Esubs_ts_list)
        Enormsq_ts[i] = subs_sym_by_ts(Enormsq[i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
                                       Esubs_ts_list)
        SD_ts[i][i] = subs_sym_by_ts(SD[i][i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
                                     Esubs_ts_list)

        total_lambdify_list = Omega_lambdify_list.copy()
        total_ts_list = Omega_ts_list.copy()
        if i > 1:
            total_lambdify_list.extend([SD_sym[i][i], Enormsq_sym[i - 1], Enormsq_sym[i]])
            total_ts_list.extend([SD_ts[i][i], Enormsq_ts[i - 1], Enormsq_ts[i]])
        elif i == 1:
            total_lambdify_list.extend([SD_sym[i][i], Enormsq_sym[i]])
            total_ts_list.extend([SD_ts[i][i], Enormsq_ts[i]])
        #total_ts_list.register_hook(lambda t: print(f'hook total_ts_list :\n {t}'))
        Curv_lambdify = lambdify([total_lambdify_list], Curv[i - 1], [{'sqrt': torch.sqrt}, 'math'])
        Curv_ts[i - 1] = Curv_lambdify(total_ts_list)

    return dGnormsq,dGnormsq_ts,Enormsq,Enormsq_ts,Curv,Curv_ts,SD,SD_ts,Esubs_list,Esubs_lambdify_list,Esubs_ts_list

#def cal_FS_bc(Evec,Enormsq_ts,U_a,U_target):#try to cal and return rot_0 and rot_T by this function
#Need to transform ts back to python var?
#use ts or sympy to sub numbers into Evec?
def get_bc(ts_subs_list): #extract the first and last element of each ts for the ts_subs_list
    rot_0_subs_list=[]
    rot_T_subs_list=[]
    for ts in ts_subs_list:
        rot_0_subs_list.append(ts[0].item())  # how to reduce fp error here from ts to float?
        rot_T_subs_list.append(ts[-1].item())
    return rot_0_subs_list,rot_T_subs_list

def get_bc_ts(ts_subs_list): #extract the first and last element of each ts for the ts_subs_list
    bc_ts_subs_list=[]
    for ts in ts_subs_list:
      bc_ts_subs_list.append(ts[[0,-1]].detach())
    return bc_ts_subs_list

def cal_vec_coeff(vec,vec_coeff):
    #Need vec to be 8X8 list
    print('vec',vec)
    for n in range(8):
        for l in range(8):
            print('vec[',n,']',vec[n])
            print(ld_sym_list[l])
            print(vec[n].coeff(ld_sym_list[l]))
            vec_coeff[n][l]=vec[n].coeff(ld_sym_list[l])

    return vec_coeff

def cal_curv_and_Evec_bc(dGnormsq,dGnormsq_ts,Evec,Evec_coeff,Evec_coeff_ts,Enormsq,Enormsq_ts,Curv,Curv_ts,SD,SD_ts,Esubs_list,Esubs_lambdify_list,Esubs_ts_list,Omega_lambdify_list,Omega_ts_list):
    bc_Esubs_ts_list = [None] * 8
    sub_start_time = time.time()  # start the clock
    Evec_coeff=cal_vec_coeff(Evec,Evec_coeff) ##can be optimized with procedure of cal Enormsq_sym, try later (take 1.5s per iteration)
    bc_Omega_ts_list= get_bc_ts(Omega_ts_list)
    for l in range(8):
        Evec_coeff_ts[0][l]=subs_sym_by_ts(Evec_coeff[0][l],0,Omega_lambdify_list,bc_Omega_ts_list,Esubs_lambdify_list=[],Esubs_ts_list=[])
    for i in range(1,8):
        """Substitute Enormsq and SD by evaluated values"""
        Esubs_list[i] = []
        Esubs_lambdify_list[i] = []
        Esubs_ts_list[i] = []
        for j in range(i):
            SD_ts[i][j] = subs_sym_by_ts(SD[i][j], j, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,Esubs_ts_list)
            Esubs_lambdify_list[i].append(SD_sym[i][j])
            Esubs_ts_list[i].append(SD_ts[i][j])
        if i > 1:
            Esubs_lambdify_list[i].append(Enormsq_sym[i - 1])
            Esubs_ts_list[i].append(Enormsq_ts[i - 1])
        #sub_start_time = time.time()  # start the clock
        #dGnormsq_ts[i]=subs_sym_by_ts(dGnormsq[i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
        #                               Esubs_ts_list)
        #print("dGnormsq_ts[",i,"] time elapsed:--- %s seconds ---" % (time.time() - sub_start_time))

        Enormsq_ts[i] = subs_sym_by_ts(Enormsq[i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
                                       Esubs_ts_list)

        SD_ts[i][i] = subs_sym_by_ts(SD[i][i], i, Omega_lambdify_list, Omega_ts_list, Esubs_lambdify_list,
                                     Esubs_ts_list)
        bc_Esubs_ts_list[i]=get_bc_ts(Esubs_ts_list[i])
        for l in range(8):
            Evec_coeff_ts[i][l]=subs_sym_by_ts(Evec_coeff[i][l], i, Omega_lambdify_list, bc_Omega_ts_list, Esubs_lambdify_list,
                                       bc_Esubs_ts_list)
        total_lambdify_list = Omega_lambdify_list.copy()
        total_ts_list = Omega_ts_list.copy()
        if i > 1:
            total_lambdify_list.extend([SD_sym[i][i], Enormsq_sym[i - 1], Enormsq_sym[i]])
            total_ts_list.extend([SD_ts[i][i], Enormsq_ts[i - 1], Enormsq_ts[i]])
        elif i == 1:
            total_lambdify_list.extend([SD_sym[i][i], Enormsq_sym[i]])
            total_ts_list.extend([SD_ts[i][i], Enormsq_ts[i]])
        Curv_lambdify = lambdify([total_lambdify_list], Curv[i - 1], [{'sqrt': torch.sqrt}, 'math'])
        Curv_ts[i - 1] = Curv_lambdify(total_ts_list)
    return dGnormsq,dGnormsq_ts,Evec_coeff_ts,Enormsq,Enormsq_ts,Curv,Curv_ts,SD,SD_ts,Esubs_list,Esubs_lambdify_list,Esubs_ts_list

def cal_FS_bc(Evec_coeff_ts,Enormsq_ts,omega,U_target,T):
    #here omega is a float, not optimizable from neuron, to speed up training in the beginning
    ld_list = [Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
               Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]), Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
               Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]), Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
               Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]), 1 / sqrt(3) * Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]])]
    rot_0 = torch.zeros(8, 8,requires_grad=False)
    rot_T = torch.zeros(8, 8,requires_grad=False)
    for n in range(8):
        for l in range(8):
            if n==0: #since Enormsq[0]=1
                #rot_0[n, l] = torch.Tensor([Evec_eval0_list[n].coeff(ld_sym_list[l])])
                rot_0[n,l] =Evec_coeff_ts[n][l][0]
            else:
                rot_0[n,l]=1 / torch.sqrt(Enormsq_ts[n][0].detach())*Evec_coeff_ts[n][l][0]
    U_a = diag(exp(I * omega*T), 1, exp(-I * omega*T))
    R_T = conjugate(U_a.T) * U_target
    R_T_conj = conjugate(U_target.T) * U_a
    Evec_evalT_matrix_list= [zeros(3,3)] * 8
    for n in range(8):
        for l in range(8):
            #Construct each Evec_n in sympy matrix
            Evec_evalT_matrix_list[n] +=Evec_coeff_ts[n][l][-1].item()*ld_list[l]
    for n in range(8):
        for l in range(8):
            _rot_T_nl = 1 / 2 *Trace(ld_list[l] * R_T_conj * Evec_evalT_matrix_list[n] * R_T).doit()
            if n==0: #since Enormsq[0]=1
                rot_T[n, l] = torch.Tensor([re(_rot_T_nl)])
            else:
                rot_T[n, l] = torch.Tensor([1 / sqrt(Enormsq_ts[n][-1]) * re(_rot_T_nl)])
            # Check coeff of GM matrices is real, since R_T_conj * Evec_evalT_list[n] * R_T is traceless Hermitian:
            if im(_rot_T_nl)>1e-10:
                print("Warning: coefficients of GM matrices are not real")
    return rot_0,rot_T


def cal_evolution_A(A):
    #A is a 8X8 list
    for n in range(8):
        commu = ex(I * SCommu(H0, ld_sym_list[n]))
        #print(n,commu)
        for l in range(8):
            A[n][l] = commu.coeff(ld_sym_list[l])
            #print(n,l,A[n][l])
    return A

def cal_evec_bc_labframe(omega,U_target,T):
    # Here assume U_target is in lab frame
    # here omega is a float, not optimizable from neuron, to speed up training in the beginning
    ld_list = [Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
               Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]), Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
               Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]), Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
               Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]), 1 / sqrt(3) * Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]])]
    rot_0 = torch.eye(8, 8, requires_grad=False)
    rot_T = torch.zeros(8, 8, requires_grad=False)

    U_a = diag(exp(I * omega * T), 1, exp(-I * omega * T))
    R_T = conjugate(U_a.T) * U_target
    R_T_conj = conjugate(U_target.T) * U_a
    Evec_evalT_matrix_list = [zeros(3, 3)] * 8
    for n in range(8):
        for l in range(8):
            _rot_T_nl = 1 / 2 * Trace(ld_list[l] * R_T_conj * ld_list[n] * R_T).doit()
            rot_T[n, l] = torch.Tensor([re(_rot_T_nl)])
            # Check coeff of GM matrices is real, since R_T_conj * Evec_evalT_list[n] * R_T is traceless Hermitian:
            if im(_rot_T_nl) > 1e-10:
                print("Warning: coefficients of GM matrices are not real")
    return rot_0, rot_T

def cal_evec_bc(U_target,T):
    # Here assume U_target is in rotating frame (i.e. qubit is encoded in rotating frame)
    # here omega is a float, not optimizable from neuron, to speed up training in the beginning
    ld_list = [Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
               Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]), Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
               Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]), Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
               Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]), 1 / sqrt(3) * Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]])]
    rot_0 = torch.eye(8, 8, requires_grad=False)
    rot_T = torch.zeros(8, 8, requires_grad=False)

    U_a = eye(3)
    R_T = conjugate(U_a.T) * U_target
    R_T_conj = conjugate(U_target.T) * U_a
    Evec_evalT_matrix_list = [zeros(3, 3)] * 8
    for n in range(8):
        for l in range(8):
            _rot_T_nl = 1 / 2 * Trace(ld_list[l] * R_T_conj * ld_list[n] * R_T).doit()
            rot_T[n, l] = torch.Tensor([re(_rot_T_nl)])
            # Check coeff of GM matrices is real, since R_T_conj * Evec_evalT_list[n] * R_T is traceless Hermitian:
            if im(_rot_T_nl) > 1e-10:
                print("Warning: coefficients of GM matrices are not real")
    return rot_0, rot_T

"""A = [[None for i in range(8)] for j in range(8)]
A=cal_evolution_A(A)
print(A)

U_target=Matrix([[0, 1, 0], [1, 0, 0],[0,0,0]])
rot_0, rot_T=cal_evec_bc(omega,U_target,100)
print(rot_0)
print(rot_T)"""
