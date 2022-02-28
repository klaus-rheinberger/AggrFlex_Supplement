# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:29:28 2022

@author: ozem
"""

# PACKAGES:

from . import tools
import time
import numpy as np
import cvxpy as cvx
import gurobipy as gp
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    Alogrithm: Barot Inner
    
    References:
    - S. Barot, “Aggregate load modeling for Demand Response via the Minkowski sum,” Thesis, University of Toronto, 2017, 2020. https://tspace.library.utoronto.ca/handle/1807/78943
    """
    
    # initialize results dictionary:
    algo_res = {}
    algo_res['sample'] = data['sample']
    
    # data:
    dt = data['dt']           # sampling time (h)
    T = len(data['demands'])  # number of all periods (>=100%)
    T_eval = data['periods']  # number of evaluation periods (100%)
    prices = data['prices']   # day-ahead prices: T-vector
    H = data['households']    # number of households
    demands = data['demands'] # demands matrix: TxH
    batts = data['batteries'] # batteries dictionary of h-vectors
    
    # make constraint matrix A and vector b w. r. t. T periods:
    A, b = tools.constraints_matrix_vector(T, dt, batts)
    b_list = list(b.T)
    
    
    # approximation:
    t0 = time.time()
    A_proj,b_proj = getAbProjection(A,b_list)
    m,n = np.shape(A_proj)
    B = cvx.Variable([n,n],symmetric=True)
    d = cvx.Variable([n,1])
    obj = cvx.Maximize(cvx.log_det(B[0:T:,0:T]))
    
    constraints = [B >> 0]
    constraints += [cvx.norm(B@A_proj[i,:].T) + A_proj[i,:]@d <= b_proj[i] for i in range(m)]
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    
    L = B.value[0:T,0:T]
    c = d.value[0:T]
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            L_inv = np.linalg.inv(L)
            F = L_inv.T@L_inv
            model = gp.Model("Approximation")
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            aux = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            model.setObjective(prices@g*dt,gp.GRB.MINIMIZE)
            model.addConstr(aux == x-c[:,0])
            model.addConstr(aux@F@aux<=1) # (x-c)^T F (x-c) <= 1 <=> x in ellipsoid
            model.addConstr(g == x + demand_aggr)
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.x
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt

            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            t0 = time.time()
            L_inv = np.linalg.inv(L)
            F = L_inv.T@L_inv
            model = gp.Model("Approximation")
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            aux = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            t = model.addVar(lb=0.0)
            model.setObjective(t,gp.GRB.MINIMIZE)
            model.addConstrs(-t <= g[i] for i in range(T))
            model.addConstrs(g[i] <= t for i in range(T))    
            model.addConstr(aux == x-c[:,0])
            model.addConstr(aux@F@aux<=1) # (x-c)^T F (x-c) <= 1 <=> x in ellipsoid
            model.addConstr(g == x + demand_aggr)
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.x
            algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf)
            
            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res


def getAbProjection(A,b_list): # gives the half-sapace representation of implicit M-sum
    A_barot_list = []
    A_barot = A
    for i in range(1,len(b_list)):
        A_barot = np.concatenate([A_barot,-A],axis=1)
    A_barot_list.append(A_barot)

    A_barot = np.concatenate([np.zeros([np.shape(A)[0],np.shape(A)[1]]),A],axis=1)
    A_barot = np.concatenate([A_barot,np.zeros((np.shape(A)[0],(len(b_list)-2)*np.shape(A)[1]))],axis=1)
    A_barot_list.append(A_barot)
    for i in range(1,len(b_list)-1):
        A_barot = np.zeros((np.shape(A)[0],(i+1)*np.shape(A)[1]))
        A_barot = np.concatenate([A_barot,A], axis=1)
        A_barot = np.concatenate([A_barot,np.zeros((np.shape(A)[0],(len(b_list)-i-2)*np.shape(A)[1]))],axis=1)
        A_barot_list.append(A_barot)
    
    A_barot = A_barot_list[0]
    for i in range(1,len(A_barot_list)):
        A_barot = np.concatenate([A_barot,A_barot_list[i]],axis=0)

    b_barot = b_list[-1]
    for i in range(0,len(b_list)-1):
        b_barot = np.concatenate([b_barot,b_list[i]])
    return A_barot,b_barot