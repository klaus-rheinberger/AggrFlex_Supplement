# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:23:20 2022

@author: ozem
"""

# PACKAGES:

from . import tools
import time
import numpy as np
import gurobipy as gp
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    Alogrithm: Homothet Projection
    
    References:
    - Zhao, L.; Hao, H.; Zhang, W. Extracting Flexibility of Heterogeneous Deferrable Loads via Polytopic Projection Approximation. 397 arXiv:1609.05966 [math] 2016. arXiv: 1609.05966.
    """
    
    # initialize results dictionary:
    algo_res = {}
    algo_res['sample'] = data['sample']
    
    # data:
    dt = data['dt']           # sampling time (h)
    T = len(data['demands'])  # number of all periods (>=100%)
    T_eval = data['periods']  # number of evaluation periods (100%)
    prices = data['prices']   # day-ahead prices: T-vector
    households = data['households']    # number of households
    demands = data['demands'] # demands matrix: TxH
    batts = data['batteries'] # batteries dictionary of h-vectors
    
    # make constraint matrix A and vector b w. r. t. T periods:
    A, b = tools.constraints_matrix_vector(T, dt, batts)
    b_list = list(b.T)
    
    # approximation:
    t0 = time.time()
    B,b_p = getAbProjection(A,b_list)
    """
    b_1 = np.ones(T)*(-np.mean(batts["x_min"]))
    b_2 = np.ones(T)*(np.mean(batts["x_max"]))
    b_3 = np.ones(T)*(np.mean(batts["S_max"])-np.mean(batts["S_0"]))/dt
    b_4 = np.ones(T)*(np.mean(batts["S_0"])/dt)
    H = np.concatenate((b_1,b_2,b_3,b_4))
    """
    H = np.mean(b_list,axis=0)

    beta,t = fitHomothetProjectionLinDescisionRule(A,H,B,b_p,T,households)
    b_approx = beta*H + A@t
    A_approx = A
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            model = gp.Model("Approximation, cost")
            model.Params.OutputFlag = 0
            x = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
            model.setObjective(prices@g*dt,gp.GRB.MINIMIZE)
            model.addConstr(A_approx@x <= b_approx)
            model.addConstr(g == x + demand_aggr)
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            
            x_sol = x.X
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt

            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            t0 = time.time()
            model = gp.Model("Approximation, peak")
            model.Params.OutputFlag = 0
            x = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
            t = model.addVar(lb=0.0)
            model.setObjective(t,gp.GRB.MINIMIZE)
            model.addConstrs(-t <= g[i] for i in range(T))
            model.addConstrs(g[i] <= t for i in range(T))
            model.addConstr(A_approx@x <= b_approx)
            model.addConstr(g == x + demand_aggr)
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            
            x_sol = x.X
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

def fitHomothetProjectionLinDescisionRule(F,H,B,c,dimension,households): # calculates optimal scaling factor and offset
    I = np.eye(dimension)
    rows_B = 4*dimension*households
    model = gp.Model("MIA")
    model.Params.OutputFlag = 0
    s = model.addMVar(shape = 1)
    G = model.addMVar(shape = (rows_B,4*dimension))
    r = model.addMVar(shape = dimension,lb=-gp.GRB.INFINITY)
    V = model.addMVar(shape = dimension*households-dimension,lb=-gp.GRB.INFINITY)
    aux = model.addMVar(shape = rows_B,lb=-gp.GRB.INFINITY)
    aux_IW = model.addMVar(shape = (dimension*households,dimension),lb=-gp.GRB.INFINITY)
    aux_rV = model.addMVar(shape = dimension*households,lb=-gp.GRB.INFINITY)
    
    model.setObjective(s,gp.GRB.MINIMIZE)
    model.addConstrs(aux_IW[i,j] == I[i,j] for i in range(dimension) for j in range(dimension))
    model.addConstr(aux_rV[0:dimension] == r)
    model.addConstr(aux_rV[dimension:] == -V)
    for i in range(rows_B):
        model.addConstrs(G[i,:]@F[:,j] == B[i,:]@aux_IW[:,j] for j in range(dimension))
    model.addConstrs(aux[i] == gp.quicksum(G[i,k]*H[k] for k in range(4*dimension)) for i in range(rows_B))
    model.addConstr(aux <= c.reshape(rows_B,1)@s + B@aux_rV)
    model.optimize()    
    beta = 1/s.X
    t = -r.X[0:dimension]/s.X
    return beta,t