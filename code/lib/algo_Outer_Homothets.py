# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:06:43 2022

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
    Alogrithm: Outer Homothets
    
    References:
    - Zhao, L.; Zhang, W.; Hao, H.; Kalsi, K. A Geometric Approach to Aggregate Flexibility Modeling of Thermostatically Controlled 391 Loads. IEEE Transactions on Power Systems 2017, 32, 4721–4731. arXiv: 1608.04422, doi:10.1109/TPWRS.2017.267469
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
    """
    b_1 = np.ones(T)*(-np.mean(batts["x_min"]))
    b_2 = np.ones(T)*(np.mean(batts["x_max"]))
    b_3 = np.ones(T)*(np.mean(batts["S_max"])-np.mean(batts["S_0"]))/dt      
    b_4 = np.ones(T)*(np.mean(batts["S_0"])/dt)   
    b_mean = np.concatenate((b_1,b_2,b_3,b_4)) 
    """
    b_mean = np.mean(b_list,axis=0)
    
    beta_list, t_list = ([] for h in range(2))
    for b in b_list:
        beta,t = fitHomothet(A,b,b_mean,False,T) # caluclate optimal scaling factor and offset
        beta_list.append(beta)
        t_list.append(t)
    
    beta_sum = np.sum(beta_list,axis=0) # sum of scaling facotrs
    t_sum = np.sum(t_list,axis=0) # sum of offsets
    b_approx = b_mean*beta_sum + A@t_sum
    A_approx = A  
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            model = gp.Model('Homothets Outer, cost')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")  
            model.setObjective(prices@demand_aggr + prices@x, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.x
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt

            x_sol_eval = x_sol[:T_eval] # # slice solution of evaluation time
            A_eval, b_eval = tools.constraints_matrix_vector(T_eval, dt, batts)
            im_en = tools.imbalance_energy(x_sol_eval, A_eval, b_eval, dt)
            algo_res[f"{obj}_im_en"] = im_en # relative imbalance energy 
        elif obj == 'peak':
            t0 = time.time()
            model = gp.Model('Homothets Outer, peak')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
            m = model.addVar(lb=0.0, name='m')
            model.setObjective(m, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.addConstrs( (-m <= demand_aggr[t] + x[t]      for t in range(T)), name='left'  )
            model.addConstrs( (      demand_aggr[t] + x[t] <= m for t in range(T)), name='right' )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.x
            algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf) 
            
            x_sol_eval = x_sol[:T_eval]
            A_eval, b_eval = tools.constraints_matrix_vector(T_eval, dt, batts)
            im_en = tools.imbalance_energy(x_sol_eval, A_eval, b_eval, dt)
            algo_res[f"{obj}_im_en"] = im_en
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res


def fitHomothet(A,b,b_mean,inner,T): # calculates optimal offest and scaling factor
    aux_len = np.shape(b_mean)[0]
    if inner == True:
        model = gp.Model("MIA")
        model.Params.OutputFlag = 0
        s = model.addMVar(shape = 1)
        G = model.addMVar(shape = (aux_len,aux_len))
        r = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
        aux = model.addMVar(shape = aux_len,lb=-gp.GRB.INFINITY)
        model.setObjective(s,gp.GRB.MINIMIZE)
        for i in range(aux_len):
            model.addConstrs(G[i,:]@A[:,j] == A[i,j] for j in range(T))
        model.addConstrs(aux[i] == gp.quicksum(G[i,k]*b_mean[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b.reshape(aux_len,1)@s + A@r)
        model.optimize()    
        beta = 1/s.X
        t = -r.X/s.X
        return beta,t
    
    elif inner == False:
        model = gp.Model("MOA")
        model.Params.OutputFlag = 0
        s = model.addMVar(shape = 1)
        G = model.addMVar(shape = (aux_len,aux_len))
        r = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
        aux = model.addMVar(shape = aux_len,lb=-gp.GRB.INFINITY)
        model.setObjective(s,gp.GRB.MINIMIZE)
        for i in range(aux_len):
            model.addConstrs(G[i,:]@A[:,j] == A[i,j] for j in range(T))
        model.addConstrs(aux[i] == gp.quicksum(G[i,k]*b[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b_mean.reshape(aux_len,1)@s + A@r)
        model.optimize()
        beta = s.X
        t = r.X
        return beta,t