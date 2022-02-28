# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:27:05 2022

@author: ozem
"""

from . import tools
import time
import numpy as np
import gurobipy as gp
# place additional package imports here


# FUNCTIONS:

def algo(data):
    
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
    
    t0 = time.time()
    vertex_list = getVertices(A,b_list,T,H)        
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            H = np.array(vertex_list).T
            model = gp.Model("Approximation")
            model.Params.OutputFlag = 0
            dim_H = np.shape(H)[1]
            alpha = model.addMVar(shape=dim_H,lb=-gp.GRB.INFINITY)
            x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            model.setObjective(prices@g*dt,gp.GRB.MINIMIZE)
            model.addConstr(H@alpha == x)
            model.addConstr(g == x + demand_aggr)
            model.addConstr(sum(alpha) == 1)
            model.addConstrs(alpha[i] >= 0 for i in range(dim_H))
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.X
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt
 
            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            t0 = time.time()
            H = np.array(vertex_list).T
            model = gp.Model("Approximation")
            model.Params.OutputFlag = 0
            dim_H = np.shape(H)[1]
            alpha = model.addMVar(shape=dim_H,lb=-gp.GRB.INFINITY)
            x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            t = model.addVar(lb=0.0)
            model.setObjective(t,gp.GRB.MINIMIZE)
            model.addConstrs(-t <= g[i] for i in range(T))
            model.addConstrs(g[i] <= t for i in range(T))
            model.addConstr(H@alpha == x)
            model.addConstr(g == x + demand_aggr)
            model.addConstr(sum(alpha) == 1)
            model.addConstrs(alpha[i] >= 0 for i in range(dim_H))
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            x_sol = x.X
            algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf) 

            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res

def getVertices(A,b_list,dimension,households):
    vertex_list = []
    c_array = np.vstack((np.eye(dimension),-np.eye(dimension)))
    G2_1 = -1/np.sqrt(2)*np.eye(dimension,dimension-1)
    G2_2 = 1/np.sqrt(2)*np.eye(dimension,dimension-1,-1)
    G2 = G2_1 + G2_2
    c_array = np.vstack((c_array,G2.T))
    for c in c_array:
        model = gp.Model("Approximation")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape=(dimension,households),lb=-gp.GRB.INFINITY)
        model.setObjective(sum(c@x[:,i] for i in range(households)),gp.GRB.MINIMIZE)
        for i in range(households):
            model.addConstr(A@x[:,i] <= b_list[i])
        model.optimize()
        vertex_list.append(x.X.sum(axis=1))
    vertex_list = np.unique(vertex_list, axis=0)
    return vertex_list

