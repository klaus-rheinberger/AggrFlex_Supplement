# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:06:09 2022

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
    Alogrithm: Barot outer aproximation with preconditioning
    
    References:
    - S. Barot and J. A. Taylor, “A concise, approximate representation of a collection of loads described by polytopes,” International Journal of Electrical Power & Energy Systems, vol. 84, pp. 55–63, Jan. 2017, doi: 10.1016/j.ijepes.2016.05.001.
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
    b_list = pc(A,b_list,T)
    A_approx = A
    b_approx = np.sum(b_list,axis=0)
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations

    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            model = gp.Model('Barot with preconditioning, cost')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")  
            model.setObjective(prices@demand_aggr + prices@x, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'Barot without preconditioning, cost' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt
                # imbalance energy w. r. t. evaluation time:
                x_sol_eval = x_sol[:T_eval]
                A_eval, b_eval = tools.constraints_matrix_vector(T_eval, dt, batts)
                im_en = tools.imbalance_energy(x_sol_eval, A_eval, b_eval, dt)
                algo_res[f"{obj}_im_en"] = im_en
        elif obj == 'peak':
            # optimization:
            t0 = time.time()
            model = gp.Model('Barot without preconditioning, peak')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
            m = model.addVar(lb=0.0, name='m')
            model.setObjective(m, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.addConstrs( (-m <= demand_aggr[t] + x[t]      for t in range(T)), name='left'  )
            model.addConstrs( (      demand_aggr[t] + x[t] <= m for t in range(T)), name='right' )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'Barot without preconditioning, peak' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf) 
                # imbalance energy w. r. t. evaluation time:
                x_sol_eval = x_sol[:T_eval]
                A_eval, b_eval = tools.constraints_matrix_vector(T_eval, dt, batts)
                im_en = tools.imbalance_energy(x_sol_eval, A_eval, b_eval, dt)
                algo_res[f"{obj}_im_en"] = im_en
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res



def pc(A,b_list,T): # preconditioning
    b_new_list = []
    for b in b_list:
        b_i_list = []
        for a in A:
            model = gp.Model("Preconditioning")
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
            model.setObjective(a@x,gp.GRB.MAXIMIZE)
            model.addConstr(A@x <= b)
            model.optimize()
            b_i_list.append(a@x.X)
        b_new_list.append(np.array(b_i_list))
    return b_new_list
    
        
        
    
    
    
    
    
    