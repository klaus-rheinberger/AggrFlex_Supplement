# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:37:00 2022

@author: ozem
"""

# PACKAGES:

from . import tools
import time
import numpy as np
import cvxpy as cp
import gurobipy as gp
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    Alogrithm: Union Homothets Stage 0
    
    References:
    - Nazir, M.S.; Hiskens, I.A.; Bernstein, A.; Dall’Anese, E. Inner Approximation of Minkowski Sums: A Union-Based Approach and 388 Applications to Aggregated Energy Resources. 2018 IEEE Conference on Decision and Control (CDC); IEEE: Miami Beach, FL, 389 2018; pp. 5708–5715. doi:10.1109/CDC.2018.8618731.
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
    A_hypercube = np.row_stack((np.eye(T),-np.eye(T)))
    xp_list_household = []
    xm_list_household = []
    b = np.array(b_list[0])
    xp,xm,r_list = MaxVolHypercubeFirst(A,b) #First Optimization (find prototype set)
    for b in b_list:
        xp,xm = MaxVolHypercubeHomothet(A,b,r_list)
        b_hypercube = np.array(list(xp)+list(-xm))
        
        xp_list,xm_list = [xp],[xm]
        if 0: #for second stage
            for a,b_k in zip(A_hypercube,b_hypercube):
                A_new = np.row_stack((A,-np.array(a)))
                b_new = np.array(list(b)+ [-b_k]) 
                xp_k,xm_k = MaxVolHypercubeHomothet(A_new,b_new,r_list)
                if xp_k is None:
                    xp_k = xp # if feasible region emtpy set, then use stage 0 solution                             
                    xm_k = xm
                xp_list.append(xp_k)
                xm_list.append(xm_k)
        xp_list_household.append(xp_list)
        xm_list_household.append(xm_list)
        
    # M-sum of Homothets (special case boxes)
    xp_sum_list,xm_sum_list = ([] for i_2 in range(2)) 
    for h in range(len(xp_list_household[0])):
        xp_sum,xm_sum = np.zeros(len(xp_list_household[0][0])),np.zeros(len(xp_list_household[0][0]))
        for xp_list,xm_list in zip(xp_list_household,xm_list_household):
            xp_sum = xp_sum + xp_list[h]
            xm_sum = xm_sum + xm_list[h]
        xp_sum_list.append(xp_sum)
        xm_sum_list.append(xm_sum)
    
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            min_list = []
            for xp,xm in zip(xp_sum_list,xm_sum_list):
                model = gp.Model("Approximation")
                model.Params.OutputFlag = 0
                x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
                g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
                model.setObjective(prices@g*dt,gp.GRB.MINIMIZE)
                
                b_hypercube = np.array(list(xp)+list(-xm))
                model.addConstr(A_hypercube@x <= b_hypercube)
                model.addConstr(g == x + demand_aggr)
                model.optimize()
                min_list.append(x.X)
                
            min_list_2 = [prices@(item + demand_aggr)*dt for item in min_list]
            z_index = min_list_2.index(min(min_list_2))
            z = min_list[z_index]
            algo_res[f"{obj}_time"] = time.time() - t0
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + z[:T_eval])*dt

            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            t0 = time.time()
            min_list = []
            for xp,xm in zip(xp_sum_list,xm_sum_list):
                model = gp.Model("Approximation")
                model.Params.OutputFlag = 0
                x = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
                g = model.addMVar(shape=T,lb=-gp.GRB.INFINITY)
                
                t = model.addVar(lb=0.0)
                model.setObjective(t,gp.GRB.MINIMIZE)
                model.addConstrs(-t <= g[i] for i in range(T))
                model.addConstrs(g[i] <= t for i in range(T))
                b_hypercube = np.array(list(xp)+list(-xm))
                model.addConstr(A_hypercube@x <= b_hypercube)
                model.addConstr(g == x + demand_aggr)
                model.optimize()
                min_list.append(x.X)
        
            min_list_2 = [np.linalg.norm(item + demand_aggr,np.inf) for item in min_list]
            z_index = min_list_2.index(min(min_list_2))
            z = min_list[z_index]
            algo_res[f"{obj}_time"] = time.time() - t0
            algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + z[:T_eval], ord=np.inf) 
            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res


def MaxVolHypercubeFirst(A,b): # calculates first max volume hypercube in polyhedron (prototype set)
    epsilon = 0.001
    m,n = np.shape(A) #Calculate A+ and A- Matrices
    Ap = np.zeros_like(A)
    Am = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            if A[i,j] > 0:
                Ap[i,j] = A[i,j]
            elif A[i,j] < 0:
                Am[i,j] = -A[i,j]

    xp = cp.Variable(n,name=("xp"))
    xm = cp.Variable(n,name=("xm"))
    constraints = [
        Ap@xp-Am@xm <= b,
        xm <= xp-epsilon] #enforce strict inequality

    volume = cp.sum(cp.log((xp-xm)))
    problem = cp.Problem(cp.Maximize(volume),constraints)
    try:
        problem.solve()
    except Exception:
        problem.solve(solver=cp.SCS)
        
    d = (xp.value-xm.value)
    r_list = [d[0]/d[k] for k in range(1,len(d))]
    return xp.value, xm.value, r_list

def MaxVolHypercubeHomothet(A,b,r_list): # calculates max valume homothets of prototype set
    epsilon = 0.001
    m,n = np.shape(A)
    Ap = np.zeros_like(A)
    Am = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            if A[i,j] > 0:
                Ap[i,j] = A[i,j]
            elif A[i,j] < 0:
                Am[i,j] = -A[i,j]

    xp = cp.Variable(n,name=("xp"))
    xm = cp.Variable(n,name=("xm"))
    
    contraitns = []
    for i in range(len(r_list)):
        constraints = contraitns + [xp[0]-xm[0] == r_list[i]*(xp[i+1]-xm[i+1])]
    constraints = constraints + [Ap@xp-Am@xm<=b,
                                 xm <= xp-epsilon] #enforce strict inequality

    volume = cp.sum(cp.log((xp-xm)))
    problem = cp.Problem(cp.Maximize(volume),constraints)
    
    try:
        problem.solve()
    except Exception:
        problem.solve(solver=cp.SCS)
        
    return xp.value, xm.value