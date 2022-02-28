# PACKAGES:

from . import tools
import time
import numpy as np
import gurobipy as gp


# FUNCTIONS:

def algo(data):
    """
    simple inner approximation
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
    
    # approximation:
    t0 = time.time()
    A_approx = A
    b_approx = b[:,0] # first household battery
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            # optimization:
            t0 = time.time()
            model = gp.Model('simple inner, cost')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")  
            model.setObjective(prices@demand_aggr + prices@x, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'simple inner, cost' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt 
            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            # optimization:
            t0 = time.time()
            model = gp.Model('simple inner, peak')
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
                raise RuntimeError(f"LP 'simple inner, peak' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.Inf) 
            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res


def sub_function(x):
    y = x**2
    return y