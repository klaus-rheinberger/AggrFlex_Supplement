# PACKAGES:

from . import tools
import time
import numpy as np
import gurobipy as gp


# FUNCTIONS:

def algo(data):
    """
    exact Minkowski sum optimization
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
    # print(f"{T = }, {H = }, {len(batts['S_max'])}")
    
    # make constraint matrix A and vector b w. r. t. T periods:
    A, b = tools.constraints_matrix_vector(T, dt, batts)
    
    # approximation: nothing to compute, the Minkwoski sum is not computed
    algo_res['algo time'] = 0.0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            # optimization:
            t0 = time.time()
            model = gp.Model('exact Minkowski sum optimization, cost')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=(T,H), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
            model.setObjective(prices@demand_aggr + gp.quicksum(
                prices[t]*x[t,h] for h in range(H) for t in range(T) ), gp.GRB.MINIMIZE)
            model.addConstrs( (A@x[:,h] <= b[:,h] for h in range(H)) )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'exact Minkowski sum optimization, cost' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x.sum(axis=1)
                algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt
            # algo is not an outer approximation: 
            algo_res[f"{obj}_im_en"] = np.NaN
        elif obj == 'peak':
            # optimization:
            t0 = time.time()
            model = gp.Model('exact Minkowski sum optimization, peak')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=(T,H), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
            m = model.addVar(lb=0.0, name='m')
            model.setObjective(m, gp.GRB.MINIMIZE)
            model.addConstrs( (A@x[:,h] <= b[:,h] for h in range(H)) )
            model.addConstrs( (-m <= demand_aggr[t] + gp.quicksum(x[t,h] for h in range(H))      for t in range(T)), name='left'  )
            model.addConstrs( (      demand_aggr[t] + gp.quicksum(x[t,h] for h in range(H)) <= m for t in range(T)), name='right' )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'exact Minkowski sum optimization, peak' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x.sum(axis=1)
                algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.Inf)
            # algo is not an outer approximation: 
            algo_res[f"{obj}_im_en"] = np.NaN
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res
