# PACKAGES:

from . import tools
import time
import numpy as np
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    x = 0 for benchmarking
    """
    
    # initialize results dictionary:
    algo_res = {}
    algo_res['sample'] = data['sample']
    
    # data:
    dt = data['dt']           # sampling time (h)
    # T = len(data['demands'])  # number of all periods (>=100%)
    T_eval = data['periods']  # number of evaluation periods (100%)
    prices = data['prices']   # day-ahead prices: T-vector
    # H = data['households']    # number of households
    demands = data['demands'] # demands matrix: TxH
    # batts = data['batteries'] # batteries dictionary of h-vectors
    
    # make constraint matrix A and vector b w. r. t. T periods:
    # A, b = tools.constraints_matrix_vector(T, dt, batts)
    
    # approximation: nothing to compute
    algo_res['algo time'] = 0.0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    x_sol = np.zeros_like(demand_aggr)
    for obj in data['objectives']:
        if obj == 'cost':
            # optimization: nothing to compute
            algo_res[f"{obj}_time"] = 0.0  # computation time
            # slice obj_value to evaluation time
            algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt
            # algo is not an outer approximation: 
            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            # optimization: nothing to compute
            algo_res[f"{obj}_time"] = 0.0  # computation time
            # slice obj_value to evaluation time
            algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.Inf)
            # algo is not an outer approximation: 
            algo_res[f"{obj}_im_en"] = np.NaN 
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')

    return algo_res
