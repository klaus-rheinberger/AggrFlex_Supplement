# PACKAGES:

from . import tools
import time
import numpy as np
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    template
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
    # TODO: compute possibly using subfunctions
    y = sub_function(1)
    
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            # TODO: optimization: compute possibly using subfunctions

            # TODO: fill results dictionary
            #       slice obj_value to evaluation time

            algo_res[f"{obj}_value"] = np.NaN
            algo_res[f"{obj}_time"]  = np.NaN  # computation time


            # TODO: uncomment to fill results dictionary
            # x_sol_eval = x_sol[:T_eval] # # slice solution of evaluation time
            # A_eval, b_eval = tools.constraints_matrix_vector(T_eval, dt, batts)
            # im_en = tools.imbalance_energy(x_sol_eval, A_eval, b_eval, dt)
            # algo_res[f"{obj}_im_en"] = im_en # relative imbalance energy 
            
            # if algo is not an outer approximation: 
            algo_res[f"{obj}_im_en"] = np.NaN 
        elif obj == 'peak':
            # TODO: analogous to the previous objective
            
            
            
            algo_res[f"{obj}_value"] = np.NaN
            algo_res[f"{obj}_time"]  = np.NaN  # computation time
            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')
        
    return algo_res


def sub_function(x):
    y = x**2
    return y