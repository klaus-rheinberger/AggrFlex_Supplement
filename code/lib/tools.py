# PACKAGES:

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp


# FUNCTIONS:

def constraints_matrix_vector(T, dt, batts):
    """
    A x_h <= b[:,h]
    A ... matrix of constraints
    b ... matrix (columnwise) of right hand side vectors
    """
    
    H = len(batts['S_max'])
    
    if False: # Klaus
        # make constraint matrix A w. r. t. T periods:
        I = np.eye(T)
        G = np.tril( np.ones( (T,T) ) )
        A = np.vstack( (I, -I, G, -G) )

        # make constraint vectors for all households:
        b = np.NaN*np.zeros( (4*T, H) )  # column = household
        for h in range(H):
            b1 =  np.ones(T)*batts['x_max'][h]
            b2 = -np.ones(T)*batts['x_min'][h]
            b3 =  np.ones(T)*( batts['S_max'][h] - batts['S_0'][h] )/dt
            b4     = -np.ones(T)*(       0.0                   - batts['S_0'][h] )/dt  # S_min = 0.0
            b4[-1] = -(batts['S_0'][h]*batts['S_end_min'][h]   - batts['S_0'][h] )/dt
            b[:,h] = np.hstack( (b1, b2, b3, b4) )
    else:  # Emrah
        # make constraint matrix A w. r. t. T periods:
        I = np.eye(T)
        G = np.tril( np.ones( (T,T) ) )
        A = np.vstack( (-I, I, G, -G) )

        # make constraint vectors for all households:
        b = np.NaN*np.zeros( (4*T, H) )  # column = household
        for h in range(H):
            b1 = -np.ones(T)*batts['x_min'][h]
            b2 =  np.ones(T)*batts['x_max'][h]
            b3 =  np.ones(T)*( batts['S_max'][h] - batts['S_0'][h] )/dt
            b4     = -np.ones(T)*(       0.0                   - batts['S_0'][h] )/dt  # S_min = 0.0
            b4[-1] = -(batts['S_0'][h]*batts['S_end_min'][h]   - batts['S_0'][h] )/dt
            b[:,h] = np.hstack( (b1, b2, b3, b4) )
        
    return A, b


def imbalance_energy(x_sol_eval, A_eval, b_eval, dt):
    
    T_eval = len(x_sol_eval)
    H = b_eval.shape[1]
    
    model = gp.Model('imbalance energy')
    model.Params.OutputFlag = 0
    x = model.addMVar(shape=(T_eval, H), lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
    a = model.addMVar(shape=T_eval, name='a')
    model.setObjective(a.sum(), sense=gp.GRB.MINIMIZE)  # or: gp.quicksum(a[t] for t in range(T_eval))
    model.addConstrs( (A_eval@x[:,h] <= b_eval[:,h] for h in range(H)) )
    model.addConstrs( (         gp.quicksum(x[t,h] for h in range(H)) - x_sol_eval[t] <= a[t] for t in range(T_eval)) )
    model.addConstrs( (-a[t] <= gp.quicksum(x[t,h] for h in range(H)) - x_sol_eval[t]         for t in range(T_eval)) )
    model.optimize()
    
    l1_norm_diff = model.objVal
    l1_norm_MS   = np.linalg.norm(x.x, ord=1)  # MS = Minkowski sum
    if np.abs(l1_norm_MS) > 1e-10:
        im_en = l1_norm_diff/l1_norm_MS*100.0  # percentage
    else:
        im_en = np.Infinity

    return im_en

def inner_obj_qc(row, comp):
    if row['no flexibility'] - row['exact'] > 1e-6:
        target = (row[comp['algo']] - row['exact'])/(row['no flexibility'] - row['exact'])*100.0  # percentage
        # print('-', end='')
    else:
        target = 0.0
        # print('0', end='')
    return target


def target_df(dsp, comp, res):
    
    if comp['view'] == 'qual. crit.':
        
        # cases:
        #     - inner and objective
        #     - outer and imbalance energy
        #     - duration

        if dsp['algo type'][ comp['algo'] ] == 'inner' and dsp['quantity type'][ comp['quantity'] ] == 'objective':
            # algo is inner approximation and quantity is of type objective:
            df = res.pivot(index=dsp['c4c'], columns='algo', values=comp['quantity']).reset_index()
            df.columns.name =''
            col_selection = dsp['c4c'] + [comp['algo']] + dsp['algos compare'] 
            df = df[col_selection]
            df['target'] = df.apply(lambda row : inner_obj_qc(row, comp), axis=1)
            df.drop(columns=[comp['algo']] + dsp['algos compare'], inplace=True)
            df.sort_values(by=dsp['c4c'], inplace=True)
        elif dsp['algo type'][ comp['algo'] ] == 'outer' and dsp['quantity type'][ comp['quantity'] ] == 'imbalance energy':
            # algo is outer approximation and quantity is of type imbalance energy:
            df = res.pivot(index=dsp['c4c'], columns='algo', values=comp['quantity']).reset_index()
            df.columns.name =''
            col_selection = dsp['c4c'] + [comp['algo']]
            df = df[col_selection]
            df['target'] = df[comp['algo']]
            df.drop(columns=[comp['algo']], inplace=True)
            df.sort_values(by=dsp['c4c'], inplace=True)
        elif dsp['quantity type'][ comp['quantity'] ] == 'duration': # seconds
            # quantity is of type duration:
            df = res.pivot(index=dsp['c4c'], columns='algo', values=comp['quantity']).reset_index()
            df.columns.name =''
            col_selection = dsp['c4c'] + [comp['algo']]
            df = df[col_selection]
            df['target'] = df[comp['algo']]
            df.drop(columns=[comp['algo']], inplace=True)
            df.sort_values(by=dsp['c4c'], inplace=True)
        else:
            print(comp)
            df = None
    
    elif comp['view'] == 'raw':
        df = res.pivot(index=dsp['c4c'], columns='algo', values=comp['quantity']).reset_index()
        df.columns.name =''
        col_selection = dsp['c4c'] + [comp['algo']]
        df = df[col_selection]
        df['target'] = df[comp['algo']]
        df.drop(columns=[comp['algo']], inplace=True)
        df.sort_values(by=dsp['c4c'], inplace=True)
        
    else:
        print(comp)
        df = None
    
    return df

def dsp_default():
    """
    settings for displays
    """
    
    dsp = {}
    
    dsp['algos compare']  = ['exact', 'no flexibility']
    # dsp['algos evaluate'] = ['Barot wo. pc.', 'Barot w. pc.']  # not yet needed
    dsp['algo type']      = {'Barot wo. pc.':'outer', 
                             'Barot w. pc.' :'outer',
                             'Homothet Stage 0':'inner',
                             'Outer Battery Homothet':'outer',
                             'Inner Battery Homothet':'inner',
                             'Homothet Projection':'inner',
                             'Union of Homothets Stage 1':'inner',
                             'Zonotopes':'inner',
                             'Zonotopes l1':'inner',
                             'Zonotopes l2':'inner',
                             'Zhen Ellipsoid Inner':'inner',
                             'Barot Ellipsoid Inner':'inner',
                             'Simple Inner':'inner',
                             'Klaus inner span':'inner'}
    dsp['type algos'] = {'inner':['Homothet Stage 0','Inner Battery Homothet', 'Homothet Projection', 'Union of Homothets Stage 1',
                                  'Zonotopes', 'Zonotopes l1', 'Zonotopes l2', 'Zhen Ellipsoid Inner', 'Barot Ellipsoid Inner'],
                         'outer':['Barot wo. pc.', 'Barot w. pc.', 'Outer Battery Homothet']}
    dsp['quantity type']  = {'cost_value':'objective', 
                             'peak_value':'objective',
                             'algo_time' :'duration', 
                             'cost_time' :'duration', 
                             'peak_time' :'duration',
                             'cost_im_en':'imbalance energy', 
                             'peak_im_en':'imbalance energy'}
    # units: 
    # - data: h, kW, kWh, EUR
    # - algo times: seconds
    dsp['quantity label'] = {'cost_value':{'raw':'costs (EUR)'    ,   'qual. crit.':'cost UPR (%)'}, 
                             'peak_value':{'raw':'peak power (kW)',   'qual. crit.':'peak power UPR (%)'},
                             'algo_time' :{'raw':'algo duration (s)', 'qual. crit.':'algo duration (s)'},
                             'cost_time' :{'raw':'cost duration (s)', 'qual. crit.':'cost duration (s)'},
                             'peak_time' :{'raw':'peak duration (s)', 'qual. crit.':'peak duration (s)'},
                             'cost_im_en':{'raw':        'cost IER (%)', 
                                           'qual. crit.':'cost IER (%)'},
                             'peak_im_en':{'raw':        'peak power IER (%)',
                                           'qual. crit.':'peak power IER (%)'}
                            }

    # columns for comparison:
    dsp['c4c'] = ['periods', 'households', 'day', 'sample'] 
    
    return dsp

def pareto(stg, res):
    
    dsp = dsp_default()
    
    ind = (res['periods'] == stg['periods']) & (res['households'] == stg['households'])
    red = res.loc[ind, ]
    algos = set(red['algo']).intersection(set(dsp['type algos'][stg['algo_type']]))
    algos = list(algos)

    comp = {}
    comp['view'] = 'qual. crit.'
    comp['quantity'] = stg['quantity']
    counter = 0
    for algo in algos:
        comp['algo'] = algo
        if counter == 0:
            df_qc = target_df(dsp, comp, red)
            df_qc.rename(columns={"target": algo}, inplace=True)
        else:
            df_add = target_df(dsp, comp, red)
            df_qc[algo] = df_add['target']
        counter += 1  
    df_qc = df_qc[algos].apply(func='median', axis=0)
    comp['quantity'] = 'algo_time'
    counter = 0
    for algo in algos:
        comp['algo'] = algo
        if counter == 0:
            df_at = target_df(dsp, comp, red)
            df_at.rename(columns={"target": algo}, inplace=True)
        else:
            df_add = target_df(dsp, comp, red)
            df_at[algo] = df_add['target']
        counter += 1  
    df_at = df_at[algos].apply(func='median', axis=0)

    pf = pd.concat( (df_qc, df_at), axis=1 )
    qual_crit = dsp['quantity label'][stg['quantity']]['qual. crit.']
    pf.columns = [qual_crit, 'algo time (seconds)']

    pf = pf.round(2).replace(to_replace=-0.00, value=0.0)
    pf.sort_values(by=list(pf.columns), inplace=True)

    if 0:
        pf.plot(y=qual_crit, x='algo time (seconds)', grid=True, figsize=(6,6),
                style='o', legend=False, ylabel=qual_crit)
    else:
        plt.figure(figsize=(9,6))
        for ind in pf.index:
            plt.plot(pf.loc[ind][1], pf.loc[ind][0], 'o', label=ind)
        plt.grid(True)
        plt.xlabel('algo time (seconds)')
        plt.ylabel(qual_crit)
        plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1));
    plt.tight_layout()
    
    return pf