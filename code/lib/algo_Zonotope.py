# PACKAGES:

from . import tools
import time
import numpy as np
import gurobipy as gp
# place additional package imports here


# FUNCTIONS:

def algo(data):
    """
    Alogrithm: Zonotope
    
    References:
    - Müller, F.L.; Sundström, O.; Szabó, J.; Lygeros, J. Aggregation of energetic flexibility using zonotopes. 2015 54th IEEE Conference 383 on Decision and Control (CDC), 2015, pp. 6564–6569. doi:10.1109/CDC.2015.7403253.
    - Müller, F.L.; Szabó, J.; Sundström, O.; Lygeros, J. Aggregation and Disaggregation of Energetic Flexibility From Distributed 385 Energy Resources. IEEE Transactions on Smart Grid 2019, 10, 1205–1214. Conference Name: IEEE Transactions on Smart Grid, 386 doi:10.1109/TSG.2017.2761439.
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
    Z,G = generateZonotope(T,[0]*T)
    C = getMatrixC(T) # calculate matrix of half-space representation Cx<=d
    Zonotope_list = []
    for b in b_list:
        d_new = getHyperplaneOffset(A,C,b,T) #calculate vector of half-space representation Cx<=d
        Z = optimalZonotopeMaxNorm(A,b,G,C,d_new) #calulate optimal center and scaling limits
        Zonotope_list.append(Z)
        
    # Calculate M-sum of zonotopes
    Zonotope_minkowski_list = []
    for l in range(len(Zonotope_list[0])):
        s = np.array(Zonotope_list[0][l])
        for h in range(1,len(Zonotope_list)):
            s = s + np.array(Zonotope_list[h][l])
        Zonotope_minkowski_list.append(list(s))
    
    b_approx = getVectord(C,Zonotope_minkowski_list,T)
    A_approx = C    
    algo_res['algo time'] = time.time() - t0  # stopping time of algo computations
    
    # objectives:
    demand_aggr = np.sum(demands, axis=1)
    for obj in data['objectives']:
        if obj == 'cost':
            t0 = time.time()
            model = gp.Model('Zonotopes, cost')
            model.Params.OutputFlag = 0
            x = model.addMVar(shape=T, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")  
            model.setObjective(prices@demand_aggr + prices@x, gp.GRB.MINIMIZE)
            model.addConstr( A_approx@x <= b_approx )
            model.optimize()
            algo_res[f"{obj}_time"] = time.time() - t0
            if model.status != 2: # 2 is optimal
                raise RuntimeError(f"LP 'Zonotope, cost' has status = {model.status}")
            else:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.dot(prices[:T_eval], demand_aggr[:T_eval] + x_sol[:T_eval])*dt
            algo_res[f"{obj}_im_en"] = np.NaN
        elif obj == 'peak':
            # optimization:
            t0 = time.time()
            model = gp.Model('Zonotopes, peak')
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
                raise RuntimeError(f"LP 'Zonotope, peak' has status = {model.status}")
            else:
                # slice solution and obj_value to evaluation time:
                x_sol = x.x
                algo_res[f"{obj}_value"] = np.linalg.norm(demand_aggr[:T_eval] + x_sol[:T_eval], ord=np.inf) 
            algo_res[f"{obj}_im_en"] = np.NaN  # imbalance energy
        else:
            raise ValueError(f'Error: objective "{obj}" is not implemented.')       
    return algo_res


def generateZonotope(T,c): # generates Zonotope Z(c,g_i,...g_p) and matrix of generators G
    G1 = np.eye(T)
    G2_1 = -1/np.sqrt(2)*np.eye(T,T-1)
    G2_2 = 1/np.sqrt(2)*np.eye(T,T-1,-1)
    G2 = G2_1 + G2_2
    G = np.column_stack((G1,G2))
    
    Z = [c]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:,i]))
    return Z,G

def getMatrixC(T): # calculates matrix of half-space representation
    N = np.eye(T)
    Bl = np.tril(np.ones([T,T]))
      
    for i in range(1,T):
        block = Bl[i,0:i+1]/np.linalg.norm(Bl[i,0:i+1])
        Nblock = np.zeros([T-i,T])
        
        for j in range(0,T-i):
            Nblock[j,j:j+i+1] = block
        
        N = np.concatenate([N,Nblock],axis=0)
    return np.concatenate([N,-N],axis=0)

def getHyperplaneOffset(A,C,b,dimension): # calculates vector of half-space representation
    m = np.shape(C)[0]
    d_list = []
    for i in range(m):
        model = gp.Model("Supporting Hyperplane")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape=dimension,lb=-gp.GRB.INFINITY,name="x")
        model.setObjective(C[i,:]@x,gp.GRB.MAXIMIZE)
        model.addConstr(A@x <= b)
        model.optimize()
        d_list.append(C[i,:]@x.X)
    return d_list

def optimalZonotopeMaxNorm(A,b,G,F,bi_list): # calculates optimal optimal center and vector of scaling limits, returns Z(c,g_i,...,g_p)
    AG = np.abs(A@G)
    W = np.abs(F[0:int(np.shape(F)[0]/2)]@G)
    delta_p = np.array(bi_list)
    W_aux = np.row_stack((W,W))

    model = gp.Model("Optimal Zonotope")
    model.Params.OutputFlag = 0
    t = model.addMVar(1,lb=0.0)
    c = model.addMVar(shape = np.shape(A)[1],lb=-gp.GRB.INFINITY)
    beta_bar = model.addMVar(shape = np.shape(G)[1],lb=0.0)
    model.setObjective(t,gp.GRB.MINIMIZE)
    for i in range(len(delta_p)):
        model.addConstr(-t <= delta_p[i]-(F[i,:]@c + W_aux[i,:]@beta_bar))
        model.addConstr(delta_p[i]-(F[i,:]@c + W_aux[i,:]@beta_bar) <= t)
    model.addConstr(AG@beta_bar + A@c <= b)
    model.optimize()

    c = c.X
    beta_bar = beta_bar.X

    Z = [list(c)]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:,i]*beta_bar[i]))
    return Z

def getVectord(C,Z,T): # calculates half-space representation vector from Z(c_i,g_i,...,g_p) representation
    p = 2*T-1 #number of generating directions
    c = Z[0]
    G_list = Z[1:]
    C = C[0:int(np.shape(C)[0]/2),:]
    delta_d_list = []
    for j in range(np.shape(C)[0]):
        delta_d = 0
        for i in range(p):
            delta_d = delta_d + np.abs(C[j,:]@np.array(G_list[i]))
        delta_d_list.append(delta_d)
    d_list = []
    for i in range(np.shape(C)[0]):
        d_list.append(C[i,:]@np.array(c)+np.array(delta_d_list[i]))
    for i in range(np.shape(C)[0]):
        d_list.append(-C[i,:]@np.array(c)+np.array(delta_d_list[i]))    
    d = np.array(d_list)
    return d