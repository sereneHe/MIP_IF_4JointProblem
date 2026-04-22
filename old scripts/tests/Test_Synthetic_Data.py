if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    from pathlib import Path
    #sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from Mip4cluster.src.mixture_lds.MIP4Cluster import MIP4Cluster
    from Mip4cluster.src.mixture_lds.utils_MIP4Cluster import utils_MIP4Cluster

    # hypeparameters
    S_len, I_len, T_len, F_len =10, 16, 20, 2
    J = 5
    name = "lds"
    hiddenstates =[2,3,4]
    windows = [3]
    feature_selection = [2]
    # regularization set
    method= ['IF-Gurobi']
     
    reg = np.array([270, 20])
        
    # Generate Data
    # data_dir=utils_MIP4Cluster().data_generation(g,f_dash,pro_rang,obs_rang,T_len,S_len)
    data, label = utils_MIP4Cluster().datacleaning([], name, S_len, I_len, T_len, F_len, J)
    method= ['IF-Gurobi']
    #,'IF','EM','FFT','DTW'
    window = np.array([[w] for w in windows])# Timeset
    F = np.array([[f] for f in feature_selection])# Feature
    N = np.array([[n] for n in hiddenstates])
    for i in range(data.shape[0]):
        num =2+i
        data_X = data[i]
        MIP4Cluster().Test_MIP4Cluster(data_X, label, method, S_len, I_len, window, F, f"lds{num}", N, J, reg, thresh=0.25) 
    utils_MIP4Cluster().plot_MIF4cluster_methods( f'./Result_{name}/', method, name, cutdown = False )
