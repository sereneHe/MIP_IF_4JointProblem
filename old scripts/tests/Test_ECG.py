# Test Real Data time window-2,3,4,5,6
if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    from pathlib import Path
    #sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from Mip4cluster.src.mixture_lds.MIP4Cluster import MIP4Cluster
    from Mip4cluster.src.mixture_lds.utils_MIP4Cluster import utils_MIP4Cluster

    S_len, I_len, T_len, F_len =1, 15, 140, 1
    J = 2
    name = "ecg"
    hiddenstates =[2,3,4]
    windows = [2,3,4,5,6]
    feature_selection = [1]
    reg = np.array([270]) # regularization set
        
    # Generate Data
    # data_dir=utils_MIP4Cluster().data_generation(g,f_dash,pro_rang,obs_rang,T_len,S_len)
    data, label = utils_MIP4Cluster().datacleaning([], name, S_len, I_len, T_len, F_len, J)
    method= ['IF-Gurobi']
    #'IF','EM','FFT','DTW'
    window = np.array([[w] for w in windows])# Timeset
    F = np.array([[f] for f in feature_selection])# Feature
    N = np.array([[n] for n in hiddenstates])
    MIP4Cluster().Test_MIP4Cluster(data, label, method, S_len, I_len, window, F, name, N, J, reg, thresh=0.25) 
