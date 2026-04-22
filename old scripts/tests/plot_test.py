if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from Mip4cluster.src.mixture_lds.utils_MIP4Cluster import utils_MIP4Cluster
    
    name = "lds"
    method= ['if_Gurobi','if','em','fft','DTW'] #,'IF','EM','FFT','DTW'
    cutdown = False 
    utils_MIP4Cluster().plot_MIF4cluster_methods( f'./Result_{name}/', method, name, cutdown = False )
