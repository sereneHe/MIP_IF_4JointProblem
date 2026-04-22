if __name__ == '__main__':
    import os
    import shutil
    import sys
    import numpy as np
    from pathlib import Path
    #sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from Mip4cluster.src.mixture_lds.MIP4Cluster import MIP4Cluster
    from Mip4cluster.src.mixture_lds.utils_MIP4Cluster import utils_MIP4Cluster

    project_dir = Path(__file__).resolve().parents[1]
    os.chdir(project_dir)
    synthetic_data_dir = project_dir / "data" / "sythetic"
    synthetic_data_dir.mkdir(parents=True, exist_ok=True)

    # Self-setting
    T_len=20
    S_len=10
    # Collect the nrmse value for each experiment
    pro_rang = np.arange(0.02,0.1,0.02)
    obs_rang = np.arange(0.02,0.1,0.02)

    g = np.array([[0.8*np.matrix([[0.9,0.2],[0.1,0.1]]), 0.8*np.matrix([[0.8,0.2],[0.2,0.1]])],
                  [0.6*np.matrix([[1.0,0.8,0.8],[0.6,0.1,0.2],[0.3,0.2,0.2]]), 0.6*np.matrix([[1.0,1.0,0.6],[0.7,0.2,0.2],[0.2,0.1,0.1]])],
                  [np.matrix([[0.9,0.8,0.5,0.2],[0.9,0.1,0.3,0.4],[0.8,0.2,0.1,0.1],[0.1,0.1,0.1,0.7]])*0.4, np.matrix([[1.0,0.8,0.5,0.3],[0.6,0.2,0.3,0.4],[0.8,0.2,0.3,0.1],[0.2,0.2,0.3,0.7]])*0.4]], dtype=object)

    f_dash = {0: [0.8 * np.array([[1.0, 1.0], [0.2, 0.2]]), 0.8 * np.array([[0.8, 1.0], [0.1, 0.2]])],
              1: [0.6 * np.array([[0.7, 0.4, 0.3], [0.2, 0.6, 0.2]]), 0.6 * np.array([[0.5, 0.4, 0.1], [0.2, 0.5, 0.1]])],
              2: [np.array([[0.2, 0.5, 0.1, 0.1], [0.8, 0.6, 0.1, 0.1]]) * 0.4, np.array([[0.2, 0.4, 0.1, 0.1], [0.6, 0.2, 0.2, 0.2]]) * 0.4]}

    data_dir=utils_MIP4Cluster().data_generation(g,f_dash,pro_rang,obs_rang,T_len,S_len)
    synthetic_paths = []
    for path in data_dir:
        source = project_dir / path
        target = synthetic_data_dir / source.name
        shutil.copy2(source, target)
        synthetic_paths.append(str(target))

    print("generated:", data_dir)
    print("saved_for_lds_yaml:", synthetic_paths)
