class MIP4Cluster:
    """Estimator based on NCPOP Regressor"""

    def __init__(self):
        super().__init__()

    def setup_environment(self,method):
        import sys
        import os
        import importlib
        import numpy as np
        import time

        """
        Set up the environment by installing necessary packages and configuring paths.

        Parameters:
        method : str - The method being used (e.g., 'IF-Gurobi', 'IF', etc.)
        """
        # Install required packages for 'IF-Gurobi'
        if method == 'IF-Gurobi':
            if not importlib.util.find_spec("gurobipy") or  not importlib.util.find_spec("sklearn") or not importlib.util.find_spec("seaborn"):
                from sklearn.metrics import f1_score
                from scipy.io import arff
            import gurobipy as gp
            from sklearn.metrics import f1_score
            from scipy.io import arff

            # Set up Gurobi license path (update this as needed)
            os.environ['GRB_LICENSE_FILE'] = './gurobi.lic'

        elif method in['EM', 'IF']:
            if 'google.colab' in sys.modules:
                os.environ['PATH'] += ':bin'
        else:
            if not importlib.util.find_spec("tslearn"):
                print("Please install tslearn package for DTW and FFT methods.")
    
    def run_experiment(self, data_in, label_in, method, name, S, I, T, M, N, J, reg):
        import time
        import numpy as np
        from MIP_IF import MIP_IF
        # S, I, T, M, N, N_Data
        f1_list = []
        validation = []
        duration = []
        for nn in N: 
            n = int(nn[0])
            t_start = time.time()
            for mm in M:
              m = int(mm[0])
              for tt in T:
                t = int(tt[0])
                for s in range(S):
                    seed =s +30
                    np.random.seed(seed)
                    T_start = np.random.randint(5, 10)
                    X = data_in[s] 
                    X = X[:, range(T_start, T_start + t), :m]

                    f1_result, valid_result = MIP_IF().MIP_estimate(X, label_in, method, n, name, reg, seed, thresh=0.25)
                    print(f"n:{n}. regularization:{reg}. F1:{f1_result}")
                    validation.append(valid_result)
                    f1_list.append(f1_result)

            duration.append(time.time() - t_start)

        return f1_list, duration, validation

    def Test_MIP4Cluster(self, data_in, label_in, met, S, I, T, M, name, N, J, regularization, MTS=True,option='bonmin',norm=True, thresh=0.25):
        import os
        import sys
        import time
        import numpy as np
        import importlib.util
        from MIP_IF import MIP_IF

        if not os.path.exists(f'./Result_{name}'):
            os.makedirs(f'./Result_{name}')
        if not os.path.exists(f'./Result_{name}'):
            os.makedirs(f'./Result_{name}')
        print(data_in.shape, label_in.shape)
        data_num = data_in.shape[0]
        for method in met:
            if method in ['IF-Gurobi', 'IF', 'EM', 'FFT', 'DTW']:
                self.setup_environment(method)
                
                if all(len(dim.shape) <= 1 for dim in [label_in, T, M, N]):  # Corrected the condition
                    raise ValueError("Cannot generate two dims result.")
                else:
                    # Reshape
                    print(T.shape, N.shape)
                    if T.shape[0] <= 1 and N.shape[0] <= 1:
                        raise ValueError("Cannot reshape.")
                    elif T.shape[0] <= 1:
                        list_shape = (N.shape[0], S*J) # 3*20
                        shape = (1, N.shape[0])
                        axis =1
                    elif len(N) <= 1:
                        list_shape = (T.shape[0], S*J) # 5*20
                        shape = (1,N.shape[0])
                        axis =1
                    else:
                        list_shape = (T.shape[0], N.shape[0], S*J) # 5*3*20
                        shape = (T.shape[0], N.shape[0])
                        axis =2

                ###################################################### Regularization ##############################################
                if regularization.shape[0]>1:
                    if method == 'IF-Gurobi':
                        for r in regularization:
                            reg=int(r)
                            f1_list, duration, validation  = self.run_experiment(data_in, label_in, method, name, S, I, T, M, N, J, reg)
                            f1 = np.array(f1_list).reshape(list_shape)
                            f1_mean = np.mean(f1, axis=axis)
                            f1_std = np.std(f1, axis=axis)
                            duration = np.array(duration).reshape(shape)
                            validation = np.array(validation).reshape(list_shape)
                            count_validation = np.sum(validation == 1)
                            print(f'f1_{method}_{name}_mean_reg{reg} is: \n {f1_mean}, '
                                  f'f1_{method}_{name}_std_reg{reg} is: \n {f1_std}, '
                                  f'duration_{method}_{name}_reg{reg} is: \n {duration}, '
                                  f'validation_{method}_{name}_reg{reg} is: \n {validation},'
                                  f'Nonzero System Matrices are: {count_validation}.')
                            save_paths = [
                                (f'./Result_{name}/f1_{method}_{name}_reg{reg}.npy', f1),
                                (f'./Result_{name}/f1_{method}_{name}_mean_reg{reg}.npy', f1_mean),
                                (f'./Result_{name}/f1_{method}_{name}_std_reg{reg}.npy', f1_std),
                                (f'./Result_{name}/duration_{method}_{name}_reg{reg}.npy', duration),
                                (f'./Result_{name}/validation_{method}_{name}_reg{reg}.npy', validation)
                            ]
                            for path, data in save_paths:
                                with open(path, 'wb') as f:
                                    np.save(f, data)
                        break

                    else:
                        print('Warning: Cannot compare regularization for other method rather than IF-Gurobi.')

                ############################################ 'IF-Gurobi', 'IF', 'EM', 'FFT', 'DTW' #############################################
                else:
                    reg = regularization[0]
                    f1_list = []
                    validation = []
                    duration = []
                    # S, I, T, M, N
                    for nn in N: 
                        n = int(nn[0])
                        t_start = time.time()
                        for mm in M:
                          m = int(mm[0])
                          for tt in T:
                            t = int(tt[0])
                            print(t)
                            for s in range(S):
                                seed =s +30
                                np.random.seed(seed)
                                I_size = np.random.randint(10,12)#I*3/2, I*2
                                sampl = np.random.choice(2*I, size=I_size, replace=False).tolist()
                                label = label_in[sampl]
                                T_start = np.random.randint(5, 10)
                                # print([s], sampl, t, m)
                                X = data_in[s][np.ix_(sampl, range(T_start, T_start + t), range(m))]
                                # X = data_in[s] 
                                # X = X[:, range(T_start, T_start + t
                                # X = data_in[s] 
                                #X= X[:, range(T_start, T_start + t), :m]
                                print(X.shape, label.shape)
                                for j in range(J):
                                    seed =seed +j
                                    if j == 0:
                                        is_plot = True
                                    else:
                                        is_plot = False
                                    # print(f"n:{n}")
                                    if method == 'IF-Gurobi':
                                          # Test IF-Gurobi
                                          f1_result, valid_result = MIP_IF().MIP_estimate(X, label, method, n, name, reg, seed, thresh=0.25)
                                          print(f"n:{n}, regularization:{reg}, {method}_F1:{f1_result}.")
                                          validation.append(valid_result)

                                    elif method == 'IF':
                                          # Test IF-Bonmin
                                          f1_result = MIP_IF().MIP_estimate(X, label, method, n, name, reg, seed)
                                          print(f"n:{n},{method}_F1:{f1_result}.")

                                    elif method == 'EM':
                                          # Test EM Heuristic
                                          f1_result = MIP_IF().MIP_estimate(X, label, method, n, name, reg, seed, MTS=True, option='bonmin', norm=True)
                                          print(f"n:{n}.{method}_F1:{f1_result}")

                                    elif method =='DTW':
                                          # Test DTW
                                          f1_result = MIP_IF().DTW_estimate(X, label, seed=seed, is_plot =is_plot)
                                          print(f"{method}_F1:{f1_result}")

                                    else:
                                          # Test FFT
                                          f1_result = MIP_IF().FFT_estimate(X, label, seed=seed, is_plot =is_plot)
                                          print(f"{method}_F1:{f1_result}")
                                    f1_list.append(f1_result)
                        duration.append(time.time() - t_start)
                f1 = np.array(f1_list).reshape(list_shape)
                f1_mean = np.mean(f1, axis=axis)
                f1_std = np.std(f1, axis=axis)
                duration = np.array(duration)#.reshape(shape)
                suffix = "_cd" if max(max(T)) < 10 and T.shape[0]!=1 and regularization.shape[0]==1 else ""

                print(f'f1_{method}_{name}_mean{suffix} is: \n {f1_mean}, '
                      f'f1_{method}_{name}_std{suffix} is: \n {f1_std}, '
                      f'duration_{method}_{name}{suffix} is: \n {duration}. ')
                save_paths = [
                    (f'./Result_{name}/f1_{method}_{name}{suffix}.npy', f1),
                    (f'./Result_{name}/f1_{method}_{name}_mean{suffix}.npy', f1_mean),
                    (f'./Result_{name}/f1_{method}_{name}_std{suffix}.npy', f1_std),
                    (f'./Result_{name}/duration_{method}_{name}{suffix}.npy', duration)
                ]
                for path, data in save_paths:
                    with open(path, 'wb') as f:
                        np.save(f, data)

        print(f'Check result under route ./Result_{name}/ .')

 
