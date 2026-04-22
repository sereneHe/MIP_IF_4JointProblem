"""3D-indexing variant of MIP_IF for K >= 2 clustering on 3D data.

This file keeps the existing 3D EEG / sample-clustering logic from
`MIP_IF.py` and exposes a stable renamed module entry point.
"""

import os
import sys
import time
import math
import random
import numpy as np
import pandas as pd
import gurobipy as gp
from copy import deepcopy

class MIP_IF():

    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py

    """

    @staticmethod
    def _infer_bounds(X, min_state_bound=10.0, min_coeff_bound=25.0):
        """Infer conservative but finite bounds from the observed data scale."""
        arr = np.asarray(X, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            scale = 1.0
        else:
            scale = float(np.max(np.abs(finite)))
        scale = max(scale, 1.0)
        state_bound = max(min_state_bound, 2.0 * scale)
        coeff_bound = max(min_coeff_bound, 5.0 * scale)
        residual_bound = max(1.0, (state_bound + scale) ** 2)
        return state_bound, coeff_bound, residual_bound

    def ind_Gurobi_function(self, X, label, UB0, UB1, M, T, reg, time_limit=3600, gap=0.01):
        from scipy.io import arff
        import gurobipy as gp
        from gurobipy import GRB
        # from gurobipy import *

        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals."""

        e = gp.Env()
        e.setParam('TimeLimit', time_limit)
        e.setParam('MIPGap', gap)
        model = gp.Model(env=e)
        T -= 1
        I = len(X)
        state_bound, coeff_bound, residual_bound = self._infer_bounds(X)
        print(UB0,UB1,T, M)

        # Var
        L = model.addVars(I, name="L", vtype='B')  # Indicator

        f0 = model.addVars(M, T, lb=-state_bound, ub=state_bound, name="f0", vtype='C')
        phi0 = model.addVars(UB0, (T + 1), lb=-state_bound, ub=state_bound, name="phi0", vtype='C')
        G0 = model.addVars(UB0, UB0, lb=-coeff_bound, ub=coeff_bound, name="G0", vtype='C')
        F0 = model.addVars(M, UB0, lb=-coeff_bound, ub=coeff_bound, name="F0", vtype='C')
        q0 = model.addVars(UB0, T, lb=-state_bound, ub=state_bound, name="q0", vtype='C')
        p0 = model.addVars(M, T, lb=-state_bound, ub=state_bound, name="p0", vtype='C')
        quatr0 = model.addVars(M, (T + 1), lb=-state_bound, ub=state_bound, name="quatr0", vtype='C')
        quatr_hidden0 = model.addVars(UB0, T, lb=-state_bound, ub=state_bound, name="quatr_hidden0", vtype='C')

        f1 = model.addVars(M, T, lb=-state_bound, ub=state_bound, name="f1", vtype='C')
        phi1 = model.addVars(UB1, (T + 1), lb=-state_bound, ub=state_bound, name="phi1", vtype='C')
        G1 = model.addVars(UB1, UB1, lb=-coeff_bound, ub=coeff_bound, name="G1", vtype='C')
        F1 = model.addVars(M, UB1, lb=-coeff_bound, ub=coeff_bound, name="F1", vtype='C')
        q1 = model.addVars(UB1, T, lb=-state_bound, ub=state_bound, name="q1", vtype='C')
        p1 = model.addVars(M, T, lb=-state_bound, ub=state_bound, name="p1", vtype='C')
        quatr1 = model.addVars(M, (T + 1), lb=-state_bound, ub=state_bound, name="quatr1", vtype='C')
        quatr_hidden1 = model.addVars(UB1, T, lb=-state_bound, ub=state_bound, name="quatr_hidden1", vtype='C')
        z0_squared = model.addVars(I, T, M, lb=0.0, ub=residual_bound, name="z0_squared", vtype='C')
        z1_squared = model.addVars(I, T, M, lb=0.0, ub=residual_bound, name="z1_squared", vtype='C')
        sel0 = model.addVars(I, T, M, lb=0.0, ub=residual_bound, name="sel0", vtype='C')
        sel1 = model.addVars(I, T, M, lb=0.0, ub=residual_bound, name="sel1", vtype='C')


        # Obj
        obj = gp.quicksum(sel0[i, t, m] + sel1[i, t, m] for m in range(M) for t in range(T) for i in range(I))
        obj += gp.quicksum(reg * q0[n0, t] ** 2 for n0 in range(UB0) for t in range(T))
        obj += gp.quicksum(reg * q1[n1, t] ** 2 for n1 in range(UB1) for t in range(T))
        obj += gp.quicksum(reg * p0[m, t] ** 2 + reg * p1[m, t] ** 2 for m in range(M) for t in range(T))

        model.setObjective(obj, GRB.MINIMIZE)

        # Constraint
        model.addConstrs(z0_squared[i, t, m] == (X[i, t, m] - f0[m, t])**2 for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(z1_squared[i, t, m] == (X[i, t, m] - f1[m, t])**2 for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel0[i, t, m] <= z0_squared[i, t, m] for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel0[i, t, m] <= residual_bound * (1 - L[i]) for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel0[i, t, m] >= z0_squared[i, t, m] - residual_bound * L[i] for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel1[i, t, m] <= z1_squared[i, t, m] for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel1[i, t, m] <= residual_bound * L[i] for i in range(I) for t in range(T) for m in range(M))
        model.addConstrs(sel1[i, t, m] >= z1_squared[i, t, m] - residual_bound * (1 - L[i]) for i in range(I) for t in range(T) for m in range(M))

        for t in range(T):
            for m in range(M):
                model.addConstr(gp.quicksum(F0[m, n] * phi0[n, t+1] for n in range(UB0)) == quatr0[m, t+1])
                model.addConstr(gp.quicksum(F1[m, n] * phi1[n, t+1] for n in range(UB1)) == quatr1[m, t+1])
        model.addConstrs(f0[m, t] == quatr0[m, t + 1] + p0[m, t] for t in range(T) for m in range(M))
        model.addConstrs(f1[m, t] == quatr1[m, t + 1] + p1[m, t] for t in range(T) for m in range(M))
        for t in range(T):
            for n in range(UB0):
                model.addConstr(gp.quicksum(G0[n, nn] * phi0[nn, t] for nn in range(UB0)) == quatr_hidden0[n, t])
            for n in range(UB1):
                model.addConstr(gp.quicksum(G1[n, nn] * phi1[nn, t] for nn in range(UB1)) == quatr_hidden1[n, t])
        model.addConstrs(phi0[n, t + 1] == quatr_hidden0[n, t] + q0[n, t] for t in range(T) for n in range(UB0))
        model.addConstrs(phi1[n, t + 1] == quatr_hidden1[n, t] + q1[n, t] for t in range(T) for n in range(UB1))

        model.update()
        model.Params.NonConvex = 2

        model.optimize()

        if model.status == GRB.Status.OPTIMAL:
            print("THIS IS OPTIMAL SOLUTION")
        else:
            print("THIS IS NOT OPTIMAL SOLUTION")

        if model.SolCount == 0:
            data_dict = {
                'G0': [0.0] * (UB0 * UB0),
                'F0': [0.0] * (M * UB0),
                'G1': [0.0] * (UB1 * UB1),
                'F1': [0.0] * (M * UB1),
            }
            label_out = np.zeros(I, dtype=int)
            return model, label_out, data_dict

        data_dict = {
            'G0': [model.getAttr("x", G0)[h, k] for (h, k) in model.getAttr("x", G0)],
            'F0': [model.getAttr("x", F0)[h, k] for (h, k) in model.getAttr("x", F0)],
            'G1': [model.getAttr("x", G1)[h, k] for (h, k) in model.getAttr("x", G1)],
            'F1': [model.getAttr("x", F1)[h, k] for (h, k) in model.getAttr("x", F1)]
        }

        label_out = np.array([round(model.getAttr("x", L)[h]) for h in model.getAttr("x", L)])

        # print(f'M={M}, T={T}')
        return model, label_out, data_dict

    def KCluster_ind_Gurobi_function(self, X, K, label, UB, M, T, reg, time_limit=60):
        """Fit a K-cluster LDS model with Gurobi."""
        import gurobipy as gp
        from gurobipy import GRB

        e = gp.Env()
        e.setParam('TimeLimit', time_limit)
        model = gp.Model(env=e)
        T -= 1
        I = len(X)
        state_bound, coeff_bound, residual_bound = self._infer_bounds(X)

        if K < 2:
            raise ValueError("K must be at least 2.")
        if K > I:
            raise ValueError("K cannot exceed the number of samples when min_cluster_size=1.")

        # Binary assignment vars: L[i, k] = 1 if sample i is in cluster k.
        L = model.addVars(I, K, name="L", vtype='B')
        f = model.addVars(K, M, T, lb=-state_bound, ub=state_bound, name="f", vtype='C')
        phi = model.addVars(K, UB, (T + 1), lb=-state_bound, ub=state_bound, name="phi", vtype='C')
        G = model.addVars(K, UB, UB, lb=-coeff_bound, ub=coeff_bound, name="G", vtype='C')
        F = model.addVars(K, M, UB, lb=-coeff_bound, ub=coeff_bound, name="F", vtype='C')
        q = model.addVars(K, UB, T, lb=-state_bound, ub=state_bound, name="q", vtype='C')
        p = model.addVars(K, M, T, lb=-state_bound, ub=state_bound, name="p", vtype='C')
        quatr = model.addVars(K, M, (T + 1), lb=-state_bound, ub=state_bound, name="quatr", vtype='C')
        quatr_hidden = model.addVars(K, UB, T, lb=-state_bound, ub=state_bound, name="quatr_hidden", vtype='C')
        z_squared = model.addVars(K, I, T, M, lb=0.0, ub=residual_bound, name="z_squared", vtype='C')
        selected = model.addVars(K, I, T, M, lb=0.0, ub=residual_bound, name="selected", vtype='C')

        obj = gp.quicksum(
            selected[k, i, t, m]
            for k in range(K)
            for i in range(I)
            for t in range(T)
            for m in range(M)
        )
        obj += gp.quicksum(
            reg * q[k, n, t] ** 2
            for k in range(K)
            for n in range(UB)
            for t in range(T)
        )
        obj += gp.quicksum(
            reg * p[k, m, t] ** 2
            for k in range(K)
            for m in range(M)
            for t in range(T)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        model.addConstrs(
            (z_squared[k, i, t, m] == (X[i, t, m] - f[k, m, t]) ** 2
             for k in range(K) for i in range(I) for t in range(T) for m in range(M)),
            name="cluster_squared_residual",
        )
        model.addConstrs(
            (selected[k, i, t, m] <= z_squared[k, i, t, m]
             for k in range(K) for i in range(I) for t in range(T) for m in range(M)),
            name="selected_le_residual",
        )
        model.addConstrs(
            (selected[k, i, t, m] <= residual_bound * L[i, k]
             for k in range(K) for i in range(I) for t in range(T) for m in range(M)),
            name="selected_le_assign",
        )
        model.addConstrs(
            (selected[k, i, t, m] >= z_squared[k, i, t, m] - residual_bound * (1 - L[i, k])
             for k in range(K) for i in range(I) for t in range(T) for m in range(M)),
            name="selected_ge_link",
        )

        model.addConstrs(
            (
                gp.quicksum(F[k, m, n] * phi[k, n, t + 1] for n in range(UB))
                == quatr[k, m, t + 1]
            )
            for k in range(K)
            for m in range(M)
            for t in range(T)
        )
        model.addConstrs(
            (
                gp.quicksum(G[k, n, nn] * phi[k, nn, t] for nn in range(UB))
                == quatr_hidden[k, n, t]
            )
            for k in range(K)
            for n in range(UB)
            for t in range(T)
        )

        # Pointwise LDS observation and hidden-state dynamics.
        model.addConstrs(
            (f[k, m, t] == quatr[k, m, t + 1] + p[k, m, t]
             for k in range(K) for m in range(M) for t in range(T)),
            name="pointwise_obs",
        )
        model.addConstrs(
            (phi[k, n, t + 1] == quatr_hidden[k, n, t] + q[k, n, t]
             for k in range(K) for n in range(UB) for t in range(T)),
            name="pointwise_state",
        )

        # Each sample is assigned to exactly one cluster.
        model.addConstrs(
            (gp.quicksum(L[i, k] for k in range(K)) == 1 for i in range(I)),
            name="one_cluster_per_sample",
        )

        min_cluster_size = 1
        max_cluster_size = I - 1
        model.addConstrs(
            (gp.quicksum(L[i, k] for i in range(I)) >= min_cluster_size for k in range(K)),
            name="min_cluster_size",
        )
        model.addConstrs(
            (gp.quicksum(L[i, k] for i in range(I)) <= max_cluster_size for k in range(K)),
            name="max_cluster_size",
        )

        # Symmetry breaking: enforce non-increasing cluster sizes.
        model.addConstrs(
            (
                gp.quicksum(L[i, k] for i in range(I))
                >= gp.quicksum(L[i, k + 1] for i in range(I))
                for k in range(K - 1)
            ),
            name="ordered_cluster_sizes",
        )

        model.update()
        model.Params.NonConvex = 2
        model.optimize()

        if model.status == GRB.Status.OPTIMAL:
            print("THIS IS OPTIMAL SOLUTION")
        else:
            print("THIS IS NOT OPTIMAL SOLUTION")

        if model.SolCount == 0:
            data_dict = {'G': [], 'F': []}
            label_out = np.full(I, -1, dtype=int)
            return model, label_out, data_dict

        G_x = model.getAttr("x", G)
        F_x = model.getAttr("x", F)
        L_x = model.getAttr("x", L)
        data_dict = {
            'G': [G_x[k, h, j] for (k, h, j) in G_x],
            'F': [F_x[k, m, h] for (k, m, h) in F_x],
        }

        label_out = np.array(
            [max(range(K), key=lambda k: L_x[i, k]) for i in range(I)]
        )

        return model, label_out, data_dict

    def system_matrix(self, arr_G, arr_F, M, UB, zero_threshold=1e-4):
        arr_rounded_G = np.round(arr_G, decimals=4)
        arr_rounded_F = np.round(arr_F, decimals=4)
        arr_rounded_G[np.abs(arr_rounded_G) < zero_threshold] = 0
        arr_rounded_F[np.abs(arr_rounded_F) < zero_threshold] = 0
        rows_G, cols_G = arr_G.shape
        rows_F, cols_F = arr_F.shape

        if not arr_rounded_G.any() and not arr_rounded_F.any():
            print("Warning: arr_rounded_G is empty or contains only zeros.")
            return arr_G, arr_F,0

        else:
            nonzero_rows_G, nonzero_cols_G = np.nonzero(arr_rounded_G.reshape(UB, UB))
            nonzero_rows_F, nonzero_cols_F = np.nonzero(arr_rounded_F.reshape(M,UB))
            def max_max(G):
                if G.size ==0:
                    return 0
                else:
                    return max(G)
            N = max(max_max(nonzero_rows_G), max_max(nonzero_cols_G), max_max(nonzero_cols_F)) + 1
            # print(f"N:{N}")
            system_G = arr_G[:N, :N]
            system_F = arr_F[:, :N]
            return system_G, system_F, N

    def obj_function(self, model, reg=270):
        if reg is None:
          reg = 270
        obj_1 = sum((1-model.L[i])*(model.X[i][t,m]-model.f[m,t])**2 + model.L[i]*(model.X[i][t,m]-model.ff[m,t])**2 for i in model.I for m in model.M for t in model.T1)
        obj_2 = reg*sum(model.p[m,t]**2 for m in model.M for t in model.T1) + reg*sum(model.q[n,t]**2 for n in model.N for t in model.T1)
        obj_3 = reg*sum(model.pp[m,t]**2 for m in model.M for t in model.T1) + reg*sum(model.qq[n,t]**2 for n in model.N for t in model.T1)
        return obj_1+obj_2+obj_3

    def ind_Bonmin_function(self, X,label,N,M,T=20,option='bonmin', reg=270):
        from pyomo.environ import (
            ConcreteModel,
            RangeSet,
            Var,
            Param,
            Reals,
            NonNegativeIntegers,
            Objective,
            Constraint,
            minimize,
            SolverFactory,
        )

        I = len(X)
        # T = len(X[0])
        # print(I,T)

        #---------------------------------------------#
        # Dimension of G & F  - G:[N,N]  Fdash:[M,N]  #
        #---------------------------------------------#
        # N = 2
        # M = 1

        # Create model
        model = ConcreteModel()

        # Create index set
        model.N = RangeSet(0, N-1)
        model.M = RangeSet(0, M-1)
        model.I = RangeSet(0, I-1)
        model.T1 = RangeSet(0,T-1)
        model.T2 = RangeSet(0,T)

        # Create variables
        model.L = Var(model.I, within=NonNegativeIntegers, bounds=(0,1), initialize =label) #indicator

        model.f = Var(model.M, model.T1, domain=Reals)
        model.m = Var(model.N, model.T2, domain=Reals)
        model.G = Var(model.N, model.N, domain=Reals)
        model.Fdash = Var(model.M, model.N, domain=Reals)
        model.q = Var(model.N, model.T1, domain=Reals)
        model.p = Var(model.M, model.T1, domain=Reals)

        model.ff = Var(model.M, model.T1, domain=Reals)
        model.mm = Var(model.N, model.T2, domain=Reals)
        model.GG = Var(model.N, model.N, domain=Reals)
        model.FFdash = Var(model.M, model.N, domain=Reals)
        model.qq = Var(model.N, model.T1, domain=Reals)
        model.pp = Var(model.M, model.T1, domain=Reals)

        # Create parameters
        model.X = Param(model.I, initialize = X,mutable=False)

        # Create objective function
        model.OBJ = Objective(rule=self.obj_function, sense=minimize)

        # Create constrains
        model.con1 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) >=0)
        model.con2 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) <=0)
        model.con3 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) >=0)
        model.con4 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) <=0)

        model.con5 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) >=0)
        model.con6 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) <=0)
        model.con7 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) >=0)
        model.con8 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) <=0)

        solver = SolverFactory(option)
        solver.solve(model, tee=True)
        # solver.solve(model, tee=True).write()

        # solver = SolverManagerFactory('neos')
        # result = solver.solve(model, opt='mosek')

        # model.display()
        iid = []
        for i in model.L:
          iid.append(model.L[i].value)

        return model, np.array(iid)

    def obj_function_2(self, model, reg=270):
        if reg is None:
          reg = 270
        obj_1 = sum((model.K[k][t,m]-model.f[m,t])**2 for k in model.KK for m in model.M for t in model.T1)
        obj_2 = sum((model.L[l][t][m]-model.ff[m,t])**2 for l in model.LL for m in model.M for t in model.T1)
        obj_3 = reg*sum(model.p[m,t]**2 for m in model.M for t in model.T1) + reg*sum(model.q[n,t]**2 for n in model.N for t in model.T1)
        obj_4 = reg*sum(model.pp[m,t]**2 for m in model.M for t in model.T1) + reg*sum(model.qq[n,t]**2 for n in model.N for t in model.T1)
        return obj_1+obj_2+obj_3+obj_4

    def SimCom(self, K,L,T,M,N, reg,option='bonmin'): # Dimension of G & F   # G:[N,N]  F_dash:[M,N]
        """option: 'bonmin', 'cplex', 'mosek'..."""
        from pyomo.environ import (
            ConcreteModel,
            RangeSet,
            Var,
            Param,
            Reals,
            Objective,
            Constraint,
            minimize,
            SolverFactory,
        )

        KK = len(K)
        LL = len(L)
        # T = 5
        print('cluster length:',KK,LL)

        # Create model
        model = ConcreteModel()
        # # Dimension of G & F   # G:[N,N]  F_dash:[M,N]
        # M = 2
        # N = 2
        # Create index set
        model.M = RangeSet(0, M-1)
        model.N = RangeSet(0, N-1)
        model.KK = RangeSet(0,KK-1)
        model.LL = RangeSet(0,LL-1)
        model.T1 = RangeSet(0,T-1)
        model.T2 = RangeSet(0,T)

        # Create variables
        model.f = Var(model.M, model.T1, domain=Reals)
        model.m = Var(model.N, model.T2, domain=Reals)
        model.G = Var(model.N, model.N, domain=Reals)
        model.Fdash = Var(model.M, model.N, domain=Reals)
        model.q = Var(model.N, model.T1, domain=Reals)
        model.p = Var(model.M, model.T1, domain=Reals)

        model.ff = Var(model.M, model.T1, domain=Reals)
        model.mm = Var(model.N, model.T2, domain=Reals)
        model.GG = Var(model.N, model.N, domain=Reals)
        model.FFdash = Var(model.M, model.N, domain=Reals)
        model.qq = Var(model.N, model.T1, domain=Reals)
        model.pp = Var(model.M, model.T1, domain=Reals)

        # Create parameters
        model.K = Param(model.KK,initialize = K,mutable=False)
        model.L = Param(model.LL,initialize = L,mutable=False)

        # Create objective function

        model.OBJ = Objective(rule=self.obj_function_2, sense=minimize)

        # Create constrains
        model.con1 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) >=0)
        model.con2 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) <=0)
        model.con3 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) >=0)
        model.con4 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) <=0)

        model.con5 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) >=0)
        model.con6 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) <=0)
        model.con7 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) >=0)
        model.con8 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) <=0)

        solver = SolverFactory(option)
        solver.solve(model)
        # solver.solve(model, tee=True).write()

        # solver = SolverManagerFactory('neos')
        # result = solver.solve(model, opt='mosek')


        # model.pprint()
        return model

    #######################  CONCRETE MODEL  #######################
    # model with G constrains
    def SimCom_norm(self, K,L,T,M,N,reg,option='bonmin'): # Dimension of G & F   # G:[N,N]  F_dash:[M,N]
        from pyomo.environ import (
            ConcreteModel,
            RangeSet,
            Var,
            Param,
            Reals,
            Objective,
            Constraint,
            minimize,
            SolverFactory,
        )
        KK = len(K)
        LL = len(L)
        # T = 5
        print('cluster length:',KK,LL)

        # Create model
        model = ConcreteModel()

        model.M = RangeSet(0, M-1)
        model.N = RangeSet(0, N-1)
        model.KK = RangeSet(0,KK-1)
        model.LL = RangeSet(0,LL-1)
        model.T1 = RangeSet(0,T-1)
        model.T2 = RangeSet(0,T)

        # Create variables
        model.f = Var(model.M, model.T1, domain=Reals)
        model.m = Var(model.N, model.T2, domain=Reals)
        model.G = Var(model.N, model.N, domain=Reals)
        model.Fdash = Var(model.M, model.N, domain=Reals)
        model.q = Var(model.N, model.T1, domain=Reals)
        model.p = Var(model.M, model.T1, domain=Reals)

        model.ff = Var(model.M, model.T1, domain=Reals)
        model.mm = Var(model.N, model.T2, domain=Reals)
        model.GG = Var(model.N, model.N, domain=Reals)
        model.FFdash = Var(model.M, model.N, domain=Reals)
        model.qq = Var(model.N, model.T1, domain=Reals)
        model.pp = Var(model.M, model.T1, domain=Reals)

        # Create parameters
        model.K = Param(model.KK,initialize = K,mutable=False)
        model.L = Param(model.LL,initialize = L,mutable=False)

        # Create objective function

        model.OBJ = Objective(rule=self.obj_function_2, sense=minimize)

        # Create constrains
        model.con1 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) >=0)
        model.con2 = Constraint(expr=sum(model.f[m,t]-sum(model.Fdash[m,n]*model.m[n,t+1] for n in model.N) -model.p[m,t] for m in model.M for t in model.T1) <=0)
        model.con3 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) >=0)
        model.con4 = Constraint(expr=sum(model.m[n,t+1]-sum(model.G[n,n2]*model.m[n2,t] for n2 in model.N) - model.q[n,t] for n in model.N for t in model.T1) <=0)
        model.con5 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) >=0)
        model.con6 = Constraint(expr=sum(model.ff[m,t]-sum(model.FFdash[m,n]*model.mm[n,t+1] for n in model.N) -model.pp[m,t] for m in model.M for t in model.T1) <=0)
        model.con7 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) >=0)
        model.con8 = Constraint(expr=sum(model.mm[n,t+1]-sum(model.GG[n,n2]*model.mm[n2,t] for n2 in model.N) - model.qq[n,t] for n in model.N for t in model.T1) <=0)
        model.con9 = Constraint(expr=sum(model.G[i,j] for i in model.N for j in model.N)-1==0)
        model.con10 = Constraint(expr=sum(model.GG[i,j] for i in model.N for j in model.N)-1==0)

        solver = SolverFactory(option)
        solver.solve(model)


        # model.pprint()
        return model


    def FFT_estimate(self, data, label, seed=42, is_plot=False):
        import numpy as np
        import pandas as pd
        import math
        import matplotlib.pyplot as plt
        from sklearn.metrics import f1_score
        from sklearn import preprocessing
        from sklearn.cluster import KMeans
        from sklearn.cluster import SpectralClustering
        import time

        X_train = data
        seed = seed
        np.random.seed(seed)
        index = np.arange(len(X_train))
        np.random.shuffle(index)

        X_train = X_train[index]
        X_label = label[index]
        print(X_label)

        data_fft = np.fft.rfft(X_train, axis=1)
        fft_features = np.abs(data_fft)
        feature_vectors = fft_features.reshape(len(X_train), -1)
        kmeans = KMeans(n_clusters=2, verbose=is_plot, random_state=seed)
        kmeans.fit(feature_vectors)
        y_pred = kmeans.labels_
        print(y_pred)
        '''
        if is_plot:
            for yi in range(2):
                plt.subplot(1, 2, 1+yi)
                for xx in X_train[y_pred == yi]:
                    if len(xx.shape) == 1:
                        plt.plot(xx, alpha=.2)
                    else:
                        plt.plot(xx[:,0], alpha=.2)
                        plt.plot(xx[:,1], alpha=.2)
                # plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
                # plt.xlim(0, 20)
                plt.ylim(-2, 4)
                plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                        transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title("FFT $k$-means")
            plt.tight_layout()
            plt.show()

            for freq in fft_features[y_pred==0]:
                if len(freq.shape) == 1:
                    plt.plot(freq, alpha=.5)
                else:
                    plt.plot(freq[:,0], alpha=.5)
                    plt.plot(freq[:,1], alpha=.5)
            plt.show()
            for freq in fft_features[y_pred==1]:
                if len(freq.shape) == 1:
                    plt.plot(freq, alpha=.5)
                else:
                    plt.plot(freq[:,0], alpha=.5)
                    plt.plot(freq[:,1], alpha=.5)
            plt.show()
        '''

        f1_1 = f1_score(X_label, y_pred)
        f1_2 = f1_score(X_label, 1-y_pred)
        if f1_1>f1_2:
            f1 = f1_1
        else:
            f1 = f1_2
        print(f1)
        return f1

    def DTW_estimate(self, data, label, seed=42, is_plot=False):
        import random
        from scipy.io import arff
        import sys
        from sklearn.metrics import f1_score
        sys.path.append('../')
        #from inputlds import *
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.datasets import CachedDatasets
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

        X_train = data
        X_label = label
        seed = seed
        np.random.seed(seed)
        index = np.arange(len(X_train))
        np.random.shuffle(index)
        # print(X_train[0])

        X_train = X_train[index]
        X_label = X_label[index]
        print(X_label)

        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        sz = X_train.shape[1]
        print(sz)

        # Soft-DTW-k-means
        print("Soft-DTW k-means")
        sdtw_km = TimeSeriesKMeans(n_clusters=2,
                                max_iter=10,
                                metric="softdtw",
                                metric_params={"gamma": .01},
                                verbose=True,
                                random_state=seed)
        y_pred = sdtw_km.fit_predict(X_train)
        print(y_pred)
        '''
        if is_plot:
            for yi in range(2):
                plt.subplot(1, 2, 1+yi)
                for xx in X_train[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.2)
                plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
                plt.xlim(0, sz)
                plt.ylim(-2, 4)
                plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                        transform=plt.gca().transAxes)
                if yi == 1:
                    plt.title("Soft-DTW $k$-means")
            plt.tight_layout()
            plt.show()
        '''
        f1_1 = f1_score(X_label, y_pred)
        f1_2 = f1_score(X_label, 1-y_pred)
        if f1_1>f1_2:
            f1 = f1_1
        else:
            f1 = f1_2

        print('f1:',f1)
        return f1



    def MIP_estimate(self, data_in,label_in,method,N,name,reg, seed, MTS=True,option='bonmin',norm=True, thresh=0.25, shuffle=True, time_limit=3600, gap=0.01):
        from sklearn.metrics import f1_score

        """
        X_data.shape: N,T,M, for ecg, X_data contains all the data
        X_lebel.shape: N     for both, X_label is like [1,1,1,0,0,0]
        name: 'lds' or 'ecg'
        """
        print("lds initialized label:", label_in)
        M = data_in.shape[2]
        T = data_in.shape[1]
        print("m,t:",M,T)

        if shuffle:
            np.random.seed(seed)
            idx = np.arange(len(data_in))
            np.random.shuffle(idx)
            label = label_in[idx]
            print("lds initialized label:", label_in)
            X = data_in[idx, :, : ]
        else:
            label = label_in
            X = data_in
        if method =='IF':
            model, label_out = self.ind_Bonmin_function(X=data_in, label=label, N=N,M=M,T=T,option='bonmin',reg =reg)
        elif method == 'IF-Gurobi':
            UB0,UB1=N,N
            model, label_out, data_dict = self.ind_Gurobi_function(X=data_in,label=label,UB0=UB0,UB1=UB1,M=M,T=T, reg=reg, time_limit=time_limit, gap=gap)
            G0 = np.array(np.array(data_dict['G0']).reshape(UB0,UB0))
            F0 = np.array(np.array(data_dict['F0']).reshape(M,UB0))
            G1 = np.array(np.array(data_dict['G1']).reshape(UB1,UB1))
            F1 = np.array(np.array(data_dict['F1']).reshape(M,UB1))
            G0, F0, N0=self.system_matrix(G0, F0, M, UB=UB0, zero_threshold=1e-4)
            G1, F1, N1=self.system_matrix(G1, F1, M, UB=UB1, zero_threshold=1e-4)
            print(f"Hidden_State Dimension of System 1 is:{N0}.")
            print(f"Hidden_State Dimension of System 2 is:{N1}.")
            print(f"System matrices are G0: \n {G0} and \n F0:\n{F0}.")
            print(f"System matrices are G1: \n {G1} and \n F1:\n{F1}.")
            if np.sum(np.array(data_dict['G0']) == 0) >= (N0*N0+N0*M)*thresh or np.sum(np.array(data_dict['F0']) == 0) >= (N0*N0+N0*M)*thresh or np.sum(np.array(data_dict['G1']) == 0) >= (N1*N1+N1*M)*thresh or np.sum(np.array(data_dict['F1']) == 0) >= (N1*N1+N1*M)*thresh:
              print("The System Matrices Is Sparse.")
              validation = 0
            else:
              print("The System Matrices Is Dense!")
              validation = 1
        else:
            idx = np.random.randint(0,2,len(data_in))
            K = [data_in[i] for i in np.where(idx==0)[0]]  #idx = 0
            L = [data_in[i] for i in np.where(idx==1)[0]]  #idx = 1
            print(len(K),len(L))
            # print(K[0].shape)

            ### EM ###
            for i in range(data_in.shape[1]):
                label_out= deepcopy(idx)
                print(i, "start: idx_=", label_out)

                if norm == True:
                  model = self.SimCom_norm(K,L,T=T,M=M, N=N, reg = 270,option='bonmin')
                else:
                  model = self.SimCom(K,L,T=T,M=M, N=N, reg = 270,option='bonmin')
                pred_k = model.f # index obj, model.f[i].value
                pred_l = model.ff

                for n in range(len(data_in)):
                    cost_k = sum((data_in[n][i,m]-pred_k[m,i].value)**2 for m in range(M) for i in range(T))
                    cost_l = sum((data_in[n][i,m]-pred_l[m,i].value)**2 for m in range(M) for i in range(T))
                    # print(cost_k, cost_l)
                    if cost_k < cost_l: # K cluster --> 0
                        idx[n] = 0
                    else:
                        idx[n] = 1

                print(i, "changed: idx=", idx)
                if np.array_equal(label_out,idx):
                    print(label_out, "end")
                    break
                else:
                    label_out = deepcopy(idx)
                    K = [data_in[i] for i in np.where(idx==0)[0]]  #idx = 0
                    L = [data_in[i] for i in np.where(idx==1)[0]]  #idx = 1
                    print('K:',len(K),'\n','L:',len(L))

        f1_1 = f1_score(label, label_out)
        f1_2 = f1_score(label, 1-label_out)

        if f1_1>f1_2:
            f1 = f1_1
        else:
            f1 = f1_2

        print('f1-score:',f1)

        if method == 'IF-Gurobi':
            return f1, validation
        else:
            return f1


MIP_IF_3Dindexing = MIP_IF
