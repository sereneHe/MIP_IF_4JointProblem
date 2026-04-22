"""K-cluster ClusterMultiLDS_Gurobi extracted and generalized from LDS_Gurobi.ipynb.

The original notebook version exposed `estimate(self, X, K)` but still used a
binary `L[0, l]` / `1 - L[0, l]` assignment, so it could only model two
clusters. This version replaces that with `Z[k, l]` and one LDS parameter set
per cluster, so `K >= 2` is actually represented in the Gurobi model.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB, tupledict


class ClusterMultiLDS_Gurobi(object):
    """K-cluster LDS clustering estimator based on Gurobi."""

    def __init__(self, **kwargs):
        super(ClusterMultiLDS_Gurobi, self).__init__()

    def estimate(self, X, K, hidden_dims=None):
        """Fit a K-cluster LDS model and return cluster labels.

        Parameters
        ----------
        X: pandas.DataFrame-like
            Input time series data with shape `(T, nL)`, where columns are the
            series/groups to cluster and rows are time points.
        K: int
            Number of clusters. This implementation supports `K >= 2`.
        hidden_dims: list[int] or int, optional
            Hidden-state dimension for each cluster. If omitted, cluster 0 uses
            dimension 2 and the remaining clusters use dimension 4, matching the
            original notebook's `n0 = 2`, `n1 = 4` pattern.

        Returns
        -------
        model: gurobipy.Model
            Optimized Gurobi model.
        label_out: numpy.ndarray
            Cluster id for each input series/group, with values in
            `{0, ..., K-1}`.
        """
        if K < 2:
            raise ValueError("This generalized version is intended for K >= 2.")

        if hidden_dims is None:
            hidden_dims = [2] + [4] * (K - 1)
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * K
        elif len(hidden_dims) != K:
            raise ValueError("hidden_dims must be an int or a list with length K.")

        T = len(X)
        nL = len(np.transpose(X))

        env = gp.Env()
        env.setParam("TimeLimit", 6 * 60)
        model = gp.Model(env=env)

        X_data = [
            ((l, t), np.transpose(X).iloc[l, t])
            for l in range(nL)
            for t in range(T)
        ]
        X_data = tupledict(X_data)

        # Assignment variables: each series l belongs to exactly one cluster k.
        Z = model.addVars(K, nL, name="Z", vtype=GRB.BINARY)
        model.addConstrs(
            (gp.quicksum(Z[k, l] for k in range(K)) == 1 for l in range(nL)),
            name="one_cluster_per_series",
        )

        G = {}
        phi = {}
        p = {}
        F = {}
        q = {}
        f = {}
        v = {}
        X_cluster = {}
        u = {}

        for k, n_k in enumerate(hidden_dims):
            G[k] = model.addVars(n_k, n_k, name=f"G{k}", vtype=GRB.CONTINUOUS)
            phi[k] = model.addVars(n_k, T + 1, name=f"phi{k}", vtype=GRB.CONTINUOUS)
            p[k] = model.addVars(n_k, T, name=f"p{k}", vtype=GRB.CONTINUOUS)
            F[k] = model.addVars(nL, n_k, name=f"F{k}", vtype=GRB.CONTINUOUS)
            q[k] = model.addVars(nL, T, name=f"q{k}", vtype=GRB.CONTINUOUS)
            f[k] = model.addVars(nL, T, name=f"f{k}", vtype=GRB.CONTINUOUS)
            v[k] = model.addVars(nL, T, name=f"v{k}", vtype=GRB.CONTINUOUS)
            X_cluster[k] = model.addVars(nL, T, name=f"X{k}", vtype=GRB.CONTINUOUS)
            u[k] = model.addVars(nL, T, name=f"u{k}", vtype=GRB.CONTINUOUS)

        decision_vars = K * nL
        for n_k in hidden_dims:
            decision_vars += n_k * n_k
            decision_vars += n_k * (T + 1)
            decision_vars += n_k * T
            decision_vars += nL * n_k
            decision_vars += 5 * nL * T
        print("This model has", decision_vars, "decision variables.")

        obj = gp.quicksum(
            (X_cluster[k][l, t] - f[k][l, t])
            * (X_cluster[k][l, t] - f[k][l, t])
            for k in range(K)
            for l in range(nL)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0005 * p[k][n_, t] * p[k][n_, t]
            for k, n_k in enumerate(hidden_dims)
            for n_ in range(n_k)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0001 * Z[k, l] * u[k][l, t]
            for k in range(K)
            for l in range(nL)
            for t in range(T)
        )

        model.setObjective(obj, GRB.MINIMIZE)

        for k, n_k in enumerate(hidden_dims):
            model.addConstrs(
                (u[k][l, t] == q[k][l, t] * q[k][l, t])
                for l in range(nL)
                for t in range(T)
            )
            model.addConstrs(
                (
                    phi[k][n_, t + 1]
                    == G[k][n_, n_] * phi[k][n_, t] + p[k][n_, t]
                )
                for n_ in range(n_k)
                for t in range(T)
            )
            model.addConstrs(
                (X_cluster[k][l, t] == Z[k, l] * X_data[l, t])
                for l in range(nL)
                for t in range(T)
            )
            model.addConstrs(
                (
                    v[k][l, t]
                    == gp.quicksum(
                        F[k][l, n_] * phi[k][n_, t + 1]
                        for n_ in range(n_k)
                    )
                )
                for l in range(nL)
                for t in range(T)
            )
            model.addConstrs(
                (
                    f[k][l, t]
                    == Z[k, l] * v[k][l, t] + Z[k, l] * q[k][l, t]
                )
                for l in range(nL)
                for t in range(T)
            )

        model.update()

        model.Params.NonConvex = 2
        model.optimize()

        if model.status == GRB.Status.OPTIMAL:
            print("THIS IS OPTIMAL SOLUTION")
        else:
            print("THIS IS NOT OPTIMAL SOLUTION")

        if model.SolCount == 0:
            return model, np.full(nL, -1, dtype=int)

        z_values = model.getAttr("x", Z)
        label_out = np.array(
            [max(range(K), key=lambda k: z_values[k, l]) for l in range(nL)]
        )
        return model, label_out
