"""K-cluster LDS Gurobi model with M-dimensional 3D indexing.

This is the K >= 2 generalization of the binary
`MIP4cluster_3Dindexing_hidden.py` model.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB, tupledict


class ClusterMultiLDS_Gurobi(object):
    """K-cluster LDS estimator based on Gurobi with 3D indexing."""

    def __init__(self, **kwargs):
        super(ClusterMultiLDS_Gurobi, self).__init__()

    def estimate(self, X, M, K=2, hidden_dims=None):
        """Fit a K-cluster LDS model with `M` independent model copies."""
        if M < 1:
            raise ValueError("M must be at least 1.")
        if K < 2:
            raise ValueError("This generalized version is intended for K >= 2.")

        if hidden_dims is None:
            hidden_dims = [2] + [4] * (K - 1)
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * K
        elif len(hidden_dims) != K:
            raise ValueError("hidden_dims must be an int or a list with length K.")

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be 2D with shape (T, nL).")

        T, nL = X_array.shape
        X_transposed = X_array.T

        env = gp.Env()
        env.setParam("TimeLimit", 6 * 60)
        model = gp.Model(env=env)

        X_data = tupledict(
            [
                ((m_, l, t), X_transposed[l, t])
                for m_ in range(M)
                for l in range(nL)
                for t in range(T)
            ]
        )

        # Z[m, k, l] = 1 if series l is assigned to cluster k in model copy m.
        Z = model.addVars(M, K, nL, name="Z", vtype=GRB.BINARY)
        model.addConstrs(
            (
                gp.quicksum(Z[m_, k, l] for k in range(K)) == 1
                for m_ in range(M)
                for l in range(nL)
            ),
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

        for m_ in range(M):
            for k, n_k in enumerate(hidden_dims):
                key = (m_, k)
                G[key] = model.addVars(n_k, n_k, name=f"G_{m_}_{k}", vtype=GRB.CONTINUOUS)
                phi[key] = model.addVars(n_k, T + 1, name=f"phi_{m_}_{k}", vtype=GRB.CONTINUOUS)
                p[key] = model.addVars(n_k, T, name=f"p_{m_}_{k}", vtype=GRB.CONTINUOUS)
                F[key] = model.addVars(nL, n_k, name=f"F_{m_}_{k}", vtype=GRB.CONTINUOUS)
                q[key] = model.addVars(nL, T, name=f"q_{m_}_{k}", vtype=GRB.CONTINUOUS)
                f[key] = model.addVars(nL, T, name=f"f_{m_}_{k}", vtype=GRB.CONTINUOUS)
                v[key] = model.addVars(nL, T, name=f"v_{m_}_{k}", vtype=GRB.CONTINUOUS)
                X_cluster[key] = model.addVars(nL, T, name=f"X_{m_}_{k}", vtype=GRB.CONTINUOUS)
                u[key] = model.addVars(nL, T, name=f"u_{m_}_{k}", vtype=GRB.CONTINUOUS)

        decision_vars = M * K * nL
        for n_k in hidden_dims:
            decision_vars += M * n_k * n_k
            decision_vars += M * n_k * (T + 1)
            decision_vars += M * n_k * T
            decision_vars += M * nL * n_k
            decision_vars += M * 5 * nL * T
        print("This model has", decision_vars, "decision variables.")

        obj = gp.quicksum(
            (X_cluster[m_, k][l, t] - f[m_, k][l, t])
            * (X_cluster[m_, k][l, t] - f[m_, k][l, t])
            for m_ in range(M)
            for k in range(K)
            for l in range(nL)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0005 * p[m_, k][n_, t] * p[m_, k][n_, t]
            for m_ in range(M)
            for k, n_k in enumerate(hidden_dims)
            for n_ in range(n_k)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0001 * Z[m_, k, l] * u[m_, k][l, t]
            for m_ in range(M)
            for k in range(K)
            for l in range(nL)
            for t in range(T)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        for m_ in range(M):
            for k, n_k in enumerate(hidden_dims):
                key = (m_, k)
                model.addConstrs(
                    (u[key][l, t] == q[key][l, t] * q[key][l, t])
                    for l in range(nL)
                    for t in range(T)
                )
                model.addConstrs(
                    (
                        phi[key][n_, t + 1]
                        == G[key][n_, n_] * phi[key][n_, t] + p[key][n_, t]
                    )
                    for n_ in range(n_k)
                    for t in range(T)
                )
                model.addConstrs(
                    (X_cluster[key][l, t] == Z[m_, k, l] * X_data[m_, l, t])
                    for l in range(nL)
                    for t in range(T)
                )
                model.addConstrs(
                    (
                        v[key][l, t]
                        == gp.quicksum(
                            F[key][l, n_] * phi[key][n_, t + 1]
                            for n_ in range(n_k)
                        )
                    )
                    for l in range(nL)
                    for t in range(T)
                )
                model.addConstrs(
                    (
                        f[key][l, t]
                        == Z[m_, k, l] * v[key][l, t] + Z[m_, k, l] * q[key][l, t]
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
            return model, np.full((M, nL), -1, dtype=int)

        z_values = model.getAttr("x", Z)
        label_out = np.array(
            [
                [max(range(K), key=lambda k: z_values[m_, k, l]) for l in range(nL)]
                for m_ in range(M)
            ],
            dtype=int,
        )
        return model, label_out


if __name__ == "__main__":
    demo_X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 1.9, 3.2],
            [1.2, 2.1, 3.1],
        ]
    )
    ClusterMultiLDS_Gurobi().estimate(demo_X, M=1, K=2)
