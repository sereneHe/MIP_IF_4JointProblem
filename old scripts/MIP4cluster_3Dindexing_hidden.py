"""Binary two-cluster LDS Gurobi model with M-dimensional 3D indexing.

This script is the cleaned standalone version of the notebook's
`class LDS_Gurobi(object)` idea:

- It keeps the extra model-copy dimension `m_ in range(M)`.
- It remains a two-cluster model through `L[m_, 0, l]` and `1 - L[m_, 0, l]`.
- It fixes the notebook draft's `super()` call.
- It uses `u[m_, l, t]` instead of repeatedly constraining one `u[m_, l, l]`
  across all time points.
- It returns labels instead of only printing the objective.
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB, tupledict


class LDS_Gurobi(object):
    """Two-cluster LDS estimator based on Gurobi with 3D indexing."""

    def __init__(self, **kwargs):
        super(LDS_Gurobi, self).__init__()

    def estimate(self, X, M, K=2, hidden_dims=(2, 4)):
        """Fit a two-cluster LDS model with `M` independent model copies.

        Parameters
        ----------
        X: array-like or pandas.DataFrame
            Input data with shape `(T, nL)`, where rows are time points and
            columns are series/groups.
        M: int
            Number of model copies in the 3D indexed formulation.
        K: int, default 2
            Kept only as a guard for compatibility with the notebook signature.
            This model is binary and requires `K == 2`.
        hidden_dims: tuple[int, int], default (2, 4)
            Hidden-state dimensions for cluster 0 and cluster 1.

        Returns
        -------
        model: gurobipy.Model
            Optimized Gurobi model.
        label_out: numpy.ndarray
            Array with shape `(M, nL)`. `label_out[m_, l]` is the cluster id
            for series `l` under model copy `m_`.
        """
        if K != 2:
            raise ValueError("LDS_Gurobi is a binary two-cluster model; use K=2.")
        if M < 1:
            raise ValueError("M must be at least 1.")
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two dimensions.")

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must be 2D with shape (T, nL).")

        T = X_array.shape[0]
        nL = X_array.shape[1]
        X_transposed = X_array.T

        env = gp.Env()
        env.setParam("TimeLimit", 6 * 60)
        model = gp.Model(env=env)

        # Convert X to tupledict with 3D index.
        X_data = [
            ((m_, l, t), X_transposed[l, t])
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        ]
        X_data = tupledict(X_data)

        n0, n1 = hidden_dims

        # Assignment and LDS variables.
        L = model.addVars(M, 1, nL, name="L", vtype=GRB.BINARY)
        G0 = model.addVars(M, n0, n0, name="G0", vtype=GRB.CONTINUOUS)
        phi0 = model.addVars(M, n0, T + 1, name="phi0", vtype=GRB.CONTINUOUS)
        p0 = model.addVars(M, n0, T, name="p0", vtype=GRB.CONTINUOUS)
        F0 = model.addVars(M, nL, n0, name="F0", vtype=GRB.CONTINUOUS)
        q = model.addVars(M, nL, T, name="q", vtype=GRB.CONTINUOUS)
        f0 = model.addVars(M, nL, T, name="f0", vtype=GRB.CONTINUOUS)

        G1 = model.addVars(M, n1, n1, name="G1", vtype=GRB.CONTINUOUS)
        F1 = model.addVars(M, nL, n1, name="F1", vtype=GRB.CONTINUOUS)
        phi1 = model.addVars(M, n1, T + 1, name="phi1", vtype=GRB.CONTINUOUS)
        p1 = model.addVars(M, n1, T, name="p1", vtype=GRB.CONTINUOUS)
        f1 = model.addVars(M, nL, T, name="f1", vtype=GRB.CONTINUOUS)

        v0 = model.addVars(M, nL, T, name="v0", vtype=GRB.CONTINUOUS)
        v1 = model.addVars(M, nL, T, name="v1", vtype=GRB.CONTINUOUS)
        X0 = model.addVars(M, nL, T, name="X0", vtype=GRB.CONTINUOUS)
        X1 = model.addVars(M, nL, T, name="X1", vtype=GRB.CONTINUOUS)
        u = model.addVars(M, nL, T, name="u", vtype=GRB.CONTINUOUS)

        obj = gp.quicksum(
            (X0[m_, l, t] - f0[m_, l, t]) ** 2
            + (X1[m_, l, t] - f1[m_, l, t]) ** 2
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0005 * p0[m_, n_, t] ** 2
            for m_ in range(M)
            for n_ in range(n0)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0001 * L[m_, 0, l] * u[m_, l, t]
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0005 * p1[m_, n_, t] ** 2
            for m_ in range(M)
            for n_ in range(n1)
            for t in range(T)
        )
        obj += gp.quicksum(
            0.0001 * (1 - L[m_, 0, l]) * u[m_, l, t]
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )

        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints with 3D indexing.
        model.addConstrs(
            (u[m_, l, t] == q[m_, l, t] ** 2)
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        model.addConstrs(
            (
                phi0[m_, n_, t + 1]
                == G0[m_, n_, n_] * phi0[m_, n_, t] + p0[m_, n_, t]
            )
            for m_ in range(M)
            for n_ in range(n0)
            for t in range(T)
        )
        model.addConstrs(
            (X0[m_, l, t] == L[m_, 0, l] * X_data[m_, l, t])
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        model.addConstrs(
            (
                v0[m_, l, t]
                == gp.quicksum(
                    F0[m_, l, n_] * phi0[m_, n_, t + 1]
                    for n_ in range(n0)
                )
            )
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        model.addConstrs(
            (
                f0[m_, l, t]
                == L[m_, 0, l] * v0[m_, l, t]
                + L[m_, 0, l] * q[m_, l, t]
            )
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )

        model.addConstrs(
            (
                phi1[m_, n_, t + 1]
                == G1[m_, n_, n_] * phi1[m_, n_, t] + p1[m_, n_, t]
            )
            for m_ in range(M)
            for n_ in range(n1)
            for t in range(T)
        )
        model.addConstrs(
            (X1[m_, l, t] == (1 - L[m_, 0, l]) * X_data[m_, l, t])
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        model.addConstrs(
            (
                v1[m_, l, t]
                == gp.quicksum(
                    F1[m_, l, n_] * phi1[m_, n_, t + 1]
                    for n_ in range(n1)
                )
            )
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )
        model.addConstrs(
            (
                f1[m_, l, t]
                == (1 - L[m_, 0, l]) * v1[m_, l, t]
                + (1 - L[m_, 0, l]) * q[m_, l, t]
            )
            for m_ in range(M)
            for l in range(nL)
            for t in range(T)
        )

        model.Params.NonConvex = 2
        model.optimize()

        if model.SolCount > 0:
            print(f"Optimal objective value: {model.objVal}")
        if model.status == GRB.Status.OPTIMAL:
            print("THIS IS OPTIMAL SOLUTION")
        else:
            print("THIS IS NOT OPTIMAL SOLUTION")

        if model.SolCount == 0:
            return model, np.full((M, nL), -1, dtype=int)

        L_x = model.getAttr("x", L)
        label_out = np.array(
            [
                [int(round(L_x[m_, 0, l])) for l in range(nL)]
                for m_ in range(M)
            ],
            dtype=int,
        )
        return model, label_out


if __name__ == "__main__":
    # Tiny shape demo only. Real optimization still requires a valid Gurobi license.
    demo_X = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 1.9, 3.2],
            [1.2, 2.1, 3.1],
        ]
    )
    LDS_Gurobi().estimate(demo_X, M=1)
