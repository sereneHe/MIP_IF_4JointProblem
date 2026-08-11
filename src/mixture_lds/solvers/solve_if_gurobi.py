"""Runnable IF-Gurobi entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def load_labels(path: str) -> np.ndarray:
    label_path = Path(path)
    if label_path.suffix == ".csv":
        labels = np.loadtxt(label_path, delimiter=",")
    else:
        labels = np.load(label_path)
    return np.asarray(labels).reshape(-1)


class QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"error: {message}\n")


def log_info(section: str, message) -> None:
    print(f"[INFO][{section}] {message}")


def infer_bounds(X, min_state_bound: float = 10.0, min_coeff_bound: float = 25.0):
    """Infer conservative finite bounds from the observed data scale."""
    arr = np.asarray(X, dtype=float)
    finite = arr[np.isfinite(arr)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    scale = max(scale, 1.0)
    state_bound = max(min_state_bound, 2.0 * scale)
    coeff_bound = max(min_coeff_bound, 5.0 * scale)
    residual_bound = max(1.0, (state_bound + scale) ** 2)
    return state_bound, coeff_bound, residual_bound


def ind_gurobi_function(
    X: np.ndarray,
    label: np.ndarray,
    UB: int,
    M: int,
    T: int,
    reg: float,
    time_limit: int = 3600,
    gap: float = 0.01,
):
    """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals."""
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env()
    env.setParam("TimeLimit", time_limit)
    model = gp.Model(env=env)
    T -= 1
    I = len(X)
    UB0 = UB1 = UB
    log_info("IF-GUROBI", f"model_dims UB0={UB0} UB1={UB1} T={T} M={M}")

    L = model.addVars(I, name="L", vtype="B")

    f0 = model.addVars(M, T, name="f0", vtype="C")
    phi0 = model.addVars(UB0, T + 1, name="phi0", vtype="C")
    G0 = model.addVars(UB0, UB0, name="G0", vtype="C")
    F0 = model.addVars(M, UB0, name="F0", vtype="C")
    q0 = model.addVars(UB0, T, name="q0", vtype="C")
    p0 = model.addVars(M, T, name="p0", vtype="C")
    quatr0 = model.addVars(M, T + 1, name="quatr0", vtype="C")
    quatr_hidden0 = model.addVars(UB0, T, name="quatr_hidden0", vtype="C")

    f1 = model.addVars(M, T, name="f1", vtype="C")
    phi1 = model.addVars(UB1, T + 1, name="phi1", vtype="C")
    G1 = model.addVars(UB1, UB1, name="G1", vtype="C")
    F1 = model.addVars(M, UB1, name="F1", vtype="C")
    q1 = model.addVars(UB1, T, name="q1", vtype="C")
    p1 = model.addVars(M, T, name="p1", vtype="C")
    quatr1 = model.addVars(M, T + 1, name="quatr1", vtype="C")
    quatr_hidden1 = model.addVars(UB1, T, name="quatr_hidden1", vtype="C")
    z0_squared = model.addVars(I, T, M, name="z0_squared", vtype="C")
    z1_squared = model.addVars(I, T, M, name="z1_squared", vtype="C")

    obj = gp.quicksum(
        (1 - L[i]) * z0_squared[i, t, m] + L[i] * z1_squared[i, t, m]
        for m in range(M)
        for t in range(T)
        for i in range(I)
    )
    obj += gp.quicksum(reg * q0[n, t] ** 2 for n in range(UB0) for t in range(T))
    obj += gp.quicksum(reg * q1[n, t] ** 2 for n in range(UB1) for t in range(T))
    obj += gp.quicksum(reg * p0[m, t] ** 2 + reg * p1[m, t] ** 2 for m in range(M) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstrs(z0_squared[i, t, m] == (X[i, t, m] - f0[m, t]) ** 2 for i in range(I) for t in range(T) for m in range(M))
    model.addConstrs(z1_squared[i, t, m] == (X[i, t, m] - f1[m, t]) ** 2 for i in range(I) for t in range(T) for m in range(M))

    for t in range(T):
        for m in range(M):
            model.addConstr(gp.quicksum(F0[m, n] * phi0[n, t + 1] for n in range(UB0)) == quatr0[m, t + 1])
            model.addConstr(gp.quicksum(F1[m, n] * phi1[n, t + 1] for n in range(UB1)) == quatr1[m, t + 1])
    model.addConstr(gp.quicksum(f0[m, t] - quatr0[m, t + 1] - p0[m, t] for t in range(T) for m in range(M)) == 0)
    model.addConstr(gp.quicksum(f1[m, t] - quatr1[m, t + 1] - p1[m, t] for t in range(T) for m in range(M)) == 0)
    for t in range(T):
        for n in range(UB0):
            model.addConstr(gp.quicksum(G0[n, nn] * phi0[nn, t] for nn in range(UB0)) == quatr_hidden0[n, t])
        for n in range(UB1):
            model.addConstr(gp.quicksum(G1[n, nn] * phi1[nn, t] for nn in range(UB1)) == quatr_hidden1[n, t])
    model.addConstr(gp.quicksum(phi0[n, t + 1] - quatr_hidden0[n, t] - q0[n, t] for t in range(T) for n in range(UB0)) == 0)
    model.addConstr(gp.quicksum(phi1[n, t + 1] - quatr_hidden1[n, t] - q1[n, t] for t in range(T) for n in range(UB1)) == 0)

    model.update()
    model.Params.NonConvex = 2
    model.optimize()

    log_info("IF-GUROBI", "solver finished with an optimal solution." if model.status == GRB.Status.OPTIMAL else "solver finished without proving optimality.")

    if model.SolCount == 0:
        return model, np.zeros(I, dtype=int), {"G": [[0.0] * (UB0 * UB0), [0.0] * (UB1 * UB1)], "F": [[0.0] * (M * UB0), [0.0] * (M * UB1)]}

    data_dict = {
        "G": [
            [model.getAttr("x", G0)[h, k] for (h, k) in model.getAttr("x", G0)],
            [model.getAttr("x", G1)[h, k] for (h, k) in model.getAttr("x", G1)],
        ],
        "F": [
            [model.getAttr("x", F0)[h, k] for (h, k) in model.getAttr("x", F0)],
            [model.getAttr("x", F1)[h, k] for (h, k) in model.getAttr("x", F1)],
        ],
    }
    L_x = model.getAttr("x", L)
    label_out = np.array([round(L_x[h]) for h in L_x])
    log_info("RESULT", f"predicted cluster labels: {label_out.tolist()}")
    return model, label_out, data_dict


def kcluster_ind_gurobi_function(
    X: np.ndarray,
    K: int,
    UB: int,
    M: int,
    T: int,
    reg: float,
    time_limit: int = 60,
):
    """Fit the clean K-cluster LDS Gurobi formulation for K >= 2."""
    import gurobipy as gp
    from gurobipy import GRB

    if K < 2:
        raise ValueError("K must be at least 2.")
    if K > len(X):
        raise ValueError("K cannot exceed the number of samples.")

    env = gp.Env()
    env.setParam("TimeLimit", time_limit)
    model = gp.Model(env=env)
    T -= 1
    I = len(X)
    state_bound, coeff_bound, residual_bound = infer_bounds(X)

    L = model.addVars(I, K, name="L", vtype="B")
    f = model.addVars(K, M, T, lb=-state_bound, ub=state_bound, name="f", vtype="C")
    phi = model.addVars(K, UB, T + 1, lb=-state_bound, ub=state_bound, name="phi", vtype="C")
    G = model.addVars(K, UB, UB, lb=-coeff_bound, ub=coeff_bound, name="G", vtype="C")
    F = model.addVars(K, M, UB, lb=-coeff_bound, ub=coeff_bound, name="F", vtype="C")
    q = model.addVars(K, UB, T, lb=-state_bound, ub=state_bound, name="q", vtype="C")
    p = model.addVars(K, M, T, lb=-state_bound, ub=state_bound, name="p", vtype="C")
    quatr = model.addVars(K, M, T + 1, lb=-state_bound, ub=state_bound, name="quatr", vtype="C")
    quatr_hidden = model.addVars(K, UB, T, lb=-state_bound, ub=state_bound, name="quatr_hidden", vtype="C")
    z_squared = model.addVars(K, I, T, M, lb=0.0, ub=residual_bound, name="z_squared", vtype="C")
    selected = model.addVars(K, I, T, M, lb=0.0, ub=residual_bound, name="selected", vtype="C")

    obj = gp.quicksum(selected[k, i, t, m] for k in range(K) for i in range(I) for t in range(T) for m in range(M))
    obj += gp.quicksum(reg * q[k, n, t] ** 2 for k in range(K) for n in range(UB) for t in range(T))
    obj += gp.quicksum(reg * p[k, m, t] ** 2 for k in range(K) for m in range(M) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstrs(z_squared[k, i, t, m] == (X[i, t, m] - f[k, m, t]) ** 2 for k in range(K) for i in range(I) for t in range(T) for m in range(M))
    model.addConstrs(selected[k, i, t, m] <= z_squared[k, i, t, m] for k in range(K) for i in range(I) for t in range(T) for m in range(M))
    model.addConstrs(selected[k, i, t, m] <= residual_bound * L[i, k] for k in range(K) for i in range(I) for t in range(T) for m in range(M))
    model.addConstrs(selected[k, i, t, m] >= z_squared[k, i, t, m] - residual_bound * (1 - L[i, k]) for k in range(K) for i in range(I) for t in range(T) for m in range(M))

    model.addConstrs(gp.quicksum(F[k, m, n] * phi[k, n, t + 1] for n in range(UB)) == quatr[k, m, t + 1] for k in range(K) for m in range(M) for t in range(T))
    model.addConstrs(gp.quicksum(G[k, n, nn] * phi[k, nn, t] for nn in range(UB)) == quatr_hidden[k, n, t] for k in range(K) for n in range(UB) for t in range(T))
    model.addConstrs(f[k, m, t] == quatr[k, m, t + 1] + p[k, m, t] for k in range(K) for m in range(M) for t in range(T))
    model.addConstrs(phi[k, n, t + 1] == quatr_hidden[k, n, t] + q[k, n, t] for k in range(K) for n in range(UB) for t in range(T))

    model.addConstrs(gp.quicksum(L[i, k] for k in range(K)) == 1 for i in range(I))
    model.addConstrs(gp.quicksum(L[i, k] for i in range(I)) >= 1 for k in range(K))
    model.addConstrs(gp.quicksum(L[i, k] for i in range(I)) <= I - 1 for k in range(K))
    model.addConstrs(gp.quicksum(L[i, k] for i in range(I)) >= gp.quicksum(L[i, k + 1] for i in range(I)) for k in range(K - 1))

    model.update()
    model.Params.NonConvex = 2
    model.optimize()

    log_info("IF-GUROBI-K", "solver finished with an optimal solution." if model.status == GRB.Status.OPTIMAL else "solver finished without proving optimality.")
    if model.SolCount == 0:
        return model, np.full(I, -1, dtype=int), {"G": [], "F": []}

    G_x = model.getAttr("x", G)
    F_x = model.getAttr("x", F)
    L_x = model.getAttr("x", L)
    data_dict = {
        "G": [G_x[k, h, j] for (k, h, j) in G_x],
        "F": [F_x[k, m, h] for (k, m, h) in F_x],
    }
    label_matrix = np.array([[L_x[i, k] for k in range(K)] for i in range(I)])
    label_out = np.array([max(range(K), key=lambda k: L_x[i, k]) for i in range(I)])
    log_info("RESULT", f"assignment matrix: {label_matrix.tolist()}")
    log_info("RESULT", f"predicted cluster labels: {label_out.tolist()}")
    return model, label_out, data_dict


def system_matrix(arr_G: np.ndarray, arr_F: np.ndarray, M: int, UB: int):
    arr_rounded_G = np.round(arr_G, decimals=3)
    arr_rounded_F = np.round(arr_F, decimals=3)
    arr_rounded_G[np.abs(arr_rounded_G) < 0.001] = 0
    arr_rounded_F[np.abs(arr_rounded_F) < 0.001] = 0

    if not arr_rounded_G.any() and not arr_rounded_F.any():
        log_info("RESULT", "system matrix is effectively all zeros after rounding.")
        return arr_G, arr_F, 0

    nonzero_rows_G, nonzero_cols_G = np.nonzero(arr_rounded_G.reshape(UB, UB))
    nonzero_rows_F, nonzero_cols_F = np.nonzero(arr_rounded_F.reshape(M, UB))

    def max_max(values):
        if values.size == 0:
            return 0
        return max(values)

    N = max(max_max(nonzero_rows_G), max_max(nonzero_cols_G), max_max(nonzero_cols_F)) + 1
    return arr_G[:N, :N], arr_F[:, :N], N


def if_gurobi_estimate(
    data_in: np.ndarray,
    label_in: np.ndarray,
    N: int,
    reg: float,
    seed: int,
    thresh: float = 0.25,
    shuffle: bool = True,
    time_limit: int = 3600,
    gap: float = 0.01,
):
    data_in = np.asarray(data_in)
    label_in = np.asarray(label_in)
    M = data_in.shape[2]
    T = data_in.shape[1]
    log_info("DATA", f"input labels: {label_in.tolist()}")
    log_info("DATA", f"input shape: samples={len(data_in)}, time_points={T}, channels={M}")

    if shuffle:
        np.random.seed(seed)
        idx = np.arange(len(data_in))
        np.random.shuffle(idx)
        label = label_in[idx]
        X = data_in[idx, :, :]
    else:
        label = label_in
        X = data_in

    UB0 = UB1 = N
    _, label_out, data_dict = ind_gurobi_function(
        X=X,
        label=label,
        UB=UB0,
        M=M,
        T=T,
        reg=reg,
        time_limit=time_limit,
        gap=gap,
    )

    G0 = np.array(data_dict["G"][0]).reshape(UB0, UB0)
    F0 = np.array(data_dict["F"][0]).reshape(M, UB0)
    G1 = np.array(data_dict["G"][1]).reshape(UB1, UB1)
    F1 = np.array(data_dict["F"][1]).reshape(M, UB1)
    G0, F0, N0 = system_matrix(G0, F0, M, UB=UB0)
    G1, F1, N1 = system_matrix(G1, F1, M, UB=UB1)

    log_info("RESULT", f"effective hidden-state dimension for system 1: {N0}")
    log_info("RESULT", f"effective hidden-state dimension for system 2: {N1}")
    log_info("RESULT", f"system 1 matrices:\nG0 =\n{G0}\nF0 =\n{F0}")
    log_info("RESULT", f"system 2 matrices:\nG1 =\n{G1}\nF1 =\n{F1}")

    sparse_threshold = (N * N + N * M) * thresh
    sparse = (
        np.sum(np.array(data_dict["G"][0]) == 0) >= sparse_threshold
        or np.sum(np.array(data_dict["F"][0]) == 0) >= sparse_threshold
        or np.sum(np.array(data_dict["G"][1]) == 0) >= sparse_threshold
        or np.sum(np.array(data_dict["F"][1]) == 0) >= sparse_threshold
    )
    validation = 0 if sparse else 1
    log_info("RESULT", "system matrices look sparse." if sparse else "system matrices look dense.")

    label_out = np.asarray(label_out)
    f1 = max(f1_score(label, label_out), f1_score(label, 1 - label_out))
    log_info("RESULT", f"final F1 score: {f1:.6f}")
    return float(f1), validation


def if_gurobi_kcluster_estimate(
    data_in: np.ndarray,
    label_in: np.ndarray,
    K: int,
    N: int,
    reg: float,
    time_limit: int = 3600,
):
    """Run the K-cluster Gurobi formulation and return clustering F1."""
    data_in = np.asarray(data_in)
    label_in = np.asarray(label_in)
    M = data_in.shape[2]
    T = data_in.shape[1]
    log_info("DATA", f"input labels: {label_in.tolist()}")
    log_info("DATA", f"input shape: samples={len(data_in)}, time_points={T}, channels={M}")

    _, label_out, _ = kcluster_ind_gurobi_function(
        X=data_in,
        K=K,
        UB=N,
        M=M,
        T=T,
        reg=reg,
        time_limit=time_limit,
    )
    label_out = np.asarray(label_out)
    if K == 2 and len(np.unique(label_in)) == 2:
        f1 = max(f1_score(label_in, label_out), f1_score(label_in, 1 - label_out))
    else:
        f1 = f1_score(label_in, label_out, average="macro")
    log_info("RESULT", f"final F1 score: {f1:.6f}")
    return float(f1), ""


def if_gurobi_kcluster_auto_estimate(
    data_in: np.ndarray,
    label_in: np.ndarray,
    N: int,
    reg: float,
    time_limit: int = 3600,
    k_min: int = 2,
    k_max: int | None = None,
    penalty_weight: float = 1.0,
):
    """Select K automatically by a penalized objective across candidate Ks.

    Note:
        The raw optimization objective is not suitable for choosing K directly,
        because it tends to improve as K grows. We therefore add a simple
        complexity penalty proportional to K * log(number of observations).
    """
    data_in = np.asarray(data_in)
    label_in = np.asarray(label_in)
    M = data_in.shape[2]
    T = data_in.shape[1]
    I = data_in.shape[0]
    log_info("DATA", f"input labels: {label_in.tolist()}")
    log_info("DATA", f"input shape: samples={len(data_in)}, time_points={T}, channels={M}")

    if k_max is None:
        k_max = min(max(2, len(np.unique(label_in)) + 2), max(2, I - 1))
    k_min = max(2, int(k_min))
    k_max = max(k_min, min(int(k_max), I - 1))
    obs_count = max(1, I * T * M)

    best = None
    trials = []
    for K in range(k_min, k_max + 1):
        model, label_out, _ = kcluster_ind_gurobi_function(
            X=data_in,
            K=K,
            UB=N,
            M=M,
            T=T,
            reg=reg,
            time_limit=time_limit,
        )
        if getattr(model, "SolCount", 0) == 0:
            log_info("IF-GUROBI-K", f"auto-K trial K={K}: no feasible solution, skipped.")
            continue
        obj = float(model.objVal)
        penalty = float(penalty_weight * K * np.log(obs_count))
        score = obj + penalty
        label_out = np.asarray(label_out)
        if K == 2 and len(np.unique(label_in)) == 2:
            f1 = max(f1_score(label_in, label_out), f1_score(label_in, 1 - label_out))
        else:
            f1 = f1_score(label_in, label_out, average="macro")
        trial = {
            "K": K,
            "obj": obj,
            "penalty": penalty,
            "score": score,
            "f1": float(f1),
            "label_out": label_out,
        }
        trials.append(trial)
        log_info("IF-GUROBI-K", f"auto-K trial K={K}: objective={obj:.6f}, penalty={penalty:.6f}, score={score:.6f}, f1={float(f1):.6f}")
        if best is None or score < best["score"]:
            best = trial

    if best is None:
        raise RuntimeError("Auto-K search found no feasible K.")

    log_info("RESULT", f"auto-K selected K={best['K']} with score={best['score']:.6f} and f1={best['f1']:.6f}")
    return float(best["f1"]), "", int(best["K"])


def main() -> None:
    parser = QuietArgumentParser(add_help=True)
    parser.add_argument("--data", required=True, help="Path to .npy data with shape (N, T, M).")
    parser.add_argument("--label", required=True, help="Path to 1D label file (.npy or .csv).")
    parser.add_argument("--name", default="EEG")
    parser.add_argument("--regularization", type=float, default=270.0)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--thresh", type=float, default=0.25)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--gap", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=3)
    parser.add_argument("--cluster-mode", choices=["2", "k"], default="2")
    parser.add_argument("--clusters", default="2")
    parser.add_argument("--cluster-min", type=int, default=2)
    parser.add_argument("--cluster-max", type=int, default=0)
    parser.add_argument("--cluster-penalty", type=float, default=1.0)
    args = parser.parse_args()

    data_in = np.load(args.data)
    label_in = load_labels(args.label)

    if args.cluster_mode == "k":
        if str(args.clusters).lower() == "auto":
            f1, validation, selected_k = if_gurobi_kcluster_auto_estimate(
                data_in,
                label_in,
                N=args.hidden_dim,
                reg=args.regularization,
                time_limit=args.time_limit,
                k_min=args.cluster_min,
                k_max=(args.cluster_max if args.cluster_max > 0 else None),
                penalty_weight=args.cluster_penalty,
            )
            log_info("RESULT", {"method": "IF-Gurobi-K", "f1": f1, "validation": validation, "selected_k": selected_k})
        else:
            f1, validation = if_gurobi_kcluster_estimate(
                data_in,
                label_in,
                K=int(args.clusters),
                N=args.hidden_dim,
                reg=args.regularization,
                time_limit=args.time_limit,
            )
            log_info("RESULT", {"method": "IF-Gurobi-K", "f1": f1, "validation": validation})
    else:
        f1, validation = if_gurobi_estimate(
            data_in,
            label_in,
            N=args.hidden_dim,
            reg=args.regularization,
            seed=args.seed,
            thresh=args.thresh,
            time_limit=args.time_limit,
            gap=args.gap,
            shuffle=False,
        )
        log_info("RESULT", {"method": "IF-Gurobi", "f1": f1, "validation": validation})


if __name__ == "__main__":
    main()
