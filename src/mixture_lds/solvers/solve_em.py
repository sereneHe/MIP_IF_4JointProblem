"""Runnable EM entry point."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from pyomo.environ import (
    Any,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    RangeSet,
    Reals,
    SolverFactory,
    Var,
    minimize,
)
from sklearn.metrics import f1_score


def log_info(section: str, message) -> None:
    print(f"[INFO][{section}] {message}")


def load_labels(path: str) -> np.ndarray:
    label_path = Path(path)
    if label_path.suffix == ".csv":
        labels = np.loadtxt(label_path, delimiter=",")
    else:
        labels = np.load(label_path)
    return np.asarray(labels).reshape(-1)


def obj_function_2(model, reg: float = 0.5):
    obj_1 = sum((model.K[k][t, m] - model.f[m, t]) ** 2 for k in model.KK for m in model.M for t in model.T1)
    obj_2 = sum((model.L[l][t, m] - model.ff[m, t]) ** 2 for l in model.LL for m in model.M for t in model.T1)
    obj_3 = reg * sum(model.p[m, t] ** 2 for m in model.M for t in model.T1)
    obj_3 += reg * sum(model.q[n, t] ** 2 for n in model.N for t in model.T1)
    obj_4 = reg * sum(model.pp[m, t] ** 2 for m in model.M for t in model.T1)
    obj_4 += reg * sum(model.qq[n, t] ** 2 for n in model.N for t in model.T1)
    return obj_1 + obj_2 + obj_3 + obj_4


def simcom(K, L, T: int, M: int, N: int, reg: float, option: str = "bonmin", normalize_g: bool = False):
    """Fit two LDS prototypes for the current EM cluster split."""
    log_info("EM", f"current cluster sizes: left={len(K)}, right={len(L)}")

    model = ConcreteModel()
    model.M = RangeSet(0, M - 1)
    model.N = RangeSet(0, N - 1)
    model.KK = RangeSet(0, len(K) - 1)
    model.LL = RangeSet(0, len(L) - 1)
    model.T1 = RangeSet(0, T - 1)
    model.T2 = RangeSet(0, T)

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

    model.K = Param(model.KK, initialize=K, within=Any, mutable=False)
    model.L = Param(model.LL, initialize=L, within=Any, mutable=False)
    model.OBJ = Objective(rule=lambda m: obj_function_2(m, reg=reg), sense=minimize)

    model.con1 = Constraint(expr=sum(model.f[m, t] - sum(model.Fdash[m, n] * model.m[n, t + 1] for n in model.N) - model.p[m, t] for m in model.M for t in model.T1) >= 0)
    model.con2 = Constraint(expr=sum(model.f[m, t] - sum(model.Fdash[m, n] * model.m[n, t + 1] for n in model.N) - model.p[m, t] for m in model.M for t in model.T1) <= 0)
    model.con3 = Constraint(expr=sum(model.m[n, t + 1] - sum(model.G[n, n2] * model.m[n2, t] for n2 in model.N) - model.q[n, t] for n in model.N for t in model.T1) >= 0)
    model.con4 = Constraint(expr=sum(model.m[n, t + 1] - sum(model.G[n, n2] * model.m[n2, t] for n2 in model.N) - model.q[n, t] for n in model.N for t in model.T1) <= 0)
    model.con5 = Constraint(expr=sum(model.ff[m, t] - sum(model.FFdash[m, n] * model.mm[n, t + 1] for n in model.N) - model.pp[m, t] for m in model.M for t in model.T1) >= 0)
    model.con6 = Constraint(expr=sum(model.ff[m, t] - sum(model.FFdash[m, n] * model.mm[n, t + 1] for n in model.N) - model.pp[m, t] for m in model.M for t in model.T1) <= 0)
    model.con7 = Constraint(expr=sum(model.mm[n, t + 1] - sum(model.GG[n, n2] * model.mm[n2, t] for n2 in model.N) - model.qq[n, t] for n in model.N for t in model.T1) >= 0)
    model.con8 = Constraint(expr=sum(model.mm[n, t + 1] - sum(model.GG[n, n2] * model.mm[n2, t] for n2 in model.N) - model.qq[n, t] for n in model.N for t in model.T1) <= 0)
    if normalize_g:
        model.con9 = Constraint(expr=sum(model.G[i, j] for i in model.N for j in model.N) - 1 == 0)
        model.con10 = Constraint(expr=sum(model.GG[i, j] for i in model.N for j in model.N) - 1 == 0)

    SolverFactory(option).solve(model)
    return model


def simcom_norm(K, L, T: int, M: int, N: int, reg: float, option: str = "bonmin"):
    return simcom(K, L, T=T, M=M, N=N, reg=reg, option=option, normalize_g=True)


def simcom_gurobi(K, L, T: int, M: int, N: int, reg: float, normalize_g: bool = False, time_limit: int = 3600):
    """Fit two LDS prototypes for the current EM cluster split with Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB

    log_info("EM-GUROBI", f"current cluster sizes: left={len(K)}, right={len(L)}")

    K = np.asarray(K, dtype=float)
    L = np.asarray(L, dtype=float)
    KK = K.shape[0]
    LL = L.shape[0]

    env = gp.Env()
    env.setParam("TimeLimit", time_limit)
    model = gp.Model(env=env)

    f0 = model.addVars(M, T, name="f0", vtype="C")
    phi0 = model.addVars(N, T + 1, name="phi0", vtype="C")
    G0 = model.addVars(N, N, name="G0", vtype="C")
    F0 = model.addVars(M, N, name="F0", vtype="C")
    q0 = model.addVars(N, T, name="q0", vtype="C")
    p0 = model.addVars(M, T, name="p0", vtype="C")

    f1 = model.addVars(M, T, name="f1", vtype="C")
    phi1 = model.addVars(N, T + 1, name="phi1", vtype="C")
    G1 = model.addVars(N, N, name="G1", vtype="C")
    F1 = model.addVars(M, N, name="F1", vtype="C")
    q1 = model.addVars(N, T, name="q1", vtype="C")
    p1 = model.addVars(M, T, name="p1", vtype="C")

    obj = gp.quicksum((float(K[k, t, m]) - f0[m, t]) ** 2 for k in range(KK) for t in range(T) for m in range(M))
    obj += gp.quicksum((float(L[l, t, m]) - f1[m, t]) ** 2 for l in range(LL) for t in range(T) for m in range(M))
    obj += gp.quicksum(reg * q0[n, t] ** 2 for n in range(N) for t in range(T))
    obj += gp.quicksum(reg * p0[m, t] ** 2 for m in range(M) for t in range(T))
    obj += gp.quicksum(reg * q1[n, t] ** 2 for n in range(N) for t in range(T))
    obj += gp.quicksum(reg * p1[m, t] ** 2 for m in range(M) for t in range(T))
    model.setObjective(obj, GRB.MINIMIZE)

    for t in range(T):
        for m in range(M):
            model.addConstr(f0[m, t] == gp.quicksum(F0[m, n] * phi0[n, t + 1] for n in range(N)) + p0[m, t])
            model.addConstr(f1[m, t] == gp.quicksum(F1[m, n] * phi1[n, t + 1] for n in range(N)) + p1[m, t])
    for t in range(T):
        for n in range(N):
            model.addConstr(phi0[n, t + 1] == gp.quicksum(G0[n, nn] * phi0[nn, t] for nn in range(N)) + q0[n, t])
            model.addConstr(phi1[n, t + 1] == gp.quicksum(G1[n, nn] * phi1[nn, t] for nn in range(N)) + q1[n, t])

    if normalize_g:
        model.addConstr(gp.quicksum(G0[i, j] for i in range(N) for j in range(N)) == 1)
        model.addConstr(gp.quicksum(G1[i, j] for i in range(N) for j in range(N)) == 1)

    model.update()
    model.Params.NonConvex = 2
    model.optimize()
    model._pred_k = f0
    model._pred_l = f1
    log_info("EM-GUROBI", "solver finished with an optimal solution." if model.status == GRB.Status.OPTIMAL else "solver finished without proving optimality.")
    return model


def simcom_norm_gurobi(K, L, T: int, M: int, N: int, reg: float, time_limit: int = 3600):
    return simcom_gurobi(K, L, T=T, M=M, N=N, reg=reg, normalize_g=True, time_limit=time_limit)


def get_series_value(container, m: int, t: int):
    """Read a scalar variable value from either a Pyomo or Gurobi container."""
    value = container[m, t]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "X"):
        return value.X
    return float(value)


def em_estimate(
    data_in: np.ndarray,
    label_in: np.ndarray,
    N: int,
    reg: float,
    seed: int,
    option: str = "bonmin",
    norm: bool = True,
    time_limit: int = 3600,
) -> float:
    data_in = np.asarray(data_in)
    label_in = np.asarray(label_in)
    M = data_in.shape[2]
    T = data_in.shape[1]
    log_info("DATA", f"input labels: {label_in.tolist()}")
    log_info("DATA", f"input shape: samples={len(data_in)}, time_points={T}, channels={M}")

    np.random.seed(seed)
    idx = np.random.randint(0, 2, len(data_in))
    K = [data_in[i] for i in np.where(idx == 0)[0]]
    L = [data_in[i] for i in np.where(idx == 1)[0]]
    log_info("EM", f"initial random split sizes: left={len(K)}, right={len(L)}")

    for i in range(data_in.shape[1]):
        label_out = deepcopy(idx)
        log_info("EM", f"iteration {i}: starting labels = {label_out.tolist()}")

        if option == "gurobi":
            if norm:
                model = simcom_norm_gurobi(K, L, T=T, M=M, N=N, reg=reg, time_limit=time_limit)
            else:
                model = simcom_gurobi(K, L, T=T, M=M, N=N, reg=reg, normalize_g=False, time_limit=time_limit)
        else:
            if norm:
                model = simcom_norm(K, L, T=T, M=M, N=N, reg=reg, option=option)
            else:
                model = simcom(K, L, T=T, M=M, N=N, reg=reg, option=option)
        pred_k = model._pred_k if hasattr(model, "_pred_k") else model.f
        pred_l = model._pred_l if hasattr(model, "_pred_l") else model.ff

        for n in range(len(data_in)):
            cost_k = sum((data_in[n][t, m] - get_series_value(pred_k, m, t)) ** 2 for m in range(M) for t in range(T))
            cost_l = sum((data_in[n][t, m] - get_series_value(pred_l, m, t)) ** 2 for m in range(M) for t in range(T))
            idx[n] = 0 if cost_k < cost_l else 1

        log_info("EM", f"iteration {i}: updated labels = {idx.tolist()}")
        if np.array_equal(label_out, idx):
            log_info("EM", f"iteration {i}: assignments stabilized.")
            break
        label_out = deepcopy(idx)
        K = [data_in[i] for i in np.where(idx == 0)[0]]
        L = [data_in[i] for i in np.where(idx == 1)[0]]
        log_info("EM", f"iteration {i}: cluster sizes now left={len(K)}, right={len(L)}")

    f1 = max(f1_score(label_in, label_out), f1_score(label_in, 1 - label_out))
    log_info("RESULT", f"final F1 score: {f1:.6f}")
    return float(f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--name", default="EEG")
    parser.add_argument("--regularization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=1)
    parser.add_argument("--option", default="bonmin")
    parser.add_argument("--time-limit", type=int, default=3600)
    args = parser.parse_args()

    data_in = np.load(args.data)
    label_in = load_labels(args.label)

    f1 = em_estimate(
        data_in,
        label_in,
        N=args.hidden_dim,
        reg=args.regularization,
        seed=args.seed,
        option=args.option,
        norm=True,
        time_limit=args.time_limit,
    )
    method_name = "EM-Gurobi" if args.option == "gurobi" else "EM"
    log_info("RESULT", {"method": method_name, "f1": f1})


if __name__ == "__main__":
    main()
