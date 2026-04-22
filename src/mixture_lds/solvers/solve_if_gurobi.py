"""Runnable IF-Gurobi entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mixture_lds.models.mip_if_3dindexing import MIP_IF


def load_labels(path: str) -> np.ndarray:
    label_path = Path(path)
    if label_path.suffix == ".csv":
        labels = np.loadtxt(label_path, delimiter=",")
    else:
        labels = np.load(label_path)
    return np.asarray(labels).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npy data with shape (N, T, M).")
    parser.add_argument("--label", required=True, help="Path to 1D label file (.npy or .csv).")
    parser.add_argument("--name", default="EEG")
    parser.add_argument("--regularization", type=int, default=270)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--thresh", type=float, default=0.25)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--gap", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=3)
    args = parser.parse_args()

    data_in = np.load(args.data)
    label_in = load_labels(args.label)

    f1, validation = MIP_IF().MIP_estimate(
        data_in,
        label_in,
        method="IF-Gurobi",
        N=args.hidden_dim,
        name=args.name,
        reg=args.regularization,
        seed=args.seed,
        thresh=args.thresh,
        time_limit=args.time_limit,
        gap=args.gap,
        shuffle=False,
    )
    print({"method": "IF-Gurobi", "f1": f1, "validation": validation})


if __name__ == "__main__":
    main()
