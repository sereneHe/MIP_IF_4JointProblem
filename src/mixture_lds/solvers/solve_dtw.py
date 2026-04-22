"""Runnable DTW entry point."""

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
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--name", default="EEG")
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    data_in = np.load(args.data)
    label_in = load_labels(args.label)

    f1 = MIP_IF().DTW_estimate(data_in, label_in, seed=args.seed, is_plot=False)
    print({"method": "DTW", "f1": f1})


if __name__ == "__main__":
    main()
