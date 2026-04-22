"""Baseline comparison helper for 3D EEG data.

This is the 3D-data cousin of `MIP4cluster.py`, but it delegates the actual
estimation to `MIP_IF_3Dindexing.py` and performs seed-based shuffling on
`X = data_in[idx, :, :]` before each run.
"""

from __future__ import annotations

import os
import time
import importlib.util
from pathlib import Path

import numpy as np


class MIP4KCluster:
    """Run baseline methods on 3D EEG-style data."""

    def __init__(self):
        super().__init__()

    def setup_environment(self, method: str) -> None:
        """Lightweight dependency checks for the selected baseline method."""
        if method == "IF-Gurobi":
            if not importlib.util.find_spec("gurobipy") or not importlib.util.find_spec("sklearn"):
                raise ImportError("IF-Gurobi requires gurobipy and sklearn.")
            os.environ.setdefault("GRB_LICENSE_FILE", "./gurobi.lic")
        elif method in {"EM", "IF"}:
            return
        else:
            if not importlib.util.find_spec("tslearn"):
                print("Please install tslearn package for DTW and FFT methods.")

    def _run_one_seed(self, data_in, label_in, method, name, reg, seed, thresh=0.25):
        from MIP_IF_3Dindexing import MIP_IF

        np.random.seed(seed)
        idx = np.arange(len(data_in))
        np.random.shuffle(idx)
        X = data_in[idx, :, :]
        label = label_in[idx]

        print(f"[seed={seed}] X.shape={X.shape}, label.shape={label.shape}")

        if method == "IF-Gurobi":
            f1_result, valid_result = MIP_IF().MIP_estimate(
                X,
                label,
                method,
                N=1,
                name=name,
                reg=reg,
                seed=seed,
                thresh=thresh,
                shuffle=False,
            )
            return f1_result, valid_result

        if method == "IF":
            return MIP_IF().MIP_estimate(
                X,
                label,
                method,
                N=1,
                name=name,
                reg=reg,
                seed=seed,
                shuffle=False,
            ), None

        if method == "EM":
            return MIP_IF().MIP_estimate(
                X,
                label,
                method,
                N=1,
                name=name,
                reg=reg,
                seed=seed,
                MTS=True,
                option="bonmin",
                norm=True,
                shuffle=False,
            ), None

        if method == "DTW":
            return MIP_IF().DTW_estimate(X, label, seed=seed, is_plot=False), None

        return MIP_IF().FFT_estimate(X, label, seed=seed, is_plot=False), None

    def Test_MIP4KCluster(
        self,
        data_in,
        label_in,
        met,
        name,
        regularization=270,
        seed_times=5,
        seed_start=30,
        thresh=0.25,
    ):
        """Compare baseline methods over 5 shuffled seeds on 3D data."""
        output_dir = Path(f"./Result_{name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        data_in = np.asarray(data_in)
        label_in = np.asarray(label_in)
        if data_in.ndim != 3:
            raise ValueError("data_in must have shape (N, T, M).")
        if label_in.ndim != 1:
            label_in = label_in.reshape(-1)

        print("data_in.shape:", data_in.shape, "label_in.shape:", label_in.shape)

        if isinstance(regularization, (list, tuple, np.ndarray)):
            reg = int(np.asarray(regularization).reshape(-1)[0])
        else:
            reg = int(regularization)

        for method in met:
            if method not in {"IF-Gurobi", "IF", "EM", "FFT", "DTW"}:
                raise ValueError(f"Unsupported method: {method}")

            self.setup_environment(method)
            f1_list = []
            validation = []
            duration = []

            t_start = time.time()
            for s in range(seed_times):
                seed = seed_start + s
                result = self._run_one_seed(
                    data_in=data_in,
                    label_in=label_in,
                    method=method,
                    name=name,
                    reg=reg,
                    seed=seed,
                    thresh=thresh,
                )
                f1_result, valid_result = result
                f1_list.append(f1_result)
                if valid_result is not None:
                    validation.append(valid_result)

            duration.append(time.time() - t_start)

            f1 = np.array(f1_list)
            f1_mean = float(np.mean(f1))
            f1_std = float(np.std(f1))

            print(
                f"method={method}, name={name}, reg={reg}, "
                f"f1={f1}, mean={f1_mean}, std={f1_std}, duration={duration}"
            )

            np.save(output_dir / f"f1_{method}_{name}.npy", f1)
            np.save(output_dir / f"f1_{method}_{name}_mean.npy", f1_mean)
            np.save(output_dir / f"f1_{method}_{name}_std.npy", f1_std)
            np.save(output_dir / f"duration_{method}_{name}.npy", np.array(duration))
            if validation:
                np.save(output_dir / f"validation_{method}_{name}.npy", np.array(validation))

        print(f"Check result under route {output_dir}/")


MIP4_Kcluster = MIP4KCluster
