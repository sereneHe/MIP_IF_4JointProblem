"""Legacy standalone utils_MIP4Cluster module.

This file is kept under old scripts for compatibility with the historical
tests. The active implementation lives under experiment_conf/utils.
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import pandas as pd
from scipy.io import arff


class utils_MIP4Cluster:
    """Utilities for MIP4Cluster experiments."""

    # ------------------------------------------------------------------
    # Part 1: data cleaning
    # ------------------------------------------------------------------

    def datacleaning(self, data_dir, name, S, I, T, M, J):
        """Load and reshape supported datasets for MIP4Cluster tests."""
        if name == "lds":
            path_list = [
                "./data/raw/2_2_test.npy",
                "./data/raw/3_2_test.npy",
                "./data/raw/4_2_test.npy",
            ]
            X_list = []
            for path in path_list:
                data_X = np.load(path)
                data_X = data_X.reshape(2 * S, I, T, M)

                X_s = np.zeros((S, 2 * I, T, M))
                for s in range(S):
                    xx_1 = data_X[s]
                    xx_2 = data_X[s + S]
                    X_s[s] = np.concatenate((xx_1, xx_2), axis=0).reshape(
                        2 * I, T, M
                    )
                X_list.append(X_s)

            X = np.array(X_list).reshape(3, S, 2 * I, T, M)
            label = np.concatenate((np.zeros(I), np.ones(I)), axis=0)
            return X, label

        if name == "ecg":
            trainpath = "./data/raw/ECG5000_TRAIN.arff"
            X_data = self._load_arff_as_dataframe(trainpath)
            print(X_data.target.value_counts())

            X_1 = X_data[X_data.target == b"1"].iloc[:, :-1].values
            X_2 = X_data[X_data.target == b"2"].iloc[:, :-1].values
            idx_1 = np.arange(len(X_1))
            idx_2 = np.arange(len(X_2))
            iidx_1 = random.sample(sorted(idx_1), I)
            iidx_2 = random.sample(sorted(idx_2), I)
            np.random.shuffle(iidx_1)
            np.random.shuffle(iidx_2)

            XX_1 = X_1[iidx_1]
            XX_2 = X_2[iidx_2]
            X = np.concatenate((XX_1, XX_2), axis=0).reshape(S, 2 * I, T, M)
            label = np.concatenate((np.zeros(I), np.ones(I)), axis=0)
            return X, label

        if data_dir:
            return self._load_npy_directory(data_dir)

        print("Wrong data type!")
        return None

    @staticmethod
    def _load_arff_as_dataframe(path):
        data, _ = arff.loadarff(path)
        return pd.DataFrame(data)

    @staticmethod
    def _load_npy_directory(data_dir):
        """Load all .npy files from a directory into a name -> array mapping."""
        data_dict = {}
        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".npy"):
                continue
            file_path = os.path.join(data_dir, file_name)
            key_name = file_name.replace(".npy", "")
            data_dict[key_name] = np.load(file_path, allow_pickle=True)

        if not data_dict:
            print("No .npy files found in the given data directory.")
            return None

        for key, value in data_dict.items():
            if value.ndim not in (3, 4):
                print(f"Warning: unsupported shape for {key}: {value.shape}")
        return data_dict

    # ------------------------------------------------------------------
    # Part 2: data generation
    # ------------------------------------------------------------------

    def dynamic_generate(self, g, f_dash, proc_noise_std, obs_noise_std, inputs, T):
        """Generate one LDS trajectory."""
        from inputlds import dynamical_system

        n = len(g)
        m = len(f_dash)
        if inputs == 0:
            inputs = np.zeros((m, T))
        dim = len(inputs)

        ds1 = dynamical_system(
            g,
            np.zeros((n, dim)),
            f_dash,
            np.zeros((m, dim)),
            process_noise="gaussian",
            observation_noise="gaussian",
            process_noise_std=proc_noise_std,
            observation_noise_std=obs_noise_std,
        )

        h0 = np.ones(ds1.d)
        ds1.solve(h0=h0, inputs=inputs, T=T)
        return np.asarray(ds1.outputs).reshape(T, m)

    def data_generation(self, g, f_dash, pro_rang, obs_rang, T, S, output_dir="./data"):
        """Generate two-cluster LDS synthetic datasets and save them to output_dir."""
        proL = len(pro_rang)
        obsL = len(obs_rang)
        file_name = []
        os.makedirs(output_dir, exist_ok=True)

        for gg in range(len(g)):
            n = gg + 2
            m = 2
            cluster_1 = []
            cluster_2 = []

            for i in range(proL):
                for j in range(obsL):
                    proc_noise_std = pro_rang[i]
                    obs_noise_std = obs_rang[j]
                    inputs = 0
                    for _ in range(S):
                        data_1 = self.dynamic_generate(
                            g[gg, 0],
                            f_dash[gg][0],
                            proc_noise_std,
                            obs_noise_std,
                            inputs,
                            T,
                        )
                        cluster_1.append(data_1)

                        data_2 = self.dynamic_generate(
                            g[gg, 1],
                            f_dash[gg][1],
                            proc_noise_std,
                            obs_noise_std,
                            inputs,
                            T,
                        )
                        cluster_2.append(data_2)

            Y = np.concatenate((np.array(cluster_1), np.array(cluster_2)), axis=0)
            Y_label = np.concatenate(
                (np.zeros(len(cluster_1)), np.ones(len(cluster_2))), axis=0
            )
            print(Y.shape, Y_label.shape)

            data = Y.reshape(320, -1)
            f_name = os.path.join(output_dir, f"{n}_{m}_test.npy")
            with open(f_name, "wb") as f:
                np.save(f, data)
            file_name.append(f_name)

        return file_name

    # ------------------------------------------------------------------
    # Part 3: plot and result summaries
    # ------------------------------------------------------------------

    def summary_reg(self, folder_path):
        """Print regularization result summaries from saved .npy files."""
        data_dict = {
            "validation": {},
            "countvalidation": {},
            "std": {},
            "mean": {},
            "duration": {},
        }

        pattern = re.compile(r"_(\d+(?:\.\d+)?)\.npy$")
        for filename in os.listdir(folder_path):
            if not filename.endswith(".npy"):
                continue

            file_path = os.path.join(folder_path, filename)
            match = pattern.search(filename)
            if not match:
                continue

            key = match.group(1)
            if "validation" in filename:
                data_dict["validation"][key] = np.load(file_path)
                data_dict["countvalidation"][key] = sum(
                    1 for x in data_dict["validation"][key] if x != 0
                )
            elif "std" in filename:
                data_dict["std"][key] = np.load(file_path)
            elif "mean" in filename:
                data_dict["mean"][key] = np.load(file_path)
            elif "duration" in filename:
                data_dict["duration"][key] = np.load(file_path)

        for category in data_dict:
            data_dict[category] = dict(sorted(data_dict[category].items()))

        for category, values in data_dict.items():
            print(f"\nCategory: {category} ")
            for key, data in values.items():
                print(f"{key}: {data}")

    def plot_MIF4cluster_methods(self, path, methods, name, cutdown):
        """Plot F1 means/stds for the MIP4Cluster methods."""
        import matplotlib.pyplot as plt

        colormap = ("#4292c6", "#696969", "#CD5C5C", "#FFD700", "#6B8E23")
        labels = methods

        mean = []
        std = []
        for method in methods:
            if cutdown:
                mean.append(np.load(f"{path}f1_{method}_{name}_mean_cd.npy"))
                std.append(np.load(f"{path}f1_{method}_{name}_std_cd.npy"))
            else:
                mean.append(np.load(f"{path}f1_{method}_{name}_mean.npy"))
                std.append(np.load(f"{path}f1_{method}_{name}_std.npy"))

        mean = np.concatenate(mean).reshape(5, 3)
        std = np.concatenate(std).reshape(5, 3)
        print([mean.shape, std.shape])

        met_range = [*range(5)]
        nx_range = np.arange(2, 5)

        _, ax = plt.subplots(figsize=(8, 5), dpi=200)
        width = 0.1
        for m in met_range:
            x = nx_range + m * width
            y = mean[m, :]
            y_error = 1.96 * std[m, :] / np.sqrt(50)
            ax.plot(x, y, color=colormap[m], label=labels[m])
            ax.errorbar(
                x,
                y,
                yerr=y_error,
                fmt=".",
                color=colormap[m],
                capsize=4,
                capthick=2,
            )

        plt.legend(fontsize=14, frameon=False, ncol=1, loc="lower right")
        plt.yticks(
            ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            labels=[0.0, 0.25, 0.5, 0.75, 1.0],
            fontsize=14,
        )
        plt.xticks(ticks=[2.3, 3.3, 4.3], labels=[2, 3, 4], fontsize=14)
        plt.xlabel("dimensions of system matrices " + r"$n$", fontsize=16)
        plt.ylabel("F1 score", fontsize=16)
        plt.savefig(f"./reports/figures/{name}_f1.png", bbox_inches="tight")
