"""Result summary and plotting helpers for MIP4Cluster experiments."""

from __future__ import annotations

import os
import re

import numpy as np


class Visualise:
    """Plot and summarize saved MIP4Cluster results."""

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
