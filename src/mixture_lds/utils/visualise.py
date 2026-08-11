"""Result summary and plotting helpers for MIP4Cluster experiments."""

from __future__ import annotations

import os
import re

import numpy as np

DISPLAY_NAME_MAP = {
    "if_gurobi": "IF-Gurobi",
    "if-gurobi": "IF-Gurobi",
    "IF-Gurobi": "IF-Gurobi",
    "if_gurobi_k": "IF-Gurobi-K",
    "if-gurobi-k": "IF-Gurobi-K",
    "IF-Gurobi-K": "IF-Gurobi-K",
    "if": "IF-Bonmin",
    "IF": "IF-Bonmin",
    "if_bonmin": "IF-Bonmin",
    "if-bonmin": "IF-Bonmin",
    "em": "EM",
    "EM": "EM",
    "fft": "FFT",
    "FFT": "FFT",
    "dtw": "DTW",
    "DTW": "DTW",
    "mosek": "MOSEK",
    "MOSEK": "MOSEK",
}


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
        from matplotlib.backends.backend_pdf import PdfPages
        from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

        del PdfPages, HostAxes, ParasiteAxes

        colormap = (
            "#4292c6",
            "#696969",
            "#CD5C5C",
            "#FFD700",
            "#6B8E23",
            "#8c564b",
            "#17becf",
            "#9467bd",
        )
        labels = [DISPLAY_NAME_MAP.get(method, method) for method in methods]
        mean = []
        std = []
        for method in methods:
            if cutdown is True:
                mean_arr = np.load(f"{path}f1_{method}_{name}_mean_cd.npy")
                std_arr = np.load(f"{path}f1_{method}_{name}_std_cd.npy")
            else:
                mean_arr = np.load(f"{path}f1_{method}_{name}_mean.npy")
                std_arr = np.load(f"{path}f1_{method}_{name}_std.npy")
            mean.append(np.asarray(mean_arr, dtype=float).reshape(-1))
            std.append(np.asarray(std_arr, dtype=float).reshape(-1))

        point_counts = {arr.size for arr in mean}
        if len(point_counts) != 1:
            raise ValueError(
                f"Inconsistent mean shapes for plotting: {[arr.shape for arr in mean]}"
            )
        if {arr.size for arr in std} != point_counts:
            raise ValueError(
                f"Inconsistent std shapes for plotting: {[arr.shape for arr in std]}"
            )

        mean = np.vstack(mean)
        std = np.vstack(std)
        print([mean.shape, std.shape])

        num_methods, num_points = mean.shape
        fig_width = max(8, 1.2 * num_methods if num_points == 1 else 8)
        fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=200)

        if num_points == 1:
            x = np.arange(num_methods)
            y = mean[:, 0]
            y_error = std[:, 0]
            colors = [colormap[i % len(colormap)] for i in range(num_methods)]
            ax.bar(x, y, yerr=y_error, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=14)
            ax.set_xlabel("methods", fontsize=16)
        else:
            nx_range = np.arange(2, 2 + num_points) if num_points == 3 else np.arange(1, num_points + 1)
            width = 0.8 / max(num_methods, 1)
            for m in range(num_methods):
                x = nx_range + m * width
                y = mean[m, :]
                y_error = 1.96 * std[m, :] / np.sqrt(50)
                color = colormap[m % len(colormap)]
                ax.plot(x, y, color=color, label=labels[m])
                ax.errorbar(
                    x,
                    y,
                    yerr=y_error,
                    fmt=".",
                    color=color,
                    capsize=4,
                    capthick=2,
                )
            ax.legend(fontsize=14, frameon=False, ncol=1, loc="lower right")
            ax.set_xticks(nx_range + ((num_methods - 1) * width / 2 if num_methods > 1 else 0))
            ax.set_xticklabels([str(v) for v in nx_range], fontsize=14)
            ax.set_xlabel("dimensions of system matrices " + r"$n$", fontsize=16)

        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0], fontsize=14)
        ax.set_ylabel("F1 score", fontsize=16)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        os.makedirs("./reports/figures", exist_ok=True)
        plt.savefig(f"./reports/figures/{name}_f1.png", bbox_inches="tight")
