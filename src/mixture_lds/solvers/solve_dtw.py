"""Runnable DTW entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def log_info(section: str, message) -> None:
    print(f"[INFO][{section}] {message}")


def load_labels(path: str) -> np.ndarray:
    label_path = Path(path)
    if label_path.suffix == ".csv":
        labels = np.loadtxt(label_path, delimiter=",")
    else:
        labels = np.load(label_path)
    return np.asarray(labels).reshape(-1)


def dtw_estimate(data: np.ndarray, label: np.ndarray, seed: int = 42, is_plot: bool = False) -> float:
    """Cluster time-series samples with Soft-DTW k-means."""
    x_train = np.asarray(data)
    x_label = np.asarray(label)

    np.random.seed(seed)
    index = np.arange(len(x_train))
    np.random.shuffle(index)

    x_train = x_train[index]
    x_label = x_label[index]
    log_info("DATA", f"input labels: {x_label.tolist()}")
    log_info("DATA", f"input shape: samples={len(x_train)}, time_points={x_train.shape[1]}, channels={x_train.shape[2] if x_train.ndim == 3 else 1}")

    x_train = TimeSeriesScalerMeanVariance().fit_transform(x_train)
    log_info("DTW", f"normalized time-series length: {x_train.shape[1]}")
    log_info("DTW", "running Soft-DTW k-means.")
    sdtw_km = TimeSeriesKMeans(
        n_clusters=2,
        max_iter=10,
        metric="softdtw",
        metric_params={"gamma": 0.01},
        verbose=is_plot,
        random_state=seed,
    )
    y_pred = sdtw_km.fit_predict(x_train)
    log_info("DTW", f"predicted cluster labels: {y_pred.tolist()}")

    f1 = max(f1_score(x_label, y_pred), f1_score(x_label, 1 - y_pred))
    log_info("RESULT", f"final F1 score: {f1:.6f}")
    return float(f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--name", default="EEG")
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    data_in = np.load(args.data)
    label_in = load_labels(args.label)

    f1 = dtw_estimate(data_in, label_in, seed=args.seed, is_plot=False)
    log_info("RESULT", {"method": "DTW", "f1": f1})


if __name__ == "__main__":
    main()
