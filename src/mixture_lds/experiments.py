"""Small experiment dispatcher for Joint-Problem cluster baselines.

The scripts in ``scripts/`` call this file with Hydra-like ``key=value`` arguments.
It intentionally keeps the parser lightweight and routes work to the existing
``mixture_lds.solvers.solve_*`` entry points.
"""

from __future__ import annotations

import ast
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
PYTHON_BIN = sys.executable

from mixture_lds.data.preprocessing import DataPreprocessing
from mixture_lds.utils.visualise import Visualise


SOLVER_METHOD = {
    "if_gurobi": "IF-Gurobi",
    "if-gurobi": "IF-Gurobi",
    "IF-Gurobi": "IF-Gurobi",
    "if": "IF",
    "IF": "IF",
    "em": "EM",
    "EM": "EM",
    "dtw": "DTW",
    "DTW": "DTW",
    "fft": "FFT",
    "FFT": "FFT",
}

SOLVER_MODULE = {
    "if_gurobi": "mixture_lds.solvers.solve_if_gurobi",
    "if": "mixture_lds.solvers.solve_if",
    "em": "mixture_lds.solvers.solve_em",
    "dtw": "mixture_lds.solvers.solve_dtw",
    "fft": "mixture_lds.solvers.solve_fft",
}


def clean_value(value: str) -> str:
    return value.strip().strip("'").strip('"')


def parse_overrides(argv: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in argv:
        if item.startswith("--"):
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        overrides[key] = clean_value(value)
    return overrides


def expand_value(value: str):
    value = clean_value(value)
    if value.startswith("range(") and value.endswith(")"):
        args = [int(x.strip()) for x in value[6:-1].split(",")]
        return list(range(*args))
    if "," in value:
        return [clean_value(part) for part in value.split(",") if clean_value(part)]
    return [value]


def as_int(overrides: dict[str, str], key: str, default: int) -> int:
    return int(clean_value(overrides.get(key, str(default))))


def as_float(overrides: dict[str, str], key: str, default: float) -> float:
    return float(clean_value(overrides.get(key, str(default))))


def solver_key(name: str) -> str:
    method = SOLVER_METHOD.get(name, name)
    for key, value in SOLVER_METHOD.items():
        if value == method and key in SOLVER_MODULE:
            return key
    raise ValueError(f"Unsupported solver: {name}")


def _prepare_chbmit_grouped_file(prep: DataPreprocessing) -> tuple[Path, Path]:
    processed_dir = PROJECT_DIR / "data" / "processed" / "EEG"
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_path = processed_dir / "eeg_seizure.npy"
    label_path = processed_dir / "eeg_seizure_label.npy"

    if data_path.exists() and label_path.exists():
        return data_path, label_path

    grouped_source = processed_dir / "eeg_seizure_X.csv.npy"
    if grouped_source.exists():
        X = np.load(grouped_source, allow_pickle=True)
        X = np.asarray(X, dtype=float)
        if X.ndim != 4:
            raise ValueError(f"Expected 4D CHB-MIT grouped data, got {X.shape} from {grouped_source}")
        X = X[:, :, :, :3]
        y_one_subject = np.concatenate((np.ones(7, dtype=int), np.zeros(7, dtype=int)))
        y = np.tile(y_one_subject, (X.shape[0], 1))
        np.save(data_path, X)
        np.save(label_path, y)
        np.savetxt(processed_dir / "eeg_seizure_label.csv", y, delimiter=",", fmt="%d")
        return data_path, label_path

    raw_root = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "EEG"
        / "archive.physionet.org"
        / "pn6"
        / "chbmit"
    )
    exported = prep.export_default_chbmit_seizure_prediction_data(
        raw_root=raw_root,
        project_dir=PROJECT_DIR,
        subjects=(1, 3, 5),
        window_size=50,
        sampling_rate=256,
        selected_channels=("FP1-F3", "FP2-F4", "FP2-F8"),
    )
    return Path(exported["data_file"]), Path(exported["label_file"])


def prepare_eeg(overrides: dict[str, str], result_dir: Path) -> tuple[Path, Path, str]:
    prep = DataPreprocessing()
    eeg_mode = clean_value(overrides.get("problem.eeg_mode", overrides.get("eeg_mode", "test"))).lower()
    n_normal = as_int(overrides, "problem.n_normal", 5)
    n_seizure = as_int(overrides, "problem.n_seizure", 5)
    time_points = as_int(overrides, "problem.time_points", 100)
    channels = as_int(overrides, "problem.channels", 5)
    result_dir.mkdir(parents=True, exist_ok=True)
    if eeg_mode in {"eeg_27s", "27s", "packed_27s"}:
        packed_path = PROJECT_DIR / "data" / "processed" / "EEG" / "eeg-27s.npy"
        packed = np.load(packed_path, allow_pickle=True).item()
        X = packed["x"]
        y = packed["y"]
        data_path = result_dir / "EEG_27s.npy"
        label_path = result_dir / "EEG_27s_label.npy"
        np.save(data_path, X)
        np.save(label_path, y)
        print("EEG 27s packed shape:", X.shape, y.shape)
        return data_path, label_path, "EEG_27s"

    if eeg_mode in {"eeg_120s", "120s", "packed_120s"}:
        packed_path = PROJECT_DIR / "data" / "processed" / "EEG" / "eeg-120s.npy"
        packed = np.load(packed_path, allow_pickle=True).item()
        X = packed["x"]
        y = packed["y"]
        data_path = result_dir / "EEG_120s.npy"
        label_path = result_dir / "EEG_120s_label.npy"
        np.save(data_path, X)
        np.save(label_path, y)
        print("EEG 120s packed shape:", X.shape, y.shape)
        return data_path, label_path, "EEG_120s"

    if eeg_mode in {"chbmit", "raw", "full", "prediction"}:
        data_path, label_path = _prepare_chbmit_grouped_file(prep)
        X = np.load(data_path, allow_pickle=True)
        y = np.load(label_path) if label_path.suffix == ".npy" else pd.read_csv(label_path, header=None).to_numpy(dtype=int)
        print("EEG CHB-MIT shape:", X.shape, y.shape)
        return data_path, label_path, "EEG_CHBMIT"

    data_path = PROJECT_DIR / "data" / "processed" / "EEG" / "eeg_seizure_3D_normalized.npy"
    label_path = PROJECT_DIR / "data" / "processed" / "EEG" / "EEG_10x100x5_label.npy"
    X, y = prep.load_eeg_data(
        data_path,
        label_path,
        n_normal=n_normal,
        n_seizure=n_seizure,
        time_points=time_points,
        channels=channels,
    )
    print("EEG shape:", X.shape, y.shape)
    out_data = result_dir / f"EEG_{len(y)}x{time_points}x{channels}.npy"
    out_label = result_dir / f"EEG_{len(y)}x{time_points}x{channels}_label.npy"
    np.save(out_data, X)
    np.save(out_label, y)
    return out_data, out_label, f"EEG_{len(y)}x{time_points}x{channels}"


def prepare_ecg(overrides: dict[str, str], result_dir: Path) -> tuple[Path, Path, str]:
    prep = DataPreprocessing()
    conf = prep._load_yaml(PROJECT_DIR / "experiment_conf" / "problems" / "ecg.yaml")
    seed = as_int(overrides, "problem.seed", as_int(overrides, "seed", 30))
    X, y = prep.load_ecg_data(
        PROJECT_DIR / conf["data_root"] / conf["data_file"],
        int(conf["S_len"]),
        int(conf["I_len"]),
        int(conf["T_len"]),
        int(conf["F_len"]),
        seed=seed,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    out_data = result_dir / "ECG.npy"
    out_label = result_dir / "ECG_label.npy"
    np.save(out_data, X.reshape(2 * int(conf["I_len"]), int(conf["T_len"]), int(conf["F_len"])))
    np.save(out_label, y)
    print("ECG shape:", X.shape, y.shape)
    return out_data, out_label, "ecg"


def prepare_lds(overrides: dict[str, str], result_dir: Path) -> list[tuple[Path, Path, str]]:
    prep = DataPreprocessing()
    prep.prepare_lds_from_config(PROJECT_DIR, result_dir)
    label = result_dir / "lds_label.npy"
    datasets = []
    for n in expand_value(overrides.get("problem.lds_n", "2,3,4")):
        datasets.append((result_dir / f"lds{n}_sample0.npy", label, f"lds{n}"))
    return datasets


def run_solver(
    solver: str,
    data_file: Path,
    label_file: Path,
    name: str,
    result_dir: Path,
    seed: int,
    regularization: int,
    thresh: float,
    time_limit: int,
    gap: float,
    hidden_dim: int,
) -> tuple[float, float | str, int]:
    key = solver_key(solver)
    log_file = result_dir / f"{key}_{name}_seed_{seed}.log"
    cmd = [
        PYTHON_BIN,
        "-m",
        SOLVER_MODULE[key],
        "--data",
        str(data_file),
        "--label",
        str(label_file),
        "--name",
        name,
        "--seed",
        str(seed),
    ]
    if key in {"if_gurobi", "if", "em"}:
        cmd.extend(["--regularization", str(regularization)])
    if key == "if_gurobi":
        cmd.extend(["--thresh", str(thresh)])
        cmd.extend(["--time-limit", str(time_limit)])
        cmd.extend(["--gap", str(gap)])
        cmd.extend(["--hidden-dim", str(hidden_dim)])

    start = time.time()
    result = None
    import os

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            stripped = line.strip()
            if stripped.startswith("{") and "'f1'" in stripped:
                try:
                    result = ast.literal_eval(stripped)
                except Exception:
                    pass
        returncode = process.wait()
    duration = int(time.time() - start)
    if returncode != 0:
        raise RuntimeError(f"{key} failed for {name} seed={seed}; see {log_file}")

    if result is None:
        raise RuntimeError(f"No f1 result found in {log_file}")
    return float(result["f1"]), result.get("validation", ""), duration


def iter_chbmit_sampled_datasets(
    data_file: Path,
    label_file: Path,
    result_dir: Path,
    name: str,
    seed: int,
    sample_count: int = 11,
):
    X = np.load(data_file, allow_pickle=True)
    y = np.load(label_file) if label_file.suffix == ".npy" else pd.read_csv(label_file, header=None).to_numpy(dtype=int)
    if X.ndim != 4:
        yield data_file, label_file, name, ""
        return

    n_patients, n_samples, _, _ = X.shape
    if y.shape != (n_patients, n_samples):
        raise ValueError(f"Expected labels with shape {(n_patients, n_samples)}, got {y.shape}")

    if name in {"EEG_27s", "EEG_120s"}:
        sample_dir = result_dir / "sampled_inputs"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for patient_idx in range(n_patients):
            sample_name = f"{name}_patient{patient_idx:02d}"
            out_data = sample_dir / f"{sample_name}_seed_{seed}.npy"
            out_label = sample_dir / f"{sample_name}_seed_{seed}_label.npy"
            np.save(out_data, X[patient_idx, :, :, :])
            np.save(out_label, y[patient_idx, :])
            yield out_data, out_label, sample_name, patient_idx
        return

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_samples, size=min(sample_count, n_samples), replace=False)
    sample_dir = result_dir / "sampled_inputs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in sample_indices:
        sample_name = f"{name}_sample{int(sample_idx):02d}"
        out_data = sample_dir / f"{sample_name}_seed_{seed}.npy"
        out_label = sample_dir / f"{sample_name}_seed_{seed}_label.npy"
        np.save(out_data, X[:, int(sample_idx), :, :])
        np.save(out_label, y[:, int(sample_idx)])
        yield out_data, out_label, sample_name, int(sample_idx)


def write_summary(result_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "summary_table_long.csv", index=False)

    summaries = []
    for (problem_name, solver), group in df.groupby(["problem", "solver"]):
        f1 = group["f1"].to_numpy(dtype=float)
        duration = group["duration_seconds"].to_numpy(dtype=float)
        summaries.append(
            {
                "problem": problem_name,
                "solver": solver,
                "seed_times": len(group),
                "seeds": " ".join(str(int(x)) for x in group["seed"].to_list()),
                "f1_mean": float(np.mean(f1)),
                "f1_std": float(np.std(f1)),
                "f1_min": float(np.min(f1)),
                "f1_max": float(np.max(f1)),
                "duration_seconds_mean": float(np.mean(duration)),
                "duration_seconds_total": float(np.sum(duration)),
            }
        )

        prefix = result_dir / f"{solver}_{problem_name}"
        np.save(result_dir / f"seed_{solver}_{problem_name}.npy", group["seed"].to_numpy(dtype=int))
        np.save(result_dir / f"f1_{solver}_{problem_name}.npy", f1)
        np.save(result_dir / f"duration_{solver}_{problem_name}.npy", duration)
        validation = group["validation"]
        validation = validation[validation.astype(str) != ""]
        if not validation.empty:
            np.save(result_dir / f"validation_{solver}_{problem_name}.npy", validation.to_numpy(dtype=float))

    summary = pd.DataFrame(summaries)
    summary.to_csv(result_dir / "summary_table.csv", index=False)
    print(summary)
    print("summary:", result_dir / "summary_table.csv")


def run_plot(overrides: dict[str, str]) -> None:
    name = clean_value(overrides.get("problem", overrides.get("problem.name", "lds")))
    result_path = clean_value(overrides.get("result_path", f"./Result_{name}/"))
    methods = expand_value(overrides.get("solver", "if_Gurobi,if,em,fft,DTW"))
    cutdown = clean_value(overrides.get("plot.cutdown", "0")) in {"1", "true", "True"}
    result_dir = Path(result_path)
    if not result_dir.is_absolute():
        result_dir = PROJECT_DIR / result_dir

    summary_path = result_dir / "summary_table.csv"
    if summary_path.exists():
        import matplotlib.pyplot as plt

        summary = pd.read_csv(summary_path)
        if summary.empty:
            print("No rows in", summary_path)
            return

        labels = [f"{row.problem}:{row.solver}" for row in summary.itertuples()]
        x = np.arange(len(summary))
        y = summary["f1_mean"].to_numpy(dtype=float)
        yerr = summary["f1_std"].to_numpy(dtype=float) if "f1_std" in summary else None

        fig_width = max(8, 0.6 * len(labels))
        _, ax = plt.subplots(figsize=(fig_width, 5), dpi=200)
        ax.bar(x, y, yerr=yerr, capsize=4, color="#4292c6")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

        figure_dir = PROJECT_DIR / "reports" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        out = figure_dir / f"{name}_f1.png"
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        print("figure:", out)
        return

    Visualise().plot_MIF4cluster_methods(result_path, methods, name, cutdown=cutdown)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    overrides = parse_overrides(argv)
    if clean_value(overrides.get("plot", "false")) in {"1", "true", "True"}:
        run_plot(overrides)
        return

    problem = clean_value(overrides.get("problem", overrides.get("problem.name", "EEG")))
    result_dir = Path(clean_value(overrides.get("result_dir", f"Result_{problem}")))
    if not result_dir.is_absolute():
        result_dir = PROJECT_DIR / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    if problem.lower() == "eeg":
        datasets = [prepare_eeg(overrides, result_dir)]
    elif problem.lower() == "ecg":
        datasets = [prepare_ecg(overrides, result_dir)]
    elif problem.lower() == "lds":
        datasets = prepare_lds(overrides, result_dir)
    else:
        raise ValueError(f"Unsupported problem: {problem}")

    solvers = expand_value(overrides.get("solver", "if_gurobi"))
    seeds = [int(x) for x in expand_value(overrides.get("seed", overrides.get("problem.seed", "30")))]
    regularization = as_int(overrides, "solver.regularization", as_int(overrides, "regularization", 270))
    thresh = as_float(overrides, "solver.thresh", as_float(overrides, "thresh", 0.25))
    time_limit = as_int(overrides, "solver.time_limit", as_int(overrides, "time_limit", 1000))
    gap = as_float(
        overrides,
        "solver.gap",
        as_float(overrides, "solver.target_mip_gap", as_float(overrides, "gap", 0.01)),
    )
    hidden_dim = as_int(overrides, "solver.hidden_dim", as_int(overrides, "hidden_dim", 3))

    chbmit_sample_count = as_int(overrides, "problem.chbmit_sample_count", 11)
    rows = []
    for data_file, label_file, name in datasets:
        for solver, seed in itertools.product(solvers, seeds):
            for sampled_data, sampled_label, sampled_name, sample_idx in iter_chbmit_sampled_datasets(
                data_file,
                label_file,
                result_dir,
                name,
                seed,
                sample_count=chbmit_sample_count,
            ):
                f1, validation, duration = run_solver(
                    solver,
                    sampled_data,
                    sampled_label,
                    sampled_name,
                    result_dir,
                    seed,
                    regularization,
                    thresh,
                    time_limit,
                    gap,
                    hidden_dim,
                )
                key = solver_key(solver)
                rows.append(
                    {
                        "problem": sampled_name,
                        "solver": key,
                        "seed": seed,
                        "sample_index": sample_idx,
                        "f1": f1,
                        "validation": validation,
                        "duration_seconds": duration,
                    }
                )
                print(f"[problem={sampled_name}] [solver={key}] [seed={seed}] f1={f1} duration={duration}s")

    write_summary(result_dir, rows)
    print("Results are under", result_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
