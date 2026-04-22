"""Build CHB-MIT scalp EEG tensors for mixture LDS experiments.

This script downloads the CHB-MIT EDF files for patients chb01, chb03, and
chb05, extracts centered seizure windows plus matched normal windows, and
exports final experiment-ready tensors:

    eeg-27s.npy   -> {"x": (3, 10, 6912, 23),  "y": (3, 10)}
    eeg-120s.npy  -> {"x": (3, 10, 30720, 23), "y": (3, 10)}

The label-details CSV files keep the sample_id-level abnormal/normal mapping.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from mixture_lds.data.preprocessing import DataPreprocessing


DEFAULT_SUBJECTS = (1, 3, 5)
DEFAULT_SAMPLING_RATE = 256
DEFAULT_NORMAL_MARGIN_SEC = 300


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_patients(patients) -> tuple[int, ...]:
    """Normalize patient identifiers like 1, "1", or "chb01" to integers."""

    normalized = []
    for patient in patients:
        value = str(patient).strip().lower()
        if value.startswith("chb"):
            value = value[3:]
        normalized.append(int(value))
    return tuple(normalized)


def download_chbmit_edf(raw_eeg_dir: Path, subjects=DEFAULT_SUBJECTS) -> Path:
    """Download CHB-MIT EDF files using the same wget logic as the notebook."""

    raw_eeg_dir.mkdir(parents=True, exist_ok=True)
    dirs = [f"chb{subject:02d}" for subject in subjects]

    for directory in dirs:
        url = f"https://archive.physionet.org/pn6/chbmit/{directory}"
        subprocess.run(
            ["wget", "-r", "-A", "edf", "-np", url],
            cwd=raw_eeg_dir,
            check=True,
        )

    downloaded_root = raw_eeg_dir / "archive.physionet.org" / "pn6" / "chbmit"
    final_root = raw_eeg_dir / "chbmit"
    if downloaded_root.exists() and downloaded_root != final_root:
        if final_root.exists():
            shutil.rmtree(final_root)
        shutil.move(str(downloaded_root), str(final_root))

    archive_root = raw_eeg_dir / "archive.physionet.org"
    if archive_root.exists():
        shutil.rmtree(archive_root)

    return final_root


def _copy_or_create_label_details(table_dir: Path, duration: int, subjects=DEFAULT_SUBJECTS):
    frames = []
    for subject in subjects:
        subject_code = f"chb{subject:02d}"
        label_file = table_dir / f"{subject_code}_centered_{duration}s_labels.csv"
        detail_file = table_dir / f"{subject_code}_centered_{duration}s_label-details.csv"
        shutil.copy2(label_file, detail_file)

        details = pd.read_csv(detail_file)
        details.insert(0, "patient", subject_code)
        frames.append(details)

    combined_details = pd.concat(frames, ignore_index=True)
    detail_out = table_dir / f"eeg-{duration}s-label-details.csv"
    combined_details.to_csv(detail_out, index=False)
    return detail_out


def _combine_centered_tables(table_dir: Path, duration: int, subjects=DEFAULT_SUBJECTS):
    seizure_arrays = []
    normal_arrays = []
    for subject in subjects:
        subject_code = f"chb{subject:02d}"
        seizure_arrays.append(
            np.load(table_dir / f"{subject_code}_centered_{duration}s_seizures.npy")
        )
        normal_arrays.append(
            np.load(table_dir / f"{subject_code}_centered_{duration}s_normal.npy")
        )

    seizures = np.stack(seizure_arrays, axis=0)
    normal = np.stack(normal_arrays, axis=0)
    x = np.concatenate([seizures, normal], axis=1)
    y = np.concatenate(
        [
            np.ones(seizures.shape[:2], dtype=int),
            np.zeros(normal.shape[:2], dtype=int),
        ],
        axis=1,
    )
    return x, y, seizures, normal


def minimum_seizure_count(raw_root: Path, subjects=DEFAULT_SUBJECTS) -> int:
    """Return the smallest seizure-event count among the selected patients."""

    events = DataPreprocessing().parse_chbmit_summary_events(raw_root, subjects=subjects)
    counts = {f"chb{subject:02d}": 0 for subject in subjects}
    for event in events:
        counts[event["subject"]] += 1

    if not counts:
        raise ValueError("No patients were provided.")
    missing = [subject for subject, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"No seizure events found for: {', '.join(missing)}")
    return min(counts.values())


def min_max_seizure_durations(raw_root: Path, subjects=DEFAULT_SUBJECTS) -> tuple[int, int]:
    """Return shortest and longest seizure durations among selected patients."""

    events = DataPreprocessing().parse_chbmit_summary_events(raw_root, subjects=subjects)
    durations = [event["duration_sec"] for event in events]
    if not durations:
        raise ValueError("No seizure durations found for selected patients.")
    return min(durations), max(durations)


def build_chbmit_eeg(
    raw_root: Path,
    output_dir: Path,
    subjects=DEFAULT_SUBJECTS,
    durations: tuple[int, ...] | None = None,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    max_events_per_subject: int | None = None,
    normal_margin_sec: int = DEFAULT_NORMAL_MARGIN_SEC,
):
    """Build per-duration CHB-MIT EEG tensors and label-details tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = DataPreprocessing()
    exported = {}
    event_limit = (
        minimum_seizure_count(raw_root, subjects=subjects)
        if max_events_per_subject is None
        else max_events_per_subject
    )
    if durations is None:
        durations = min_max_seizure_durations(raw_root, subjects=subjects)

    for duration in durations:
        table_dir = output_dir / f"centered_{duration}s_seizure_tables"
        table_dir.mkdir(parents=True, exist_ok=True)

        for old_file in table_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

        processor.export_chbmit_centered_seizure_event_tables(
            raw_root=raw_root,
            output_dir=table_dir,
            subjects=subjects,
            window_duration_sec=duration,
            sampling_rate=sampling_rate,
            max_events_per_subject=event_limit,
            include_normal=True,
            normal_margin_sec=normal_margin_sec,
        )

        x, y, seizures, normal = _combine_centered_tables(
            table_dir=table_dir,
            duration=duration,
            subjects=subjects,
        )
        detail_file = _copy_or_create_label_details(
            table_dir=table_dir,
            duration=duration,
            subjects=subjects,
        )

        data_file = output_dir / f"eeg-{duration}s.npy"
        np.save(
            data_file,
            {
                "x": x,
                "y": y,
                "subjects": np.array([f"chb{subject:02d}" for subject in subjects]),
                "duration_sec": duration,
                "sampling_rate": sampling_rate,
            },
            allow_pickle=True,
        )

        exported[duration] = {
            "data_file": data_file,
            "label_details_file": detail_file,
            "x_shape": x.shape,
            "y_shape": y.shape,
            "seizure_shape": seizures.shape,
            "normal_shape": normal.shape,
            "events_per_subject": event_limit,
            "normal_margin_sec": normal_margin_sec,
        }

    return exported


def parse_args():
    project = _project_root()
    parser = argparse.ArgumentParser(
        description="Download and build CHB-MIT EEG tensors for mixture LDS."
    )
    parser.add_argument(
        "--raw-eeg-dir",
        type=Path,
        default=project / "data" / "raw" / "EEG",
        help="Directory where archive.physionet.org/ or chbmit/ is downloaded.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Existing CHB-MIT root containing chb01/chb03/chb05. "
        "Defaults to raw-eeg-dir/chbmit, then raw-eeg-dir/archive.physionet.org/pn6/chbmit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "data" / "processed" / "EEG",
        help="Output directory for eeg-27s.npy, eeg-120s.npy, and tables.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing raw EDF files and do not run wget.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="Patient numbers to use, e.g. 1 3 5. Kept for compatibility; --patients is preferred.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Patients to use, e.g. chb01 chb03 chb05 or 1 3 5.",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Centered window durations in seconds. "
            "Default: shortest and longest seizure durations from selected patient summaries."
        ),
    )
    parser.add_argument("--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE)
    parser.add_argument(
        "--normal-margin-sec",
        type=int,
        default=DEFAULT_NORMAL_MARGIN_SEC,
        help="Minimum gap between seizure and same-file normal window when possible.",
    )
    parser.add_argument(
        "--max-events-per-subject",
        type=int,
        default=None,
        help=(
            "Keep this many seizure events per patient. "
            "Default: auto-balance to the smallest seizure count among selected patients."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patients = normalize_patients(args.patients if args.patients is not None else args.subjects)

    if args.skip_download:
        raw_root = args.raw_root
        if raw_root is None:
            candidates = [
                args.raw_eeg_dir / "chbmit",
                args.raw_eeg_dir / "archive.physionet.org" / "pn6" / "chbmit",
            ]
            raw_root = next((candidate for candidate in candidates if candidate.exists()), None)
            if raw_root is None:
                raise FileNotFoundError(
                    "Cannot find existing CHB-MIT raw root. Pass --raw-root or run without --skip-download."
                )
    else:
        raw_root = download_chbmit_edf(args.raw_eeg_dir, subjects=patients)

    exported = build_chbmit_eeg(
        raw_root=Path(raw_root),
        output_dir=args.output_dir,
        subjects=patients,
        durations=tuple(args.durations) if args.durations is not None else None,
        sampling_rate=args.sampling_rate,
        max_events_per_subject=args.max_events_per_subject,
        normal_margin_sec=args.normal_margin_sec,
    )

    for duration, info in exported.items():
        print(f"eeg-{duration}s.npy:", info["data_file"])
        print("  x:", info["x_shape"])
        print("  y:", info["y_shape"])
        print("  seizures:", info["seizure_shape"])
        print("  normal:", info["normal_shape"])
        print("  events-per-subject:", info["events_per_subject"])
        print("  normal-margin-sec:", info["normal_margin_sec"])
        print("  label-details:", info["label_details_file"])


if __name__ == "__main__":
    main()
