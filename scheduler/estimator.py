"""Time estimation for experiment runs.

Estimates wall-clock time per run using, in priority order:

  1. Historical run records (from MLflow / result CSVs) for the same
     method x dataset x param-group.
  2. A conservative heuristic based on the .sh parameters, dataset size and
     any time limit in the script.

The estimate is used to (a) split runs into shards and (b) set the PBS
walltime (estimate + safety margin, capped at 24 h).
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import MAX_SLOT_SECONDS, WALLTIME_SAFETY_FACTOR, WALLTIME_SAFETY_MINUTES
from .manifest import RunSpec

# ---------------------------------------------------------------------------
# Historical record store
# ---------------------------------------------------------------------------


class HistoryStore:
    """Loads historical run durations from CSV/MLflow summary files.

    The store is a simple JSON map keyed by (method, dataset, param_group)
    -> list of observed durations in seconds.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._cache: Optional[Dict[str, List[float]]] = None

    def _load(self) -> Dict[str, List[float]]:
        if self._cache is not None:
            return self._cache
        store = {}
        path = self.data_dir / "history.json"
        if path.exists():
            try:
                store = json.loads(path.read_text())
            except Exception:
                store = {}
        self._cache = store
        return store

    def _save(self, store: Dict[str, List[float]]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "history.json").write_text(
            json.dumps(store, indent=2, ensure_ascii=False)
        )
        self._cache = store

    def record(self, method: str, dataset: str, param_group: str, seconds: float) -> None:
        store = self._load()
        key = self._key(method, dataset, param_group)
        store.setdefault(key, []).append(seconds)
        # keep only the most recent 50 observations
        store[key] = store[key][-50:]
        self._save(store)

    @staticmethod
    def _key(method: str, dataset: str, param_group: str) -> str:
        return f"{method}|{dataset}|{param_group}"

    def median_seconds(self, method: str, dataset: str, param_group: str) -> Optional[float]:
        store = self._load()
        vals = store.get(self._key(method, dataset, param_group))
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2.0

    def ingest_csv(self, csv_path: str, method_col: str = "method",
                   dataset_col: str = "dataset", group_col: str = "param_group",
                   time_col: str = "runtime_seconds") -> int:
        """Bulk-load historical durations from a results CSV."""
        path = Path(csv_path)
        if not path.exists():
            return 0
        count = 0
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.get(method_col, "unknown")
                dataset = row.get(dataset_col, "unknown")
                group = row.get(group_col, "default")
                try:
                    seconds = float(row[time_col])
                except (KeyError, ValueError):
                    continue
                self.record(method, dataset, group, seconds)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Heuristic estimator
# ---------------------------------------------------------------------------

# Rough per-dataset baseline (seconds) when no history exists.
_DATASET_BASELINE = {
    "ecg": 120.0,
    "eeg": 600.0,
    "lds": 300.0,
    "synthetic": 60.0,
    "unknown": 300.0,
}

# Rough per-method multiplier.
_METHOD_MULTIPLIER = {
    "dtw": 1.0,
    "em": 1.5,
    "fft": 0.8,
    "if": 2.0,
    "if_gurobi": 3.0,
    "unknown": 1.0,
}


def _dataset_baseline(dataset: str) -> float:
    d = dataset.lower()
    for key, val in _DATASET_BASELINE.items():
        if key in d:
            return val
    return _DATASET_BASELINE["unknown"]


def _method_multiplier(method: str) -> float:
    m = method.lower()
    for key, val in _METHOD_MULTIPLIER.items():
        if key in m:
            return val
    return _METHOD_MULTIPLIER["unknown"]


def _scale_from_params(command: str) -> float:
    """Scale the baseline by dataset size / sample count found in the command."""
    scale = 1.0
    m = re.search(r"(?:problem\.)?number_of_samples=(\d+)", command)
    if m:
        n = int(m.group(1))
        scale *= max(1.0, n / 1000.0)
    m = re.search(r"(?:problem\.)?number_of_variables=(\d+)", command)
    if m:
        d = int(m.group(1))
        scale *= max(1.0, d / 10.0)
    return scale


def estimate_run_seconds(run: RunSpec, history: Optional[HistoryStore] = None) -> float:
    """Estimate wall-clock seconds for a single run.

    Priority: history median > heuristic.
    """
    if history is not None:
        med = history.median_seconds(run.method, run.dataset, run.param_group)
        if med is not None and med > 0:
            return med

    baseline = _dataset_baseline(run.dataset)
    mult = _method_multiplier(run.method)
    scale = _scale_from_params(run.command)
    return baseline * mult * scale


def estimate_manifest_runs(
    runs: List[RunSpec],
    history: Optional[HistoryStore] = None,
) -> List[RunSpec]:
    """Fill in estimated_seconds for each run in place and return them."""
    for run in runs:
        run.estimated_seconds = estimate_run_seconds(run, history)
    return runs


def pbs_walltime(estimated_seconds: float) -> str:
    """Convert an estimate into a PBS walltime string (HH:MM:SS).

    Adds a safety margin and caps at 24:00:00.
    """
    margin = estimated_seconds * (WALLTIME_SAFETY_FACTOR - 1.0)
    margin += WALLTIME_SAFETY_MINUTES * 60
    total = int(estimated_seconds + margin)
    total = min(total, MAX_SLOT_SECONDS)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_duration(seconds: float) -> str:
    """Human-readable duration like '3 小时 12 分'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec} 秒"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} 小时 {minutes} 分"
    days, hours = divmod(hours, 24)
    return f"{days} 天 {hours} 小时 {minutes} 分"
