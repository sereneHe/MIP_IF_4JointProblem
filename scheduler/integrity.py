"""Heartbeat and result-table integrity checking.

Determines whether a slot is "运行正常" by combining:

  - local process alive / server job running
  - log / MLflow / result file updated recently
  - no Gurobi / OOM / Python traceback / disk-full / timeout errors
  - result CSV readable with valid columns
  - completed row count matches or grows toward the shard's expected count

If no log/metric/result change for 10-15 minutes, the slot is marked
"疑似卡住" and a pending item is created.
"""

from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    HEALTH_ERROR,
    HEALTH_RUNNING,
    HEALTH_STALLED,
    HEALTH_STOPPED,
    HEARTBEAT_STALE_SECONDS,
    HEARTBEAT_SUSPECT_SECONDS,
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    STATUS_INVALID,
    STATUS_MISSING,
    STATUS_UNKNOWN,
)
from .manifest import ExperimentManifest, RunSpec
from .sharding import Shard

# Error patterns that indicate a broken run.
_ERROR_PATTERNS = [
    re.compile(r"GurobiError", re.I),
    re.compile(r"OutOfMemory|MemoryError|Killed", re.I),
    re.compile(r"Traceback \(most recent call last\)", re.I),
    re.compile(r"No space left on device", re.I),
    re.compile(r"Disk quota exceeded", re.I),
    re.compile(r"timed out|TimeoutError|time limit exceeded", re.I),
    re.compile(r"Segmentation fault", re.I),
]

# Required columns for a valid result CSV.
_REQUIRED_COLUMNS = ["method", "dataset", "seed"]


def check_log_for_errors(log_path: str) -> List[str]:
    """Scan a log file for known error patterns. Returns list of matches."""
    path = Path(log_path)
    if not path.exists():
        return []
    errors = []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    for pat in _ERROR_PATTERNS:
        if pat.search(text):
            errors.append(pat.pattern)
    return errors


def read_result_csv(csv_path: str) -> Tuple[bool, List[str], int]:
    """Read a result CSV. Returns (valid, missing_columns, row_count)."""
    path = Path(csv_path)
    if not path.exists():
        return False, _REQUIRED_COLUMNS, 0
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
            rows = list(reader)
    except Exception:
        return False, _REQUIRED_COLUMNS, 0
    return (not missing), missing, len(rows)


def check_manifest_integrity(
    manifest: ExperimentManifest,
    shards: List[Shard],
    result_csv: str = "",
) -> Dict:
    """Compare the manifest against the result CSV and shard statuses.

    Returns a dict with status and details.
    """
    expected = {r.run_id for r in manifest.runs}
    total_expected = len(expected)

    # Which run ids are currently running / queued across shards?
    active_ids: set = set()
    for shard in shards:
        if shard.status in ("running", "queued"):
            active_ids.update(shard.run_ids)

    # Which run ids are present in the result CSV?
    present_ids: set = set()
    valid, missing_cols, row_count = read_result_csv(result_csv)
    if valid and result_csv:
        try:
            with open(result_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = row.get("method", "")
                    dataset = row.get("dataset", "")
                    seed = row.get("seed", "")
                    if method and dataset and seed:
                        present_ids.add(f"{method}|{dataset}|default|{seed}")
        except Exception:
            pass

    missing_ids = expected - present_ids
    missing_not_active = missing_ids - active_ids

    if not valid:
        status = STATUS_INVALID
        detail = f"CSV 存在但缺列 {missing_cols} 或无法读取。"
    elif not missing_ids:
        status = STATUS_COMPLETE
        detail = f"所有 {total_expected} 个预期 run 均存在且字段有效。"
    elif missing_not_active:
        status = STATUS_MISSING
        detail = (
            f"缺少 {len(missing_not_active)} 个 run 且不在任何队列中："
            f"{sorted(missing_not_active)[:5]}..."
        )
    else:
        status = STATUS_IN_PROGRESS
        detail = (
            f"已有 {len(present_ids)}/{total_expected} 行；"
            f"缺少的 {len(missing_ids)} 个 run 正在运行/排队。"
        )

    return {
        "status": status,
        "detail": detail,
        "expected_total": total_expected,
        "present": len(present_ids),
        "missing": len(missing_ids),
        "missing_not_active": len(missing_not_active),
        "row_count": row_count,
        "csv_valid": valid,
        "missing_columns": missing_cols,
    }


def evaluate_slot_health(
    shard: Shard,
    is_alive: bool,
    heartbeat_age: float,
    log_path: str,
    result_csv: str,
    manifest: ExperimentManifest,
    shards: List[Shard],
) -> Dict:
    """Combine all signals into a slot health verdict."""
    errors = check_log_for_errors(log_path)

    if shard.status in ("stopped", "done", "error"):
        return {
            "health": HEALTH_STOPPED if shard.status == "stopped" else HEALTH_ERROR,
            "errors": errors,
            "heartbeat_age": heartbeat_age,
            "integrity": check_manifest_integrity(manifest, shards, result_csv),
        }

    if not is_alive:
        return {
            "health": HEALTH_ERROR,
            "errors": errors or ["进程/作业不存在"],
            "heartbeat_age": heartbeat_age,
            "integrity": check_manifest_integrity(manifest, shards, result_csv),
        }

    if errors:
        return {
            "health": HEALTH_ERROR,
            "errors": errors,
            "heartbeat_age": heartbeat_age,
            "integrity": check_manifest_integrity(manifest, shards, result_csv),
        }

    if heartbeat_age > HEARTBEAT_SUSPECT_SECONDS:
        return {
            "health": HEALTH_STALLED,
            "errors": [],
            "heartbeat_age": heartbeat_age,
            "integrity": check_manifest_integrity(manifest, shards, result_csv),
        }

    return {
        "health": HEALTH_RUNNING,
        "errors": [],
        "heartbeat_age": heartbeat_age,
        "integrity": check_manifest_integrity(manifest, shards, result_csv),
    }
