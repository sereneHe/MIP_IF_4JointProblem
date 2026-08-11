"""Local experiment report generation.

Immediately after a task starts, the backend generates a local report (no GPT
involved) containing:

  - plan name, script, parameters and this batch's scope
  - the four slots' assignments and estimated end times
  - each task's result table, log, MLflow address
  - expected total runs and current completed count
  - detected risks (license, server mount, time limit)
  - next automatic check time

The UI refreshes every 10 s; a daily summary report is saved automatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import REFRESH_INTERVAL_SECONDS
from .estimator import format_duration
from .manifest import ExperimentManifest
from .sharding import Shard, ShardPlan


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def estimate_end_time(shard: Shard, started_at: Optional[str] = None) -> str:
    """Estimate the wall-clock end time for a shard."""
    start = started_at or shard.started_at
    if not start:
        return ""
    try:
        start_dt = datetime.fromisoformat(start)
    except ValueError:
        return ""
    end = start_dt + timedelta(seconds=shard.estimated_seconds)
    return _iso(end)


def build_task_report(
    manifest: ExperimentManifest,
    plan: ShardPlan,
    shards: List[Shard],
    runs_by_id: Dict,
    result_csv: str = "",
    mlflow_uri: str = "",
    risks: Optional[List[str]] = None,
) -> Dict:
    """Build the full report dict shown in the UI."""
    risks = risks or []
    total_runs = len(manifest.runs)
    done_runs = 0
    for shard in shards:
        if shard.status in ("done", "stopped"):
            done_runs += len(shard.run_ids)

    slot_rows = []
    for shard in shards:
        slot_rows.append(
            {
                "slot": shard.slot,
                "status": shard.status,
                "run_count": len(shard.run_ids),
                "estimated_seconds": shard.estimated_seconds,
                "estimated_duration": format_duration(shard.estimated_seconds),
                "estimated_end": estimate_end_time(shard),
                "job_id": shard.job_id,
                "pid": shard.pid,
                "log_path": shard.log_path,
                "result_csv": shard.result_csv or result_csv,
                "mlflow_experiment": shard.mlflow_experiment or manifest.plan_name,
                "walltime": shard.walltime,
            }
        )

    return {
        "plan_id": manifest.plan_id,
        "plan_name": manifest.plan_name,
        "script": manifest.script,
        "created_at": manifest.created_at,
        "generated_at": _iso(_now()),
        "total_runs": total_runs,
        "done_runs": done_runs,
        "result_csv": result_csv,
        "mlflow_uri": mlflow_uri,
        "slots": slot_rows,
        "risks": risks,
        "next_check_in_seconds": REFRESH_INTERVAL_SECONDS,
        "notes": plan.notes,
    }


def save_report(report: Dict, reports_dir: str, plan_id: str, suffix: str = "") -> Path:
    """Persist a report JSON to the reports dir."""
    path = Path(reports_dir) / f"{plan_id}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def save_daily_report(report: Dict, reports_dir: str, plan_id: str) -> Path:
    """Save a daily summary report (one per day)."""
    day = _now().strftime("%Y%m%d")
    return save_report(report, reports_dir, plan_id, suffix=f"_daily_{day}")


def detect_risks(
    manifest: ExperimentManifest,
    plan: ShardPlan,
    server_configured: bool,
    license_file: str = "",
) -> List[str]:
    """Detect known risks before/at task start."""
    risks: List[str] = []

    # License risk
    if any("gurobi" in r.method.lower() or "if_gurobi" in r.method.lower()
           for r in manifest.runs):
        if not license_file:
            risks.append("检测到 Gurobi 求解器，但未配置许可证文件路径。")
        else:
            risks.append(f"Gurobi 许可证：{license_file}")

    # Server mount risk
    if not server_configured:
        risks.append("服务器槽位未配置 host/user，服务器任务将不会自动提交。")

    # Time-limit risk
    for shard in plan.shards:
        if shard.estimated_seconds > 24 * 3600:
            risks.append(
                f"槽位 {shard.slot} 预计 {format_duration(shard.estimated_seconds)} "
                f"超过 24 小时上限，需要拆分或 checkpoint。"
            )

    return risks
