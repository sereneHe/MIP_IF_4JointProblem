"""Configuration for the Experiment Scheduling Center.

The resource rules are fixed as:

    local slot 1:  max 24 hours
    local slot 2:  max 24 hours
    server slot 1: max 24 hours
    server slot 2: max 24 hours
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Fixed resource rules
# ---------------------------------------------------------------------------

SLOT_NAMES = ["local-1", "local-2", "server-1", "server-2"]

MAX_SLOT_HOURS = 24.0  # hard cap per slot / per submitted job
MAX_SLOT_SECONDS = int(MAX_SLOT_HOURS * 3600)

# Safety margin added to the estimated walltime when submitting to PBS.
WALLTIME_SAFETY_FACTOR = 1.2
WALLTIME_SAFETY_MINUTES = 30

# Heartbeat / liveness thresholds
HEARTBEAT_STALE_SECONDS = 600      # 10 min without any file/log change
HEARTBEAT_SUSPECT_SECONDS = 900    # 15 min -> mark "疑似卡住"
REFRESH_INTERVAL_SECONDS = 10      # UI refresh cadence
DAILY_REPORT_HOUR = 23             # hour of day for the daily summary report

# Result table integrity statuses
STATUS_COMPLETE = "完整"
STATUS_IN_PROGRESS = "进行中"
STATUS_MISSING = "异常缺失"
STATUS_INVALID = "无效结果"
STATUS_UNKNOWN = "未知"

# Slot health statuses
HEALTH_RUNNING = "运行正常"
HEALTH_STALLED = "疑似卡住"
HEALTH_STOPPED = "已停止"
HEALTH_QUEUED = "排队中"
HEALTH_ERROR = "错误"


@dataclass
class SlotConfig:
    """Configuration for a single worker slot."""

    name: str
    kind: str  # "local" or "server"
    max_hours: float = MAX_SLOT_HOURS
    enabled: bool = True

    # Local-only
    cwd: str = ""
    python: str = ""

    # Server-only (PBS)
    host: str = ""
    user: str = ""
    project_dir: str = ""
    queue: str = ""
    ncpus: int = 2
    mem_gb: int = 32
    scratch_gb: int = 30
    license_file: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "max_hours": self.max_hours,
            "enabled": self.enabled,
            "cwd": self.cwd,
            "python": self.python,
            "host": self.host,
            "user": self.user,
            "project_dir": self.project_dir,
            "queue": self.queue,
            "ncpus": self.ncpus,
            "mem_gb": self.mem_gb,
            "scratch_gb": self.scratch_gb,
            "license_file": self.license_file,
        }


@dataclass
class SchedulerConfig:
    """Top-level scheduler configuration."""

    slots: List[SlotConfig] = field(default_factory=list)
    data_dir: str = ""
    reports_dir: str = ""
    logs_dir: str = ""
    mlflow_uri: str = ""
    mlflow_experiment: str = ""

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        base = Path(os.environ.get("SCHEDULER_BASE", str(Path.cwd())))
        data_dir = os.environ.get("SCHEDULER_DATA_DIR", str(base / "scheduler_data"))
        reports_dir = os.environ.get("SCHEDULER_REPORTS_DIR", str(base / "scheduler_reports"))
        logs_dir = os.environ.get("SCHEDULER_LOGS_DIR", str(base / "scheduler_logs"))

        slots = [
            SlotConfig(
                name="local-1",
                kind="local",
                cwd=os.environ.get("SCHEDULER_LOCAL1_CWD", str(base)),
                python=os.environ.get("SCHEDULER_LOCAL1_PYTHON", "python"),
            ),
            SlotConfig(
                name="local-2",
                kind="local",
                cwd=os.environ.get("SCHEDULER_LOCAL2_CWD", str(base)),
                python=os.environ.get("SCHEDULER_LOCAL2_PYTHON", "python"),
            ),
            SlotConfig(
                name="server-1",
                kind="server",
                host=os.environ.get("SCHEDULER_SERVER1_HOST", ""),
                user=os.environ.get("SCHEDULER_SERVER1_USER", ""),
                project_dir=os.environ.get(
                    "SCHEDULER_SERVER1_PROJECT",
                    "/storage/praha1/home/hexiaoyu/project-bestdagsolverintheworld",
                ),
                queue=os.environ.get("SCHEDULER_SERVER1_QUEUE", ""),
                license_file=os.environ.get(
                    "SCHEDULER_SERVER1_LICENSE",
                    "/storage/praha1/home/hexiaoyu/gurobi.lic",
                ),
            ),
            SlotConfig(
                name="server-2",
                kind="server",
                host=os.environ.get("SCHEDULER_SERVER2_HOST", ""),
                user=os.environ.get("SCHEDULER_SERVER2_USER", ""),
                project_dir=os.environ.get(
                    "SCHEDULER_SERVER2_PROJECT",
                    "/storage/praha1/home/hexiaoyu/project-bestdagsolverintheworld",
                ),
                queue=os.environ.get("SCHEDULER_SERVER2_QUEUE", ""),
                license_file=os.environ.get(
                    "SCHEDULER_SERVER2_LICENSE",
                    "/storage/praha1/home/hexiaoyu/gurobi.lic",
                ),
            ),
        ]

        return cls(
            slots=slots,
            data_dir=data_dir,
            reports_dir=reports_dir,
            logs_dir=logs_dir,
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
            mlflow_experiment=os.environ.get("MLFLOW_EXPERIMENT_NAME", "default"),
        )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.reports_dir, self.logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
