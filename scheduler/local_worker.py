"""Local worker slot.

Holds a Python subprocess PID for a local slot.  Provides start / stop /
poll / heartbeat operations.  The subprocess runs a generated shard script
that executes the batch of runs sequentially.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import SlotConfig
from .manifest import RunSpec
from .sharding import Shard


def generate_shard_script(
    shard: Shard,
    runs_by_id: Dict[str, RunSpec],
    log_path: str,
    python: str = "python",
) -> str:
    """Generate a temporary .sh that runs the shard's runs sequentially.

    Each run is executed with its own env (DASHBOARD_RUN_ID /
    MLFLOW_EXPERIMENT_NAME) so MLflow results flow back to the correct slot.
    """
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    lines.append(f"echo '=== Shard {shard.slot} started at $(date) ==='")
    lines.append(f"echo 'Runs: {len(shard.run_ids)}'")
    lines.append("")

    for rid in shard.run_ids:
        run = runs_by_id.get(rid)
        if run is None:
            continue
        lines.append(f"echo '--- run {rid} at $(date) ---'")
        env_prefix = ""
        if run.env:
            env_prefix = " ".join(f"{k}={v}" for k, v in run.env.items()) + " "
        if run.command:
            lines.append(f"{env_prefix}{run.command}")
        else:
            lines.append(f"echo 'no command for {rid}'")
        lines.append("")

    lines.append(f"echo '=== Shard {shard.slot} finished at $(date) ==='")
    return "\n".join(lines)


class LocalWorker:
    """Manages a single local slot's subprocess."""

    def __init__(self, config: SlotConfig, logs_dir: str):
        self.config = config
        self.logs_dir = Path(logs_dir)
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: str = ""
        self.last_heartbeat: float = 0.0
        self.last_log_mtime: float = 0.0

    # ------------------------------------------------------------------
    def start(self, shard: Shard, runs_by_id: Dict[str, RunSpec]) -> bool:
        """Start the shard's script as a subprocess. Returns True on success."""
        if self.proc is not None and self.proc.poll() is None:
            return False  # already running

        self.log_path = str(self.logs_dir / f"{shard.slot}_{shard.shard_index}.log")
        script = generate_shard_script(
            shard, runs_by_id, self.log_path, python=self.config.python
        )
        script_path = self.logs_dir / f"{shard.slot}_{shard.shard_index}.sh"
        script_path.write_text(script)

        log_file = open(self.log_path, "a")
        try:
            self.proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=self.config.cwd or ".",
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            self.proc = None
            return False

        shard.pid = self.proc.pid
        shard.status = "running"
        shard.log_path = self.log_path
        self.last_heartbeat = time.time()
        self.last_log_mtime = self._log_mtime()
        return True

    # ------------------------------------------------------------------
    def stop(self, shard: Shard) -> None:
        """Stop the subprocess (SIGTERM then SIGKILL)."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        shard.status = "stopped"
        shard.finished_at = _now_iso()
        self.proc = None

    # ------------------------------------------------------------------
    def poll(self, shard: Shard) -> str:
        """Return the current status string for the shard."""
        if self.proc is None:
            return shard.status
        rc = self.proc.poll()
        if rc is None:
            shard.status = "running"
            self.last_heartbeat = time.time()
            return "running"
        # process exited
        shard.finished_at = _now_iso()
        self.proc = None
        shard.status = "done" if rc == 0 else "error"
        return shard.status

    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _log_mtime(self) -> float:
        try:
            return Path(self.log_path).stat().st_mtime
        except OSError:
            return 0.0

    def heartbeat_age(self) -> float:
        """Seconds since the log file was last modified."""
        mtime = self._log_mtime()
        if mtime > 0:
            self.last_log_mtime = mtime
        return time.time() - (self.last_log_mtime or self.last_heartbeat)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
