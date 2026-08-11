"""PBS (MetaCentrum) server worker slot.

Submits a generated PBS script via `qsub` and polls the job status via
`qstat`.  The worker never auto-submits unless the server slot is configured
(host/user set) and the user has explicitly approved submission.

The generated PBS script mirrors the existing MetaCentrum pattern:
  - rsync project to scratch
  - run the shard script
  - background sync of outputs back to the project dir
  - walltime set to estimate + safety margin, capped at 24 h
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import SlotConfig
from .estimator import pbs_walltime
from .manifest import RunSpec
from .sharding import Shard


def generate_pbs_script(
    shard: Shard,
    runs_by_id: Dict[str, RunSpec],
    config: SlotConfig,
    project_dir: str,
    walltime: str,
    job_name: str,
) -> str:
    """Generate a PBS submission script for a shard."""
    run_ids = " ".join(shard.run_ids)
    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -l walltime={walltime}",
        f"#PBS -l select=1:ncpus={config.ncpus}:mem={config.mem_gb}gb:scratch_local={config.scratch_gb}gb",
        "#PBS -j oe",
        "",
        "set -euo pipefail",
        "",
        f"PROJECT={project_dir}",
        "WORK=$SCRATCHDIR/project-bestdagsolverintheworld",
        f"OUT=$PROJECT/metacentrum_runs/$PBS_JOBID",
        "",
        "echo \"Job $PBS_JOBID started at $(date)\"",
        "echo \"Node: $(hostname -f)\"",
        "echo \"Scratch: $SCRATCHDIR\"",
        "",
        "mkdir -p \"$OUT\"",
        "",
        "SYNC_PID=\"\"",
        "sync_outputs() {",
        "  mkdir -p \"$OUT/dagsolvers\"",
        "  rsync -a \"$WORK/dagsolvers/multirun\" \"$WORK/dagsolvers/mlruns\" \"$WORK/dagsolvers/results\" \"$WORK/dagsolvers/outputs\" \"$OUT/dagsolvers/\" 2>/dev/null || true",
        "}",
        "",
        "finish() {",
        "  status=$?",
        "  if [ -n \"$SYNC_PID\" ]; then",
        "    kill \"$SYNC_PID\" 2>/dev/null || true",
        "    wait \"$SYNC_PID\" 2>/dev/null || true",
        "  fi",
        "  sync_outputs",
        "  clean_scratch || true",
        "  exit \"$status\"",
        "}",
        "trap finish EXIT",
        "",
        "module add python",
        "source \"$PROJECT/.venv/bin/activate\"",
        "",
        "rsync -a \\",
        "  --exclude .git \\",
        "  --exclude .venv \\",
        "  --exclude metacentrum_runs \\",
        "  --exclude dagsolvers/multirun \\",
        "  --exclude dagsolvers/mlruns \\",
        "  --exclude dagsolvers/results \\",
        "  --exclude dagsolvers/outputs \\",
        "  \"$PROJECT/\" \"$WORK/\"",
        "",
        "cd \"$WORK/dagsolvers\"",
        "export PYTHONPATH=\"$WORK:$WORK/dagsolvers:${PYTHONPATH:-}\"",
        "export PYTHONNOUSERSITE=1",
        "export HYDRA_FULL_ERROR=1",
        "export MLFLOW_ALLOW_FILE_STORE=true",
        f"export GRB_LICENSE_FILE={config.license_file or '/storage/praha1/home/hexiaoyu/gurobi.lic'}",
        "export KMP_DUPLICATE_LIB_OK=TRUE",
        f"export OMP_NUM_THREADS={config.ncpus}",
        f"export MKL_NUM_THREADS={config.ncpus}",
        f"export OPENBLAS_NUM_THREADS={config.ncpus}",
        f"export NUMEXPR_NUM_THREADS={config.ncpus}",
        "",
        "while true; do",
        "  sync_outputs",
        "  sleep 300",
        "done &",
        "SYNC_PID=$!",
        "",
        f"echo '=== Shard {shard.slot} runs: {run_ids} ==='",
        "",
    ]

    # Emit each run command with its env.
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


class PBSWorker:
    """Manages a single server slot's PBS job."""

    def __init__(self, config: SlotConfig, logs_dir: str):
        self.config = config
        self.logs_dir = Path(logs_dir)
        self.job_id: str = ""
        self.last_status: str = ""
        self.last_poll: float = 0.0

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        """A server slot is usable only if host+user are configured."""
        return bool(self.config.host and self.config.user)

    # ------------------------------------------------------------------
    def submit(self, shard: Shard, runs_by_id: Dict[str, RunSpec],
               project_dir: str, walltime: str) -> bool:
        """Submit the shard to PBS. Returns True on success."""
        if not self.configured:
            return False

        job_name = f"shard_{shard.slot.replace('-', '')}"
        script = generate_pbs_script(
            shard, runs_by_id, self.config, project_dir, walltime, job_name
        )
        script_path = self.logs_dir / f"{shard.slot}_{shard.shard_index}.pbs"
        script_path.write_text(script)

        # Copy the script to the server and submit.
        remote_script = f"~/{script_path.name}"
        try:
            self._scp(str(script_path), remote_script)
            out = self._ssh(f"qsub {remote_script}")
        except Exception:
            return False

        match = re.search(r"(\d+\.\S+)", out)
        if not match:
            return False
        self.job_id = match.group(1)
        shard.job_id = self.job_id
        shard.status = "queued"
        shard.walltime = walltime
        shard.log_path = f"{project_dir}/metacentrum_runs/{self.job_id}"
        self.last_status = "queued"
        self.last_poll = time.time()
        return True

    # ------------------------------------------------------------------
    def poll(self, shard: Shard) -> str:
        """Poll qstat for the job status. Returns 'running'/'queued'/'done'/'error'."""
        if not self.job_id:
            return shard.status
        try:
            out = self._ssh(f"qstat -f {self.job_id}")
        except Exception:
            return shard.status

        if "Unknown Job Id" in out or "Unknown Job" in out:
            shard.status = "done"
            shard.finished_at = _now_iso()
            self.last_status = "done"
            return "done"

        if "job_state = R" in out:
            shard.status = "running"
            self.last_status = "running"
        elif "job_state = Q" in out:
            shard.status = "queued"
            self.last_status = "queued"
        elif "job_state = E" in out:
            shard.status = "error"
            self.last_status = "error"
        self.last_poll = time.time()
        return shard.status

    # ------------------------------------------------------------------
    def stop(self, shard: Shard) -> None:
        """Cancel the PBS job."""
        if not self.job_id:
            return
        try:
            self._ssh(f"qdel {self.job_id}")
        except Exception:
            pass
        shard.status = "stopped"
        shard.finished_at = _now_iso()

    # ------------------------------------------------------------------
    def _ssh(self, cmd: str) -> str:
        host = f"{self.config.user}@{self.config.host}"
        return subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", host, cmd],
            text=True,
            timeout=30,
        )

    def _scp(self, local: str, remote: str) -> None:
        host = f"{self.config.user}@{self.config.host}"
        subprocess.check_call(
            ["scp", "-o", "BatchMode=yes", local, f"{host}:{remote}"],
            timeout=60,
        )


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
