"""Scheduler orchestrator.

Ties together the manifest, estimator, sharding, local/PBS workers, integrity
checker and reporting.  Provides the high-level operations the web UI calls:

  - parse a script into a manifest
  - build a shard plan
  - start a plan (launch local subprocesses / submit PBS jobs)
  - poll / refresh status
  - stop a shard or a whole plan
  - generate reports
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    HEALTH_RUNNING,
    HEALTH_STALLED,
    HEALTH_STOPPED,
    HEALTH_ERROR,
    HEARTBEAT_SUSPECT_SECONDS,
    SchedulerConfig,
    SlotConfig,
)
from .estimator import HistoryStore, estimate_manifest_runs, pbs_walltime
from .integrity import evaluate_slot_health
from .local_worker import LocalWorker
from .manifest import ExperimentManifest, RunSpec, parse_script_for_runs
from .pbs_worker import PBSWorker
from .reporting import (
    build_task_report,
    detect_risks,
    estimate_end_time,
    save_daily_report,
    save_report,
)
from .sharding import Shard, ShardPlan, build_shard_plan, split_overlong_batch


class Scheduler:
    """Top-level scheduler for one experiment plan."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        config.ensure_dirs()
        self.history = HistoryStore(config.data_dir)
        self.manifest: Optional[ExperimentManifest] = None
        self.plan: Optional[ShardPlan] = None
        self.runs_by_id: Dict[str, RunSpec] = {}
        self.local_workers: Dict[str, LocalWorker] = {}
        self.pbs_workers: Dict[str, PBSWorker] = {}
        self._lock = threading.Lock()
        self._last_report: Dict = {}
        self._last_daily_day: str = ""
        self._pending_items: List[Dict] = []

        for slot in config.slots:
            if slot.kind == "local":
                self.local_workers[slot.name] = LocalWorker(slot, config.logs_dir)
            else:
                self.pbs_workers[slot.name] = PBSWorker(slot, config.logs_dir)

    # ------------------------------------------------------------------
    # Plan lifecycle
    # ------------------------------------------------------------------
    def load_plan(self, script_path: str, plan_name: str = "") -> ExperimentManifest:
        """Parse a script into a manifest and build the shard plan."""
        manifest = parse_script_for_runs(script_path, plan_name=plan_name)
        estimate_manifest_runs(manifest.runs, self.history)
        manifest.total_estimated_seconds = sum(r.estimated_seconds for r in manifest.runs)
        manifest.save(self.config.data_dir)

        self.manifest = manifest
        self.runs_by_id = {r.run_id: r for r in manifest.runs}
        self.plan = build_shard_plan(manifest.plan_id, manifest.runs)
        self.plan.save(self.config.data_dir)
        return manifest

    def load_manifest(self, plan_id: str) -> Optional[ExperimentManifest]:
        manifest = ExperimentManifest.load(self.config.data_dir, plan_id)
        if manifest is None:
            return None
        self.manifest = manifest
        self.runs_by_id = {r.run_id: r for r in manifest.runs}
        self.plan = ShardPlan.load(self.config.data_dir, plan_id)
        return manifest

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    def start_plan(self, result_csv: str = "", mlflow_experiment: str = "") -> Dict:
        """Start all shards across the 4 slots."""
        if self.manifest is None or self.plan is None:
            raise RuntimeError("No plan loaded. Call load_plan first.")

        with self._lock:
            for shard in self.plan.shards:
                self._start_shard(shard, result_csv, mlflow_experiment)

        self._generate_report(result_csv, mlflow_experiment)
        return self.status()

    def _start_shard(self, shard: Shard, result_csv: str, mlflow_experiment: str) -> None:
        shard.result_csv = result_csv
        shard.mlflow_experiment = mlflow_experiment or self.manifest.plan_name
        shard.started_at = _now_iso()

        # Attach per-run env so MLflow results flow back to this slot.
        for rid in shard.run_ids:
            run = self.runs_by_id.get(rid)
            if run is not None:
                run.env["DASHBOARD_RUN_ID"] = rid
                run.env["MLFLOW_EXPERIMENT_NAME"] = shard.mlflow_experiment

        if shard.slot in self.local_workers:
            worker = self.local_workers[shard.slot]
            worker.start(shard, self.runs_by_id)
        elif shard.slot in self.pbs_workers:
            worker = self.pbs_workers[shard.slot]
            if worker.configured:
                walltime = pbs_walltime(shard.estimated_seconds)
                worker.submit(shard, self.runs_by_id, worker.config.project_dir, walltime)
            else:
                shard.status = "pending"
                shard.notes = "服务器槽位未配置，未自动提交。"

    # ------------------------------------------------------------------
    # Poll / refresh
    # ------------------------------------------------------------------
    def refresh(self) -> Dict:
        """Poll all workers and update shard statuses."""
        if self.plan is None:
            return self.status()

        with self._lock:
            for shard in self.plan.shards:
                if shard.slot in self.local_workers:
                    worker = self.local_workers[shard.slot]
                    worker.poll(shard)
                elif shard.slot in self.pbs_workers:
                    worker = self.pbs_workers[shard.slot]
                    if shard.job_id:
                        worker.poll(shard)

        self._maybe_daily_report()
        return self.status()

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    def stop_shard(self, slot: str) -> Dict:
        """Stop a single shard (stop after current run or immediately)."""
        if self.plan is None:
            return self.status()
        with self._lock:
            for shard in self.plan.shards:
                if shard.slot == slot:
                    if shard.slot in self.local_workers:
                        self.local_workers[shard.slot].stop(shard)
                    elif shard.slot in self.pbs_workers:
                        self.pbs_workers[shard.slot].stop(shard)
        return self.status()

    def stop_plan(self) -> Dict:
        """Stop all shards."""
        if self.plan is None:
            return self.status()
        with self._lock:
            for shard in self.plan.shards:
                if shard.slot in self.local_workers:
                    self.local_workers[shard.slot].stop(shard)
                elif shard.slot in self.pbs_workers:
                    self.pbs_workers[shard.slot].stop(shard)
        return self.status()

    # ------------------------------------------------------------------
    # Status / report
    # ------------------------------------------------------------------
    def status(self) -> Dict:
        """Build the full status dict shown in the UI."""
        if self.manifest is None or self.plan is None:
            return {"plan_id": "", "slots": [], "integrity": {}, "pending_items": []}

        result_csv = self.plan.shards[0].result_csv if self.plan.shards else ""
        slot_status = []
        for shard in self.plan.shards:
            slot_status.append(self._slot_status(shard, result_csv))

        integrity = self._overall_integrity(result_csv)
        return {
            "plan_id": self.manifest.plan_id,
            "plan_name": self.manifest.plan_name,
            "script": self.manifest.script,
            "total_runs": len(self.manifest.runs),
            "slots": slot_status,
            "integrity": integrity,
            "pending_items": self._pending_items,
            "report": self._last_report,
        }

    def _slot_status(self, shard: Shard, result_csv: str) -> Dict:
        is_alive = False
        heartbeat_age = 0.0
        if shard.slot in self.local_workers:
            worker = self.local_workers[shard.slot]
            is_alive = worker.is_alive()
            heartbeat_age = worker.heartbeat_age()
        elif shard.slot in self.pbs_workers:
            is_alive = shard.status in ("running", "queued")
            heartbeat_age = 0.0

        health = evaluate_slot_health(
            shard, is_alive, heartbeat_age, shard.log_path, result_csv,
            self.manifest, self.plan.shards,
        )

        # Create a pending item if stalled.
        if health["health"] == HEALTH_STALLED:
            item = {
                "slot": shard.slot,
                "type": "疑似卡住",
                "detail": f"{shard.slot} 超过 {HEARTBEAT_SUSPECT_SECONDS // 60} 分钟无日志/结果更新。",
                "created_at": _now_iso(),
            }
            if item not in self._pending_items:
                self._pending_items.append(item)

        return {
            "slot": shard.slot,
            "kind": "local" if shard.slot in self.local_workers else "server",
            "status": shard.status,
            "health": health["health"],
            "errors": health["errors"],
            "heartbeat_age": heartbeat_age,
            "run_count": len(shard.run_ids),
            "run_ids": shard.run_ids,
            "estimated_seconds": shard.estimated_seconds,
            "estimated_end": estimate_end_time(shard),
            "job_id": shard.job_id,
            "pid": shard.pid,
            "log_path": shard.log_path,
            "result_csv": shard.result_csv or result_csv,
            "mlflow_experiment": shard.mlflow_experiment,
            "walltime": shard.walltime,
            "integrity": health["integrity"],
        }

    def _overall_integrity(self, result_csv: str) -> Dict:
        from .integrity import check_manifest_integrity
        return check_manifest_integrity(self.manifest, self.plan.shards, result_csv)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def _generate_report(self, result_csv: str, mlflow_experiment: str) -> None:
        risks = detect_risks(
            self.manifest,
            self.plan,
            server_configured=any(w.configured for w in self.pbs_workers.values()),
            license_file=self.config.slots[2].license_file if len(self.config.slots) > 2 else "",
        )
        report = build_task_report(
            self.manifest, self.plan, self.plan.shards, self.runs_by_id,
            result_csv=result_csv, mlflow_uri=self.config.mlflow_uri, risks=risks,
        )
        self._last_report = report
        save_report(report, self.config.reports_dir, self.manifest.plan_id)

    def _maybe_daily_report(self) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day != self._last_daily_day:
            self._last_daily_day = day
            if self._last_report:
                save_daily_report(self._last_report, self.config.reports_dir,
                                  self.manifest.plan_id)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
