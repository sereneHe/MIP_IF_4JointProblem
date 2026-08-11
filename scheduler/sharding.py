"""Sharding and scheduling.

Splits the manifest's independent runs across the 4 fixed worker slots using
a "longest-task-first, load-balanced, each item <= 24 h" strategy.

If the total estimated time exceeds 24 h, each submitted batch is capped at
24 h; a batch that would exceed 24 h is split into two ~18 h batches, with
the later batch queued automatically after the earlier one finishes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import MAX_SLOT_SECONDS, SLOT_NAMES
from .estimator import format_duration
from .manifest import RunSpec


@dataclass
class Shard:
    """A batch of runs assigned to one slot."""

    slot: str
    shard_index: int
    run_ids: List[str] = field(default_factory=list)
    estimated_seconds: float = 0.0
    # For server slots: the PBS job id once submitted.
    job_id: str = ""
    # For local slots: the subprocess PID once started.
    pid: int = 0
    status: str = "pending"  # pending | running | queued | done | error | stopped
    started_at: str = ""
    finished_at: str = ""
    log_path: str = ""
    result_csv: str = ""
    mlflow_experiment: str = ""
    walltime: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ShardPlan:
    """The full sharding plan for one manifest."""

    plan_id: str
    shards: List[Shard] = field(default_factory=list)
    created_at: str = ""
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "notes": self.notes,
            "shards": [s.to_dict() for s in self.shards],
        }

    def save(self, data_dir: str) -> Path:
        path = Path(data_dir) / "shard_plans" / f"{self.plan_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, data_dir: str, plan_id: str) -> Optional["ShardPlan"]:
        path = Path(data_dir) / "shard_plans" / f"{self.plan_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        shards = [Shard(**s) for s in data.get("shards", [])]
        return cls(
            plan_id=data["plan_id"],
            shards=shards,
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_shard_plan(
    plan_id: str,
    runs: List[RunSpec],
    slot_names: Optional[List[str]] = None,
    max_slot_seconds: float = MAX_SLOT_SECONDS,
) -> ShardPlan:
    """Build a load-balanced shard plan.

    Strategy:
      - Sort runs by estimated time descending (longest first).
      - Greedily assign each run to the slot with the least current load.
      - If a single run alone exceeds max_slot_seconds, it is flagged in notes
        (cannot be safely split without checkpointing).
    """
    slot_names = slot_names or SLOT_NAMES
    slots = [{"name": n, "load": 0.0, "run_ids": []} for n in slot_names]

    # Longest first
    ordered = sorted(runs, key=lambda r: r.estimated_seconds, reverse=True)

    notes = []
    for run in ordered:
        if run.estimated_seconds > max_slot_seconds:
            notes.append(
                f"run {run.run_id} 预计 {format_duration(run.estimated_seconds)} "
                f"超过单槽上限 {format_duration(max_slot_seconds)}，"
                f"无法安全拆分（需 checkpoint 支持）。"
            )
        # pick least-loaded slot
        target = min(slots, key=lambda s: s["load"])
        target["load"] += run.estimated_seconds
        target["run_ids"].append(run.run_id)

    shards = []
    for i, slot in enumerate(slots):
        shards.append(
            Shard(
                slot=slot["name"],
                shard_index=i,
                run_ids=slot["run_ids"],
                estimated_seconds=slot["load"],
            )
        )

    return ShardPlan(
        plan_id=plan_id,
        shards=shards,
        created_at=_now_iso(),
        notes="\n".join(notes) if notes else "负载均衡分片完成。",
    )


def split_overlong_batch(
    run_ids: List[str],
    runs_by_id: Dict[str, RunSpec],
    max_slot_seconds: float = MAX_SLOT_SECONDS,
) -> List[List[str]]:
    """Split a batch whose total estimate exceeds max_slot_seconds.

    Returns a list of sub-batches, each with total estimate <= max_slot_seconds.
    Used to turn a >24 h batch into two ~18 h batches queued sequentially.
    """
    ordered = sorted(
        run_ids,
        key=lambda rid: runs_by_id[rid].estimated_seconds,
        reverse=True,
    )
    batches: List[List[str]] = [[]]
    loads = [0.0]
    for rid in ordered:
        est = runs_by_id[rid].estimated_seconds
        if loads[-1] + est > max_slot_seconds and batches[-1]:
            batches.append([])
            loads.append(0.0)
        batches[-1].append(rid)
        loads[-1] += est
    return batches
