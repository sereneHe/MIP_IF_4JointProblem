"""Experiment plan manifest.

A manifest records exactly what an approved experiment plan is expected to
produce:

    method x dataset x parameter-group x seed

plus the expected total run count, a unique id per run, the target CSV path
and the MLflow run info.  The integrity checker compares the CSV and MLflow
against this manifest.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import STATUS_UNKNOWN


@dataclass
class RunSpec:
    """A single independent run unit."""

    run_id: str
    method: str
    dataset: str
    param_group: str
    seed: int
    estimated_seconds: float = 0.0
    command: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    result_csv: str = ""
    mlflow_run_id: str = ""
    mlflow_experiment: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentManifest:
    """Full manifest for one approved experiment plan."""

    plan_id: str
    plan_name: str
    script: str
    created_at: str
    runs: List[RunSpec] = field(default_factory=list)
    total_estimated_seconds: float = 0.0
    notes: str = ""
    parse_status: str = STATUS_UNKNOWN  # "ok" | "needs_confirmation" | "unknown"

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "script": self.script,
            "created_at": self.created_at,
            "total_estimated_seconds": self.total_estimated_seconds,
            "notes": self.notes,
            "parse_status": self.parse_status,
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentManifest":
        runs = [RunSpec(**r) for r in data.get("runs", [])]
        return cls(
            plan_id=data["plan_id"],
            plan_name=data["plan_name"],
            script=data["script"],
            created_at=data["created_at"],
            runs=runs,
            total_estimated_seconds=data.get("total_estimated_seconds", 0.0),
            notes=data.get("notes", ""),
            parse_status=data.get("parse_status", STATUS_UNKNOWN),
        )

    # ------------------------------------------------------------------
    def save(self, data_dir: str) -> Path:
        path = Path(data_dir) / "manifests" / f"{self.plan_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, data_dir: str, plan_id: str) -> Optional["ExperimentManifest"]:
        path = Path(data_dir) / "manifests" / f"{plan_id}.json"
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def list_all(cls, data_dir: str) -> List["ExperimentManifest"]:
        manifests_dir = Path(data_dir) / "manifests"
        if not manifests_dir.exists():
            return []
        out = []
        for p in sorted(manifests_dir.glob("*.json")):
            try:
                out.append(cls.from_dict(json.loads(p.read_text())))
            except Exception:
                continue
        return out


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# A run id is stable across shards: method|dataset|param_group|seed
def make_run_id(method: str, dataset: str, param_group: str, seed: int) -> str:
    return f"{method}|{dataset}|{param_group}|{seed}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_plan_id() -> str:
    return datetime.now(timezone.utc).strftime("plan_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


# ---------------------------------------------------------------------------
# Heuristic parser for the existing .sh experiment scripts
# ---------------------------------------------------------------------------

# Recognise lines like:
#   python run_experiments.py experiment=... ++problem.seed=1 ...
#   python run_experiments.py experiment=... problem.seed=1 ...
#   python run_experiments.py experiment=... seed=1 ...
#   uv run python -m mixture_lds.experiments --multirun ... seed="30"
_RUN_LINE_RE = re.compile(
    r"(?:python|python3|uv\s+run)\s+"
    r"(?:(?:-m\s+)?(?P<script>\S*experiments\.py)|-m\s+(?P<module>\S+))"
    r"(?P<args>.*)"
)

# A variable holding the base command, e.g. CMD="uv run python -m ...experiments ..."
_CMD_DEF_RE = re.compile(
    r"^\s*CMD\s*=\s*[\"'](?P<cmd>.*experiments.*)[\"']\s*$"
)

# An invocation of that variable, e.g. ${CMD} \  or  $CMD \
_CMD_INVOKE_RE = re.compile(r"^\s*\$\{?CMD\}?\s*(?P<args>.*)$")


def _extract_kv(args: str) -> Dict[str, str]:
    """Pull key=value pairs out of a command line, ignoring ++/-- prefixes."""
    out: Dict[str, str] = {}
    for token in args.split():
        token = token.strip()
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        key = key.lstrip("+-")
        out[key] = value
    return out


def _collect_continuation(lines: List[str], idx: int) -> str:
    """Join a command line with its backslash-continuation lines."""
    parts = [lines[idx]]
    j = idx
    while j < len(lines) and lines[j].rstrip().endswith("\\"):
        j += 1
        if j < len(lines):
            parts.append(lines[j])
    return " ".join(p.strip() for p in parts)


def parse_script_for_runs(
    script_path: str,
    plan_name: str = "",
    plan_id: str = "",
) -> ExperimentManifest:
    """Parse a .sh file into a manifest of independent run units.

    Returns a manifest with parse_status:
      - "ok"                 if we found run lines with seed/dataset/method
      - "needs_confirmation" if we found run lines but cannot fully split
      - "unknown"            if nothing parseable was found
    """
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    plan_id = plan_id or new_plan_id()
    plan_name = plan_name or path.stem

    lines = path.read_text().splitlines()
    runs: List[RunSpec] = []
    seen: set = set()

    # Track a CMD=... base command if present.
    base_cmd = ""
    for line in lines:
        m = _CMD_DEF_RE.match(line.strip())
        if m:
            base_cmd = m.group("cmd")

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Case 1: direct python invocation.
        m = _RUN_LINE_RE.search(line)
        if m:
            full = _collect_continuation(lines, i)
            kv = _extract_kv(full)
            method = kv.get("experiment") or kv.get("solver") or "unknown"
            dataset = kv.get("problem.dataset") or kv.get("dataset") or "unknown"
            seed = kv.get("problem.seed") or kv.get("seed") or kv.get("++problem.seed")
            param_group = kv.get("problem.intra_edge_ratio") or kv.get("ratio") or "default"
            if seed is not None:
                try:
                    seed_int = int(seed)
                except ValueError:
                    seed_int = 0
                run_id = make_run_id(method, dataset, param_group, seed_int)
                if run_id not in seen:
                    seen.add(run_id)
                    runs.append(RunSpec(
                        run_id=run_id, method=method, dataset=dataset,
                        param_group=param_group, seed=seed_int, command=full,
                    ))
            continue

        # Case 2: ${CMD} invocation with continuation lines.
        if base_cmd:
            m = _CMD_INVOKE_RE.match(line)
            if m:
                full = _collect_continuation(lines, i)
                kv = _extract_kv(full)
                method = kv.get("experiment") or kv.get("solver") or "unknown"
                dataset = kv.get("problem.dataset") or kv.get("dataset") or "unknown"
                seed = kv.get("problem.seed") or kv.get("seed") or kv.get("++problem.seed")
                param_group = kv.get("problem.intra_edge_ratio") or kv.get("ratio") or "default"
                if seed is not None:
                    try:
                        seed_int = int(seed)
                    except ValueError:
                        seed_int = 0
                    run_id = make_run_id(method, dataset, param_group, seed_int)
                    if run_id not in seen:
                        seen.add(run_id)
                        runs.append(RunSpec(
                            run_id=run_id, method=method, dataset=dataset,
                            param_group=param_group, seed=seed_int,
                            command=f"{base_cmd} {full}",
                        ))
                continue


    if not runs:
        return ExperimentManifest(
            plan_id=plan_id,
            plan_name=plan_name,
            script=str(path),
            created_at=now_iso(),
            runs=[],
            parse_status=STATUS_UNKNOWN,
            notes="无法自动解析该脚本；请在界面中补充任务定义。",
        )

    # If every run line had a seed, we can split; otherwise needs confirmation.
    parse_status = "ok"
    notes = f"解析到 {len(runs)} 个独立 run。"
    if len(seen) < len(runs):
        parse_status = "needs_confirmation"
        notes = "部分 run 缺少 seed，无法完全自动拆分；请在界面确认。"

    return ExperimentManifest(
        plan_id=plan_id,
        plan_name=plan_name,
        script=str(path),
        created_at=now_iso(),
        runs=runs,
        total_estimated_seconds=sum(r.estimated_seconds for r in runs),
        notes=notes,
        parse_status=parse_status,
    )
