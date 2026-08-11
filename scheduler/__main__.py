"""CLI entrypoint for the Experiment Scheduling Center.

Usage:
    python -m scheduler serve            # start the FastAPI web UI
    python -m scheduler parse <script>   # parse a .sh into a manifest
    python -m scheduler plan <script>    # parse + build shard plan (dry run)
"""

from __future__ import annotations

import json
import sys

from .config import SchedulerConfig
from .estimator import format_duration
from .manifest import parse_script_for_runs
from .orchestrator import Scheduler


def cmd_serve() -> None:
    import uvicorn
    from .api import app
    uvicorn.run(app, host="127.0.0.1", port=8000)


def cmd_parse(script: str) -> None:
    manifest = parse_script_for_runs(script)
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))


def cmd_plan(script: str) -> None:
    sched = Scheduler(SchedulerConfig.from_env())
    manifest = sched.load_plan(script)
    print(f"计划: {manifest.plan_id} ({manifest.plan_name})")
    print(f"解析状态: {manifest.parse_status}")
    print(f"总 runs: {len(manifest.runs)}")
    print(f"预计总时长: {format_duration(manifest.total_estimated_seconds)}")
    print(f"备注: {manifest.notes}")
    print()
    for shard in sched.plan.shards:
        print(
            f"  {shard.slot}: {len(shard.run_ids)} runs, "
            f"预计 {format_duration(shard.estimated_seconds)}"
        )


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "serve":
        cmd_serve()
    elif cmd == "parse" and len(args) >= 2:
        cmd_parse(args[1])
    elif cmd == "plan" and len(args) >= 2:
        cmd_plan(args[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
