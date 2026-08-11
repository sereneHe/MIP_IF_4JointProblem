"""FastAPI backend for the Experiment Scheduling Center.

Exposes the scheduler operations over HTTP so the web UI can:
  - load a plan from a .sh script
  - start / stop / refresh
  - view slot status, integrity, reports
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .config import SchedulerConfig
from .orchestrator import Scheduler

app = FastAPI(title="Experiment Scheduling Center", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scheduler instance (created lazily).
_scheduler: Optional[Scheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = Scheduler(SchedulerConfig.from_env())
    return _scheduler


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class LoadPlanRequest(BaseModel):
    script_path: str
    plan_name: str = ""


class StartRequest(BaseModel):
    result_csv: str = ""
    mlflow_experiment: str = ""


class StopRequest(BaseModel):
    slot: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> HTMLResponse:
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status() -> Dict:
    return get_scheduler().status()


@app.post("/api/load_plan")
def api_load_plan(req: LoadPlanRequest) -> Dict:
    try:
        manifest = get_scheduler().load_plan(req.script_path, req.plan_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "plan_id": manifest.plan_id,
        "plan_name": manifest.plan_name,
        "parse_status": manifest.parse_status,
        "notes": manifest.notes,
        "total_runs": len(manifest.runs),
        "total_estimated_seconds": manifest.total_estimated_seconds,
        "shards": [s.to_dict() for s in get_scheduler().plan.shards],
    }


@app.post("/api/start")
def api_start(req: StartRequest) -> Dict:
    try:
        return get_scheduler().start_plan(req.result_csv, req.mlflow_experiment)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/refresh")
def api_refresh() -> Dict:
    return get_scheduler().refresh()


@app.post("/api/stop_shard")
def api_stop_shard(req: StopRequest) -> Dict:
    return get_scheduler().stop_shard(req.slot)


@app.post("/api/stop_plan")
def api_stop_plan() -> Dict:
    return get_scheduler().stop_plan()


@app.get("/api/report")
def api_report() -> Dict:
    return get_scheduler()._last_report


@app.get("/api/config")
def api_config() -> Dict:
    sched = get_scheduler()
    return {
        "slots": [s.to_dict() for s in sched.config.slots],
        "data_dir": sched.config.data_dir,
        "reports_dir": sched.config.reports_dir,
        "logs_dir": sched.config.logs_dir,
        "mlflow_uri": sched.config.mlflow_uri,
        "mlflow_experiment": sched.config.mlflow_experiment,
    }


@app.get("/api/logs/{slot}")
def api_log(slot: str) -> Dict:
    sched = get_scheduler()
    if sched.plan is None:
        raise HTTPException(status_code=404, detail="No plan loaded")
    for shard in sched.plan.shards:
        if shard.slot == slot:
            path = Path(shard.log_path)
            if not path.exists():
                return {"slot": slot, "content": "(日志文件不存在)"}
            return {"slot": slot, "content": path.read_text(errors="ignore")[-20000:]}
    raise HTTPException(status_code=404, detail=f"Slot {slot} not found")
