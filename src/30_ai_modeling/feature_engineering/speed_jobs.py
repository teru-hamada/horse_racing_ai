from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .speed_service import (
    SpeedIndexGenerationCancelled,
    SpeedIndexRunConfig,
    generate_speed_index_features,
)


@dataclass
class SpeedIndexGenerationJob:
    job_id: str
    config: SpeedIndexRunConfig
    status: str = "running"
    progress: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    result: dict[str, object] = field(default_factory=dict)
    error: str = ""
    logs: list[str] = field(default_factory=list)


_LOCK = threading.RLock()
_JOBS: dict[str, SpeedIndexGenerationJob] = {}
_CANCEL_EVENTS: dict[str, threading.Event] = {}


def _snapshot(job: SpeedIndexGenerationJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "config": job.config,
        "status": job.status,
        "progress": job.progress,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": dict(job.result),
        "error": job.error,
        "logs": list(job.logs),
    }


def list_speed_index_jobs() -> list[dict[str, Any]]:
    with _LOCK:
        jobs = [_snapshot(job) for job in _JOBS.values()]
    return sorted(jobs, key=lambda item: item["started_at"], reverse=True)


def start_speed_index_job(config: SpeedIndexRunConfig) -> str:
    with _LOCK:
        if any(job.status in {"running", "cancelling"} for job in _JOBS.values()):
            raise RuntimeError("Speed-index generation is already running")
        job_id = f"speed_job_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        _JOBS[job_id] = SpeedIndexGenerationJob(job_id=job_id, config=config)
        _CANCEL_EVENTS[job_id] = threading.Event()
    threading.Thread(
        target=_run_job, args=(job_id,), name=job_id, daemon=True
    ).start()
    return job_id


def cancel_speed_index_job(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        event = _CANCEL_EVENTS.get(job_id)
        if job is None or event is None or job.status not in {"running", "cancelling"}:
            return False
        event.set()
        job.status = "cancelling"
        job.logs.append("中止を要求しました。完成済みデータは維持されます。")
        return True


def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    event = _CANCEL_EVENTS[job_id]

    def log(message: str) -> None:
        with _LOCK:
            job.logs.append(f"[{datetime.now():%H:%M:%S}] {message}")
            job.logs[:] = job.logs[-300:]

    def progress(value: float) -> None:
        with _LOCK:
            job.progress = min(max(float(value), 0.0), 1.0)

    try:
        result = generate_speed_index_features(
            job.config,
            log=log,
            progress=progress,
            is_cancelled=event.is_set,
        )
        with _LOCK:
            job.result = result
            job.progress = 1.0
            job.status = "completed"
            job.completed_at = datetime.now()
    except SpeedIndexGenerationCancelled:
        with _LOCK:
            job.status = "cancelled"
            job.completed_at = datetime.now()
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            job.error = str(exc)
            job.status = "failed"
            job.completed_at = datetime.now()
