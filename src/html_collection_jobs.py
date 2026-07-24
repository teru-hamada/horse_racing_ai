from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .config import PATHS
from .logging_utils import AppLogger
from .scrapers_netkeiba import (
    PAGE_REQUEST_INTERVAL_SECONDS,
    NetkeibaScraper,
)


@dataclass
class HtmlCollectionJob:
    job_id: str
    dataset_type: str
    start_date: date
    end_date: date
    force: bool
    status: str = "running"
    progress: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    result: dict[str, int] = field(default_factory=dict)
    error: str = ""
    logs: list[str] = field(default_factory=list)


_LOCK = threading.RLock()
_JOBS: dict[str, HtmlCollectionJob] = {}
_CANCEL_EVENTS: dict[str, threading.Event] = {}


class CollectionCancelled(Exception):
    """利用者からHTML収集の中止が要求された。"""


def _snapshot(job: HtmlCollectionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_type": job.dataset_type,
        "start_date": job.start_date,
        "end_date": job.end_date,
        "force": job.force,
        "status": job.status,
        "progress": job.progress,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": dict(job.result),
        "error": job.error,
        "logs": list(job.logs),
    }


def list_jobs(dataset_type: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        jobs = [
            _snapshot(job)
            for job in _JOBS.values()
            if dataset_type is None or job.dataset_type == dataset_type
        ]
    return sorted(jobs, key=lambda item: item["started_at"], reverse=True)


def start_job(
    dataset_type: str,
    start_date: date,
    end_date: date,
    force: bool,
) -> str:
    if dataset_type not in {"historical", "upcoming"}:
        raise ValueError(f"未対応のdataset_typeです: {dataset_type}")
    with _LOCK:
        for existing in _JOBS.values():
            if existing.dataset_type == dataset_type and existing.status == "running":
                raise RuntimeError("同じ種類のHTML収集がすでに実行中です。")
        job_id = (
            f"html_{dataset_type}_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        job = HtmlCollectionJob(
            job_id=job_id,
            dataset_type=dataset_type,
            start_date=start_date,
            end_date=end_date,
            force=force,
        )
        _JOBS[job_id] = job
        _CANCEL_EVENTS[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_job,
        args=(job_id,),
        name=job_id,
        daemon=True,
    )
    thread.start()
    return job_id


def cancel_job(job_id: str) -> bool:
    """実行中ジョブへ中止を要求する。すでに終了している場合はFalse。"""
    with _LOCK:
        job = _JOBS.get(job_id)
        cancel_event = _CANCEL_EVENTS.get(job_id)
        if (
            job is None
            or cancel_event is None
            or job.status not in {"running", "cancelling"}
        ):
            return False
        cancel_event.set()
        job.status = "cancelling"
        job.logs.append(
            f"[{datetime.now():%H:%M:%S}] INFO    中止要求を受け付けました。"
        )
        return True


def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    cancel_event = _CANCEL_EVENTS[job_id]

    def log_callback(line: str) -> None:
        with _LOCK:
            job.logs.append(line)
            job.logs[:] = job.logs[-300:]

    def progress_callback(value: float) -> None:
        if cancel_event.is_set():
            raise CollectionCancelled()
        with _LOCK:
            job.progress = min(max(float(value), 0.0), 1.0)

    logger = AppLogger(PATHS.logs, callback=log_callback)
    scraper = NetkeibaScraper(
        logger=logger,
        interval_seconds=PAGE_REQUEST_INTERVAL_SECONDS,
    )
    try:
        logger.info(
            f"HTML収集開始: {job.start_date}～{job.end_date}, "
            f"dataset_type={job.dataset_type}, force={job.force}"
        )
        result = scraper.collect_html_date_range(
            start_date=job.start_date,
            end_date=job.end_date,
            dataset_type=job.dataset_type,
            force=job.force,
            progress_callback=progress_callback,
        )
        with _LOCK:
            job.result = result
            job.progress = 1.0
            job.status = "completed"
            job.completed_at = datetime.now()
        logger.info(f"HTML収集完了: {result}")
    except CollectionCancelled:
        logger.info("HTML収集を中止しました。取得済みHTMLは保持します。")
        with _LOCK:
            job.status = "cancelled"
            job.completed_at = datetime.now()
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"HTML収集失敗: {exc}")
        with _LOCK:
            job.error = str(exc)
            job.status = "failed"
            job.completed_at = datetime.now()
