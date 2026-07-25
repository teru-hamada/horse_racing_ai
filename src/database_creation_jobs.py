from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .config import PATHS
from .logging_utils import AppLogger
from .scrapers_netkeiba import NetkeibaScraper
from .storage import save_collection_run, save_race_records


@dataclass
class DatabaseCreationJob:
    job_id: str
    dataset_type: str
    start_date: date
    end_date: date
    status: str = "running"
    progress: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    result: dict[str, int] = field(default_factory=dict)
    error: str = ""
    logs: list[str] = field(default_factory=list)


_LOCK = threading.RLock()
_JOBS: dict[str, DatabaseCreationJob] = {}
_CANCEL_EVENTS: dict[str, threading.Event] = {}


class DatabaseCreationCancelled(Exception):
    """利用者からデータベース作成の中止が要求された。"""


def _snapshot(job: DatabaseCreationJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_type": job.dataset_type,
        "start_date": job.start_date,
        "end_date": job.end_date,
        "status": job.status,
        "progress": job.progress,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": dict(job.result),
        "error": job.error,
        "logs": list(job.logs),
    }


def list_database_jobs(
    dataset_type: str | None = None,
) -> list[dict[str, Any]]:
    with _LOCK:
        jobs = [
            _snapshot(job)
            for job in _JOBS.values()
            if (
                dataset_type is None
                or job.dataset_type == dataset_type
            )
        ]
    return sorted(
        jobs,
        key=lambda item: item["started_at"],
        reverse=True,
    )


def start_database_job(
    dataset_type: str,
    start_date: date,
    end_date: date,
) -> str:
    if dataset_type not in {"historical", "upcoming"}:
        raise ValueError(
            f"未対応のdataset_typeです: {dataset_type}"
        )
    with _LOCK:
        if any(
            job.status in {"running", "cancelling"}
            for job in _JOBS.values()
        ):
            raise RuntimeError(
                "データベース作成がすでに実行中です。"
            )
        job_id = (
            f"database_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        _JOBS[job_id] = DatabaseCreationJob(
            job_id=job_id,
            dataset_type=dataset_type,
            start_date=start_date,
            end_date=end_date,
        )
        _CANCEL_EVENTS[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_job,
        args=(job_id,),
        name=job_id,
        daemon=True,
    )
    thread.start()
    return job_id


def cancel_database_job(job_id: str) -> bool:
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
            f"[{datetime.now():%H:%M:%S}] INFO    "
            "中止要求を受け付けました。"
        )
        return True


def _save_run(
    job: DatabaseCreationJob,
    *,
    status: str,
    race_count: int,
    row_count: int,
    message: str,
) -> None:
    save_collection_run(
        {
            "run_id": job.job_id,
            "started_at": job.started_at,
            "completed_at": datetime.now(),
            "status": status,
            "dataset_type": job.dataset_type,
            "source": "cached_html",
            "start_date": job.start_date,
            "end_date": job.end_date,
            "race_count": race_count,
            "row_count": row_count,
            "output_path": "",
            "message": message,
        }
    )


def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    cancel_event = _CANCEL_EVENTS[job_id]

    def log_callback(line: str) -> None:
        with _LOCK:
            job.logs.append(line)
            job.logs[:] = job.logs[-300:]

    def progress_callback(value: float) -> None:
        if cancel_event.is_set():
            raise DatabaseCreationCancelled()
        with _LOCK:
            job.progress = min(
                max(float(value), 0.0),
                0.95,
            )

    logger = AppLogger(PATHS.logs, callback=log_callback)
    scraper = NetkeibaScraper(logger=logger)
    try:
        logger.info(
            f"データベース作成開始: "
            f"{job.start_date}～{job.end_date}, "
            f"dataset_type={job.dataset_type}"
        )
        frame = scraper.parse_cached_date_range(
            start_date=job.start_date,
            end_date=job.end_date,
            dataset_type=job.dataset_type,
            progress_callback=progress_callback,
        )
        if cancel_event.is_set():
            raise DatabaseCreationCancelled()
        if frame.empty:
            raise ValueError(
                "対象期間の解析可能なHTMLがありません。"
                "先にHTML収集を実行してください。"
            )

        logger.info(
            f"解析完了: {frame['race_id'].nunique():,}レース / "
            f"{len(frame):,}頭。データベースへ保存します。"
        )
        save_race_records(
            frame,
            job.job_id,
            job.dataset_type,
        )
        race_count = int(frame["race_id"].nunique())
        row_count = len(frame)
        result = {
            "race_count": race_count,
            "row_count": row_count,
        }
        _save_run(
            job,
            status="completed",
            race_count=race_count,
            row_count=row_count,
            message="取得済みHTMLからデータベース作成完了",
        )
        with _LOCK:
            job.result = result
            job.progress = 1.0
            job.status = "completed"
            job.completed_at = datetime.now()
        logger.info(
            f"データベース作成完了: {race_count:,}レース / "
            f"{row_count:,}頭"
        )
    except DatabaseCreationCancelled:
        logger.info(
            "データベース作成を中止しました。"
            "保存前の解析結果は登録していません。"
        )
        _save_run(
            job,
            status="cancelled",
            race_count=0,
            row_count=0,
            message="データベース作成を中止",
        )
        with _LOCK:
            job.status = "cancelled"
            job.completed_at = datetime.now()
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"データベース作成失敗: {exc}")
        _save_run(
            job,
            status="failed",
            race_count=0,
            row_count=0,
            message=str(exc),
        )
        with _LOCK:
            job.error = str(exc)
            job.status = "failed"
            job.completed_at = datetime.now()
