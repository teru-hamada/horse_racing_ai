from __future__ import annotations

import time
from importlib import import_module

from src.public_api import SpeedIndexRunConfig


jobs = import_module("src.30_ai_modeling.feature_engineering.speed_jobs")


def _wait(job_id: str, statuses: set[str]):
    for _ in range(100):
        job = next(item for item in jobs.list_speed_index_jobs() if item["job_id"] == job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError("Speed-index job did not finish")


def test_speed_index_background_job_completes(monkeypatch):
    def generate(config, *, log, progress, is_cancelled):
        log("started")
        progress(0.5)
        return {"row_count": 20, "status": "completed"}

    monkeypatch.setattr(jobs, "generate_speed_index_features", generate)
    job_id = jobs.start_speed_index_job(SpeedIndexRunConfig())
    job = _wait(job_id, {"completed", "failed"})
    assert job["status"] == "completed"
    assert job["result"]["row_count"] == 20


def test_speed_index_background_job_can_be_cancelled(monkeypatch):
    def generate(config, *, log, progress, is_cancelled):
        while not is_cancelled():
            time.sleep(0.005)
        raise jobs.SpeedIndexGenerationCancelled()

    monkeypatch.setattr(jobs, "generate_speed_index_features", generate)
    job_id = jobs.start_speed_index_job(SpeedIndexRunConfig())
    assert jobs.cancel_speed_index_job(job_id)
    job = _wait(job_id, {"cancelled", "failed"})
    assert job["status"] == "cancelled"
