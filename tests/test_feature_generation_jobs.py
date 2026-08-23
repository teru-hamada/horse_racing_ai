from __future__ import annotations

import time
from importlib import import_module

from src.public_api import RecentFormRunConfig


jobs = import_module("src.30_ai_modeling.feature_engineering.jobs")


def _wait_for(job_id: str, statuses: set[str]) -> dict[str, object]:
    for _ in range(100):
        job = next(
            item for item in jobs.list_feature_generation_jobs()
            if item["job_id"] == job_id
        )
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach {statuses}")


def test_background_feature_job_completes(monkeypatch):
    def generate(config, *, log, progress, is_cancelled):
        log("started")
        progress(0.5)
        return {"row_count": 12, "status": "completed"}

    monkeypatch.setattr(jobs, "generate_recent_form_features", generate)
    job_id = jobs.start_feature_generation_job(RecentFormRunConfig())
    job = _wait_for(job_id, {"completed", "failed"})

    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["result"]["row_count"] == 12
    assert "started" in job["logs"][0]


def test_background_feature_job_can_be_cancelled(monkeypatch):
    def generate(config, *, log, progress, is_cancelled):
        while not is_cancelled():
            time.sleep(0.005)
        raise jobs.FeatureGenerationCancelled()

    monkeypatch.setattr(jobs, "generate_recent_form_features", generate)
    job_id = jobs.start_feature_generation_job(RecentFormRunConfig())
    assert jobs.cancel_feature_generation_job(job_id)
    job = _wait_for(job_id, {"cancelled", "failed"})

    assert job["status"] == "cancelled"
