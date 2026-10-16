# 検証対象：速度指数生成ジョブの正常完了・結果件数と、取消要求による停止を確認する。
#
from __future__ import annotations

import time
from importlib import import_module

from src.public_api import SpeedIndexRunConfig


jobs = import_module("src.40_ai_modeling.feature_engineering.speed_jobs")


# バックグラウンドジョブを上限時間までポーリングし、指定した終了状態の情報を返す。
def _wait(job_id: str, statuses: set[str]):
    for _ in range(100):
        job = next(item for item in jobs.list_speed_index_jobs() if item["job_id"] == job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError("Speed-index job did not finish")


# 速度指数サービスの正常結果が、完了状態と生成件数としてジョブへ反映される。
def test_speed_index_background_job_completes(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        log("started")
        progress(0.5)
        return {"row_count": 20, "status": "completed"}

    monkeypatch.setattr(jobs, "generate_speed_index_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_speed_index_job(SpeedIndexRunConfig())
    job = _wait(job_id, {"completed", "failed"})
    assert job["status"] == "completed"
    assert job["result"]["row_count"] == 20


# 速度指数の計算中に取消を要求し、ジョブがcancelledへ移る。
def test_speed_index_background_job_can_be_cancelled(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        while not is_cancelled():
            time.sleep(0.005)
        raise jobs.SpeedIndexGenerationCancelled()

    monkeypatch.setattr(jobs, "generate_speed_index_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_speed_index_job(SpeedIndexRunConfig())
    assert jobs.cancel_speed_index_job(job_id)
    job = _wait(job_id, {"cancelled", "failed"})
    assert job["status"] == "cancelled"
