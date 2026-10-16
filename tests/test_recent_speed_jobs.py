# 検証対象：近走速度指数ジョブの完了状態・結果件数と、実行中の取消を確認する。
#
from __future__ import annotations

import time
from importlib import import_module

from src.public_api import RecentSpeedRunConfig


jobs = import_module("src.40_ai_modeling.feature_engineering.recent_speed_jobs")


# バックグラウンドジョブを上限時間までポーリングし、指定した終了状態の情報を返す。
def _wait(job_id: str, statuses: set[str]):
    for _ in range(100):
        job = next(item for item in jobs.list_recent_speed_jobs() if item["job_id"] == job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError("Recent-speed job did not finish")


# 近走速度生成サービスの結果が、完了状態と保存件数としてジョブへ伝わる。
def test_recent_speed_background_job_completes(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        log("started")
        progress(0.5)
        return {"row_count": 20, "status": "completed"}

    monkeypatch.setattr(jobs, "generate_recent_speed_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_recent_speed_job(RecentSpeedRunConfig())
    job = _wait(job_id, {"completed", "failed"})
    assert job["status"] == "completed"
    assert job["result"]["row_count"] == 20


# 近走速度生成中の取消要求で、協調的にcancelled状態へ移る。
def test_recent_speed_background_job_can_be_cancelled(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        while not is_cancelled():
            time.sleep(0.005)
        raise jobs.RecentSpeedGenerationCancelled()

    monkeypatch.setattr(jobs, "generate_recent_speed_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_recent_speed_job(RecentSpeedRunConfig())
    assert jobs.cancel_recent_speed_job(job_id)
    job = _wait(job_id, {"cancelled", "failed"})
    assert job["status"] == "cancelled"
