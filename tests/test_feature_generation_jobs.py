# 検証対象：近走成績生成ジョブの完了・進捗・ログ通知と、実行中の協調的取消を確認する。
#
from __future__ import annotations

import time
from importlib import import_module

from src.public_api import RecentFormRunConfig


jobs = import_module("src.40_ai_modeling.feature_engineering.jobs")


# 非同期ジョブが対象の状態に達するまで有限時間だけ待ち、未終了ならテストを失敗させる。
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


# 代替サービスからの進捗・ログ・結果がジョブへ伝わり、completedと進捗1.0になることを確認する。
def test_background_feature_job_completes(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        log("started")
        progress(0.5)
        return {"row_count": 12, "status": "completed"}

    monkeypatch.setattr(jobs, "generate_recent_form_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_feature_generation_job(RecentFormRunConfig())
    job = _wait_for(job_id, {"completed", "failed"})

    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["result"]["row_count"] == 12
    assert "started" in job["logs"][0]


# 実行中のジョブへ取消を要求し、サービス側の取消検知によってcancelledへ移ることを確認する。
def test_background_feature_job_can_be_cancelled(monkeypatch):
    # 本物の生成サービスを置き換え、進捗・取消などテスト対象の挙動を制御する。
    def generate(config, *, log, progress, is_cancelled):
        while not is_cancelled():
            time.sleep(0.005)
        raise jobs.FeatureGenerationCancelled()

    monkeypatch.setattr(jobs, "generate_recent_form_features", generate)
    # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
    job_id = jobs.start_feature_generation_job(RecentFormRunConfig())
    assert jobs.cancel_feature_generation_job(job_id)
    job = _wait_for(job_id, {"cancelled", "failed"})

    assert job["status"] == "cancelled"
