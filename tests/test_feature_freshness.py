# 検証対象：元DBの更新によるstale判定と、生成時の状態情報がない場合のunknown判定を確認する。
# 特徴量が存在することだけを理由に最新と判定しないことを検証する。
#
from __future__ import annotations

from datetime import datetime

import duckdb

from src.public_api import (
    feature_freshness,
    save_feature_run,
    save_features,
    source_data_state,
)
from tests.test_feature_storage import _feature_frame


# 一時ディレクトリに元レースDBを作り、生成サービスを本番DBから隔離する。
def _create_source(path):
    connection = duckdb.connect(str(path))
    connection.execute("""
        CREATE TABLE race_records (
            race_id VARCHAR,
            horse_id VARCHAR,
            race_date DATE,
            dataset_type VARCHAR,
            collected_at TIMESTAMP,
            collection_run_id VARCHAR
        )
    """)
    connection.execute("""
        INSERT INTO race_records VALUES
        ('r1', 'h1', '2026-01-01', 'historical', '2026-01-02 10:00:00', 'run-1')
    """)
    connection.close()


# 元DBへ行を追加するとfreshからstaleへ変わり、件数と収集runの差も返ることを確認する。
def test_freshness_detects_source_update(tmp_path):
    source_db = tmp_path / "racing.duckdb"
    # このテスト専用の特徴量DBパス。本番のfeatures.duckdbは使わない。
    feature_db = tmp_path / "features.duckdb"
    _create_source(source_db)
    state = source_data_state(source_db)
    from src.public_api import source_state_token

    save_features(_feature_frame("1", 1.0), feature_db)
    save_feature_run({
        "feature_run_id": "run-1",
        "feature_set_name": "baseline",
        "feature_set_version": "1",
        "generated_at": datetime(2026, 1, 2),
        "status": "completed",
        "source_state": state,
        "source_state_token": source_state_token(state),
    }, feature_db)

    fresh = feature_freshness(
        "baseline", "1", source_database=source_db, feature_database=feature_db
    )
    assert fresh["status"] == "fresh"

    connection = duckdb.connect(str(source_db))
    connection.execute("""
        INSERT INTO race_records VALUES
        ('r2', 'h2', '2026-01-03', 'historical', '2026-01-04 10:00:00', 'run-2')
    """)
    connection.close()

    stale = feature_freshness(
        "baseline", "1", source_database=source_db, feature_database=feature_db
    )
    assert stale["status"] == "stale"
    assert stale["differences"]["row_count"] == {"generated": 1, "current": 2}
    assert stale["differences"]["latest_collection_run_id"] == {
        "generated": "run-1", "current": "run-2"
    }


# 特徴量が保存済みでも生成元の状態情報がなければ、最新と断定せずunknownを返す。
def test_existing_features_without_source_state_are_unknown(tmp_path):
    # このテスト専用の特徴量DBパス。本番のfeatures.duckdbは使わない。
    feature_db = tmp_path / "features.duckdb"
    source_db = tmp_path / "racing.duckdb"
    _create_source(source_db)
    save_features(_feature_frame("1", 1.0), feature_db)
    save_feature_run({
        "feature_run_id": "old-run",
        "feature_set_name": "baseline",
        "feature_set_version": "1",
        "status": "completed",
    }, feature_db)

    result = feature_freshness(
        "baseline", "1", source_database=source_db, feature_database=feature_db
    )
    assert result["status"] == "unknown"
