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


def test_freshness_detects_source_update(tmp_path):
    source_db = tmp_path / "racing.duckdb"
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


def test_existing_features_without_source_state_are_unknown(tmp_path):
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
