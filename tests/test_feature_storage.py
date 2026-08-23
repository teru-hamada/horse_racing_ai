from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.public_api import (
    clear_features,
    connect_feature_store,
    load_features,
    save_feature_run,
    save_features,
)


def _feature_frame(version: str, score: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "race_id": "r1",
        "horse_id": "h1",
        "race_date": "2026-01-01",
        "feature_set_name": "baseline",
        "feature_set_version": version,
        "history_cutoff": datetime(2025, 12, 31, 23, 59),
        "feature_run_id": f"run-{version}",
        "generated_at": datetime(2026, 1, 2),
        "sample_score": score,
    }])


def test_feature_store_has_independent_schema(tmp_path):
    feature_db = tmp_path / "features.duckdb"
    with connect_feature_store(feature_db) as connection:
        tables = {
            row[0]
            for row in connection.execute("SHOW TABLES").fetchall()
        }
    assert tables == {"feature_runs", "race_features"}
    assert feature_db.exists()


def test_feature_versions_can_coexist_and_one_can_be_cleared(tmp_path):
    feature_db = tmp_path / "features.duckdb"
    save_features(_feature_frame("1", 1.0), feature_db)
    save_features(_feature_frame("2", 2.0), feature_db)

    assert load_features("baseline", "1", feature_db)["sample_score"].iloc[0] == 1.0
    assert load_features("baseline", "2", feature_db)["sample_score"].iloc[0] == 2.0
    assert clear_features("baseline", "1", feature_db) == 1
    assert load_features("baseline", "1", feature_db).empty
    assert len(load_features("baseline", "2", feature_db)) == 1


def test_feature_run_configuration_is_recorded(tmp_path):
    feature_db = tmp_path / "features.duckdb"
    save_feature_run({
        "feature_run_id": "run-1",
        "feature_set_name": "baseline",
        "feature_set_version": "1",
        "status": "completed",
        "config": {"half_life_days": 180},
    }, feature_db)

    with connect_feature_store(feature_db) as connection:
        row = connection.execute(
            "SELECT feature_set_version, config_json FROM feature_runs"
        ).fetchone()
    assert row[0] == "1"
    assert '"half_life_days": 180' in row[1]
