from dataclasses import replace
from importlib import import_module

from src.public_api import generate_demo_records, normalize_race_frame


storage = import_module("src.00_common.storage")


def test_normalize_race_frame_has_expected_columns():
    historical, _ = generate_demo_records(historical_races=5, upcoming_races=1)
    historical["collection_run_id"] = "test"
    historical["collected_at"] = "2026-01-01"
    normalized = normalize_race_frame(historical)
    assert "race_id" in normalized.columns
    assert "finish_position" in normalized.columns
    assert "last_3f_time" in normalized.columns
    assert "passing_position_1" in normalized.columns
    assert "passing_position_4" in normalized.columns
    assert len(normalized) == len(historical)


def test_new_database_schema_has_all_current_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage,
        "PATHS",
        replace(storage.PATHS, database=tmp_path / "racing.duckdb"),
    )

    with storage.connect() as connection:
        race_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('race_records')").fetchall()
        }
        model_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('model_runs')").fetchall()
        }

    assert race_columns == set(storage.RACE_RECORD_COLUMNS)
    assert model_columns == set(storage.MODEL_RUN_COLUMNS)
