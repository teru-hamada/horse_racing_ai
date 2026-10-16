# 検証対象：レース表の列正規化と、新規DuckDBに作成されるテーブルの列構成を確認する。
# 定数と実DBの列一覧を比較し、新しい環境でも現行スキーマが揃うことを検証する。
#
from dataclasses import replace
from importlib import import_module

from src.public_api import generate_demo_records, normalize_race_frame


storage = import_module("src.00_common.storage")


# 正規化後に必須の結果列・通過順列が揃い、入力行数が変わらないことを確認する。
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


# 新規DBのレース・モデル・オッズ表が、定義定数と同じ列集合を持つことを確認する。
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
        # 登録日時などを除いた、オッズ保存前後の比較対象列。
        odds_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('race_odds')").fetchall()
        }

    assert race_columns == set(storage.RACE_RECORD_COLUMNS)
    assert model_columns == set(storage.MODEL_RUN_COLUMNS)
    assert odds_columns == set(storage.ODDS_COLUMNS)
