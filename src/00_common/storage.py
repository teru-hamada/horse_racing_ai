from __future__ import annotations

import json
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

PATHS = import_module("src.00_common.config").PATHS


RACE_RECORD_COLUMNS = [
    "race_id", "race_date", "course_name", "race_number", "race_name",
    "surface", "distance", "weather", "track_condition", "horse_id",
    "horse_name", "horse_number", "frame_number", "sex", "age",
    "carried_weight", "jockey_id", "jockey_name", "trainer_id", "trainer_name",
    "sire_id", "sire_name", "dam_id", "dam_name",
    "damsire_id", "damsire_name",
    "odds", "popularity", "body_weight", "body_weight_change", "finish_position",
    "time_seconds", "last_3f_time",
    "passing_position_1", "passing_position_2",
    "passing_position_3", "passing_position_4",
    "dataset_type", "collection_run_id", "collected_at",
]

MODEL_RUN_COLUMNS = [
    "model_run_id", "created_at", "status", "target",
    "train_rows", "validation_rows", "test_rows", "best_epoch",
    "validation_auc", "test_auc", "test_log_loss", "model_path",
    "metrics_json", "task_name", "estimator_name", "model_version",
    "feature_version",
]


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collection_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR,
    dataset_type VARCHAR,
    source VARCHAR,
    start_date DATE,
    end_date DATE,
    race_count BIGINT,
    row_count BIGINT,
    output_path VARCHAR,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS race_records (
    race_id VARCHAR,
    race_date DATE,
    course_name VARCHAR,
    race_number INTEGER,
    race_name VARCHAR,
    surface VARCHAR,
    distance DOUBLE,
    weather VARCHAR,
    track_condition VARCHAR,
    horse_id VARCHAR,
    horse_name VARCHAR,
    horse_number INTEGER,
    frame_number INTEGER,
    sex VARCHAR,
    age INTEGER,
    carried_weight DOUBLE,
    jockey_id VARCHAR,
    jockey_name VARCHAR,
    trainer_id VARCHAR,
    trainer_name VARCHAR,
    sire_id VARCHAR,
    sire_name VARCHAR,
    dam_id VARCHAR,
    dam_name VARCHAR,
    damsire_id VARCHAR,
    damsire_name VARCHAR,
    odds DOUBLE,
    popularity DOUBLE,
    body_weight DOUBLE,
    body_weight_change DOUBLE,
    finish_position DOUBLE,
    time_seconds DOUBLE,
    last_3f_time DOUBLE,
    passing_position_1 INTEGER,
    passing_position_2 INTEGER,
    passing_position_3 INTEGER,
    passing_position_4 INTEGER,
    dataset_type VARCHAR,
    collection_run_id VARCHAR,
    collected_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    status VARCHAR,
    target VARCHAR,
    train_rows BIGINT,
    validation_rows BIGINT,
    test_rows BIGINT,
    best_epoch INTEGER,
    validation_auc DOUBLE,
    test_auc DOUBLE,
    test_log_loss DOUBLE,
    model_path VARCHAR,
    metrics_json VARCHAR,
    task_name VARCHAR,
    estimator_name VARCHAR,
    model_version VARCHAR,
    feature_version VARCHAR
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    prediction_run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    model_run_id VARCHAR,
    race_id VARCHAR,
    row_count BIGINT,
    output_path VARCHAR
);
"""


def connect() -> duckdb.DuckDBPyConnection:
    PATHS.database.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(PATHS.database))
    con.execute(SCHEMA_SQL)
    for column in [
        "sire_id",
        "sire_name",
        "dam_id",
        "dam_name",
        "damsire_id",
        "damsire_name",
    ]:
        con.execute(
            f"ALTER TABLE race_records "
            f"ADD COLUMN IF NOT EXISTS {column} VARCHAR"
        )
    for column in [
        "last_3f_time",
    ]:
        con.execute(
            f"ALTER TABLE race_records "
            f"ADD COLUMN IF NOT EXISTS {column} DOUBLE"
        )
    for column in [
        "passing_position_1",
        "passing_position_2",
        "passing_position_3",
        "passing_position_4",
    ]:
        con.execute(
            f"ALTER TABLE race_records "
            f"ADD COLUMN IF NOT EXISTS {column} INTEGER"
        )
    for column in [
        "task_name",
        "estimator_name",
        "model_version",
        "feature_version",
    ]:
        con.execute(
            f"ALTER TABLE model_runs "
            f"ADD COLUMN IF NOT EXISTS {column} VARCHAR"
        )
    return con


def empty_race_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RACE_RECORD_COLUMNS)


def normalize_race_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in RACE_RECORD_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[RACE_RECORD_COLUMNS]
    result["race_date"] = pd.to_datetime(result["race_date"], errors="coerce").dt.date
    result["collected_at"] = pd.to_datetime(result["collected_at"], errors="coerce")
    numeric_cols = [
        "race_number", "distance", "horse_number", "frame_number", "age",
        "carried_weight", "odds", "popularity", "body_weight",
        "body_weight_change", "finish_position", "time_seconds",
        "last_3f_time",
        "passing_position_1", "passing_position_2",
        "passing_position_3", "passing_position_4",
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    string_cols = [c for c in RACE_RECORD_COLUMNS if c not in numeric_cols + ["race_date", "collected_at"]]
    for col in string_cols:
        result[col] = result[col].astype("string")
    return result


def save_collection_run(metadata: dict[str, Any]) -> None:
    row = pd.DataFrame([metadata])
    with connect() as con:
        con.register("run_row", row)
        con.execute("DELETE FROM collection_runs USING run_row WHERE collection_runs.run_id = run_row.run_id")
        con.execute("INSERT INTO collection_runs SELECT * FROM run_row")
        con.unregister("run_row")


def save_race_records(
    df: pd.DataFrame,
    run_id: str,
    dataset_type: str,
) -> None:
    if df.empty:
        raise ValueError("保存対象のレースデータが0件です。")

    normalized = normalize_race_frame(df)
    normalized["dataset_type"] = dataset_type
    normalized["collection_run_id"] = run_id
    normalized["collected_at"] = datetime.now()

    with connect() as con:
        con.register("incoming", normalized)
        con.execute(
            """
            DELETE FROM race_records
            USING incoming
            WHERE race_records.race_id = incoming.race_id
              AND COALESCE(race_records.horse_id, '') = COALESCE(incoming.horse_id, '')
              AND race_records.dataset_type = incoming.dataset_type
            """
        )
        column_list = ", ".join(RACE_RECORD_COLUMNS)
        con.execute(
            f"INSERT INTO race_records ({column_list}) "
            f"SELECT {column_list} FROM incoming"
        )
        con.unregister("incoming")


def load_records(dataset_type: str | None = None) -> pd.DataFrame:
    with connect() as con:
        if dataset_type:
            return con.execute(
                "SELECT * FROM race_records WHERE dataset_type = ? ORDER BY race_date, race_id, horse_number",
                [dataset_type],
            ).df()
        return con.execute("SELECT * FROM race_records ORDER BY race_date, race_id, horse_number").df()


def race_record_summary(
    dataset_type: str,
    start_date: Any,
    end_date: Any,
) -> dict[str, int]:
    with connect() as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT race_id) AS race_count
            FROM race_records
            WHERE dataset_type = ?
              AND race_date BETWEEN ? AND ?
            """,
            [dataset_type, start_date, end_date],
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "race_count": int(row[1]),
    }


def race_record_years(
    dataset_type: str,
) -> set[int]:
    with connect() as con:
        rows = con.execute(
            """
            SELECT DISTINCT YEAR(race_date)
            FROM race_records
            WHERE dataset_type = ?
              AND race_date IS NOT NULL
            """,
            [dataset_type],
        ).fetchall()
    return {int(row[0]) for row in rows}


def collection_runs() -> pd.DataFrame:
    with connect() as con:
        return con.execute("SELECT * FROM collection_runs ORDER BY started_at DESC").df()


def save_model_run(metadata: dict[str, Any]) -> None:
    normalized = {
        column: metadata.get(column)
        for column in MODEL_RUN_COLUMNS
    }
    row = pd.DataFrame([normalized])
    with connect() as con:
        con.register("model_row", row)
        con.execute("DELETE FROM model_runs USING model_row WHERE model_runs.model_run_id = model_row.model_run_id")
        columns = ", ".join(MODEL_RUN_COLUMNS)
        con.execute(
            f"INSERT INTO model_runs ({columns}) "
            f"SELECT {columns} FROM model_row"
        )
        con.unregister("model_row")


def model_runs() -> pd.DataFrame:
    with connect() as con:
        return con.execute("SELECT * FROM model_runs ORDER BY created_at DESC").df()


def save_prediction_run(metadata: dict[str, Any]) -> None:
    row = pd.DataFrame([metadata])
    with connect() as con:
        con.register("prediction_row", row)
        con.execute(
            "DELETE FROM prediction_runs USING prediction_row "
            "WHERE prediction_runs.prediction_run_id = prediction_row.prediction_run_id"
        )
        con.execute("INSERT INTO prediction_runs SELECT * FROM prediction_row")
        con.unregister("prediction_row")


def dashboard_summary() -> dict[str, Any]:
    with connect() as con:
        history_rows = con.execute(
            "SELECT COUNT(*) FROM race_records WHERE dataset_type = 'historical'"
        ).fetchone()[0]
        upcoming_rows = con.execute(
            "SELECT COUNT(*) FROM race_records WHERE dataset_type = 'upcoming'"
        ).fetchone()[0]
        races = con.execute("SELECT COUNT(DISTINCT race_id) FROM race_records").fetchone()[0]
        model_count = con.execute("SELECT COUNT(*) FROM model_runs WHERE status = 'completed'").fetchone()[0]
        latest_date = con.execute("SELECT MAX(race_date) FROM race_records").fetchone()[0]
    return {
        "history_rows": history_rows,
        "upcoming_rows": upcoming_rows,
        "races": races,
        "model_count": model_count,
        "latest_date": latest_date,
    }


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
