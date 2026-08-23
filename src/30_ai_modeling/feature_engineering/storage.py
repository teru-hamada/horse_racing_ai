from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


PATHS = import_module("src.00_common.config").PATHS
FEATURE_KEY_COLUMNS = [
    "race_id", "horse_id", "race_date", "feature_set_name",
    "feature_set_version", "history_cutoff", "feature_run_id", "generated_at",
]
_SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feature_runs (
    feature_run_id VARCHAR PRIMARY KEY,
    feature_set_name VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    source_data_fingerprint VARCHAR,
    generated_at TIMESTAMP NOT NULL,
    start_date DATE,
    end_date DATE,
    row_count BIGINT,
    config_json VARCHAR,
    status VARCHAR NOT NULL,
    message VARCHAR
);

CREATE TABLE IF NOT EXISTS race_features (
    race_id VARCHAR NOT NULL,
    horse_id VARCHAR NOT NULL,
    race_date DATE NOT NULL,
    feature_set_name VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    history_cutoff TIMESTAMP NOT NULL,
    feature_run_id VARCHAR NOT NULL,
    generated_at TIMESTAMP NOT NULL
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    database = Path(path or PATHS.feature_database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute(SCHEMA_SQL)
    return connection


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "TIMESTAMP"
    return "VARCHAR"


def save_feature_run(metadata: dict[str, Any], path: Path | None = None) -> None:
    row = pd.DataFrame([{
        "feature_run_id": metadata["feature_run_id"],
        "feature_set_name": metadata["feature_set_name"],
        "feature_set_version": metadata["feature_set_version"],
        "source_data_fingerprint": metadata.get("source_data_fingerprint"),
        "generated_at": metadata.get("generated_at", datetime.now()),
        "start_date": metadata.get("start_date"),
        "end_date": metadata.get("end_date"),
        "row_count": metadata.get("row_count", 0),
        "config_json": json.dumps(metadata.get("config", {}), ensure_ascii=False, default=str),
        "status": metadata["status"],
        "message": metadata.get("message", ""),
    }])
    with connect(path) as con:
        con.register("incoming_run", row)
        con.execute(
            "DELETE FROM feature_runs USING incoming_run "
            "WHERE feature_runs.feature_run_id = incoming_run.feature_run_id"
        )
        con.execute("INSERT INTO feature_runs BY NAME SELECT * FROM incoming_run")


def save_features(frame: pd.DataFrame, path: Path | None = None) -> None:
    missing = set(FEATURE_KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature frame is missing metadata columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature frame is empty")
    if frame.duplicated(["race_id", "horse_id", "feature_set_name", "feature_set_version"]).any():
        raise ValueError("Feature frame contains duplicate race/horse/version rows")

    feature_columns = [column for column in frame.columns if column not in FEATURE_KEY_COLUMNS]
    invalid = [column for column in feature_columns if not _SAFE_COLUMN.fullmatch(column)]
    if invalid:
        raise ValueError(f"Unsafe feature column names: {invalid}")

    with connect(path) as con:
        existing = {
            row[1] for row in con.execute("PRAGMA table_info('race_features')").fetchall()
        }
        for column in feature_columns:
            if column not in existing:
                con.execute(f'ALTER TABLE race_features ADD COLUMN "{column}" {_sql_type(frame[column])}')
        con.register("incoming_features", frame)
        con.execute(
            "DELETE FROM race_features USING incoming_features "
            "WHERE race_features.race_id = incoming_features.race_id "
            "AND race_features.horse_id = incoming_features.horse_id "
            "AND race_features.feature_set_name = incoming_features.feature_set_name "
            "AND race_features.feature_set_version = incoming_features.feature_set_version"
        )
        columns = list(frame.columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        con.execute(f"INSERT INTO race_features ({quoted}) SELECT {quoted} FROM incoming_features")


def load_features(
    feature_set_name: str,
    feature_set_version: str,
    path: Path | None = None,
) -> pd.DataFrame:
    with connect(path) as con:
        return con.execute(
            "SELECT * FROM race_features "
            "WHERE feature_set_name = ? AND feature_set_version = ? "
            "ORDER BY race_date, race_id, horse_id",
            [feature_set_name, feature_set_version],
        ).df()


def clear_features(
    feature_set_name: str | None = None,
    feature_set_version: str | None = None,
    path: Path | None = None,
) -> int:
    if feature_set_version is not None and feature_set_name is None:
        raise ValueError("feature_set_name is required when clearing one version")
    conditions: list[str] = []
    parameters: list[str] = []
    if feature_set_name is not None:
        conditions.append("feature_set_name = ?")
        parameters.append(feature_set_name)
    if feature_set_version is not None:
        conditions.append("feature_set_version = ?")
        parameters.append(feature_set_version)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    with connect(path) as con:
        count = int(con.execute(f"SELECT COUNT(*) FROM race_features{where}", parameters).fetchone()[0])
        con.execute(f"DELETE FROM race_features{where}", parameters)
        return count


def feature_runs(path: Path | None = None) -> pd.DataFrame:
    with connect(path) as con:
        return con.execute("SELECT * FROM feature_runs ORDER BY generated_at DESC").df()
