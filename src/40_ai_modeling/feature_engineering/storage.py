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
PERFORMANCE_KEY_COLUMNS = [
    "race_id", "horse_id", "race_date", "performance_feature_name",
    "performance_feature_version", "history_cutoff", "feature_run_id", "generated_at",
]
_SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feature_runs (
    feature_run_id VARCHAR PRIMARY KEY,
    feature_set_name VARCHAR NOT NULL,
    feature_set_version VARCHAR NOT NULL,
    source_data_fingerprint VARCHAR,
    source_state_token VARCHAR,
    source_state_json VARCHAR,
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

CREATE TABLE IF NOT EXISTS performance_features (
    race_id VARCHAR NOT NULL,
    horse_id VARCHAR NOT NULL,
    race_date DATE NOT NULL,
    performance_feature_name VARCHAR NOT NULL,
    performance_feature_version VARCHAR NOT NULL,
    history_cutoff TIMESTAMP NOT NULL,
    feature_run_id VARCHAR NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    speed_index DOUBLE,
    speed_index_raw DOUBLE,
    speed_index_was_clipped BOOLEAN,
    speed_standard_time DOUBLE,
    speed_time_difference DOUBLE,
    speed_weight_adjustment DOUBLE,
    speed_track_adjustment DOUBLE,
    speed_reference_race_count DOUBLE,
    speed_reference_effective_count DOUBLE,
    speed_fallback_level VARCHAR
);
"""


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    database = Path(path or PATHS.feature_database)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute(SCHEMA_SQL)
    connection.execute(
        "ALTER TABLE feature_runs ADD COLUMN IF NOT EXISTS source_state_token VARCHAR"
    )
    connection.execute(
        "ALTER TABLE feature_runs ADD COLUMN IF NOT EXISTS source_state_json VARCHAR"
    )
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
        "source_state_token": metadata.get("source_state_token"),
        "source_state_json": json.dumps(
            metadata.get("source_state", {}), ensure_ascii=False, default=str
        ),
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


def _validate_feature_frame(frame: pd.DataFrame) -> list[str]:
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

    return feature_columns


def _write_features(
    frame: pd.DataFrame,
    path: Path | None,
    *,
    replace_version: bool,
) -> None:
    feature_columns = _validate_feature_frame(frame)
    with connect(path) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            if replace_version:
                identities = frame[["feature_set_name", "feature_set_version"]].drop_duplicates()
                if len(identities) != 1:
                    raise ValueError("Replacement requires exactly one feature set and version")
                identity = identities.iloc[0]
                con.execute(
                    "DELETE FROM race_features "
                    "WHERE feature_set_name = ? AND feature_set_version = ?",
                    [identity["feature_set_name"], identity["feature_set_version"]],
                )
            existing = {
                row[1]
                for row in con.execute("PRAGMA table_info('race_features')").fetchall()
            }
            for column in feature_columns:
                if column not in existing:
                    con.execute(
                        f'ALTER TABLE race_features ADD COLUMN "{column}" '
                        f'{_sql_type(frame[column])}'
                    )
            con.register("incoming_features", frame)
            if not replace_version:
                con.execute(
                    "DELETE FROM race_features USING incoming_features "
                    "WHERE race_features.race_id = incoming_features.race_id "
                    "AND race_features.horse_id = incoming_features.horse_id "
                    "AND race_features.feature_set_name = incoming_features.feature_set_name "
                    "AND race_features.feature_set_version = incoming_features.feature_set_version"
                )
            columns = list(frame.columns)
            quoted = ", ".join(f'"{column}"' for column in columns)
            con.execute(
                f"INSERT INTO race_features ({quoted}) "
                f"SELECT {quoted} FROM incoming_features"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise


def save_features(frame: pd.DataFrame, path: Path | None = None) -> None:
    _write_features(frame, path, replace_version=False)


def replace_features(frame: pd.DataFrame, path: Path | None = None) -> None:
    """Atomically replace every row for one feature-set version."""

    _write_features(frame, path, replace_version=True)


def replace_performance_features(
    frame: pd.DataFrame,
    path: Path | None = None,
) -> None:
    """Atomically replace one version of per-performance features."""

    missing = set(PERFORMANCE_KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Performance frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Performance feature frame is empty")
    identities = frame[
        ["performance_feature_name", "performance_feature_version"]
    ].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("Replacement requires one performance feature and version")
    if frame.duplicated(
        ["race_id", "horse_id", "performance_feature_name", "performance_feature_version"]
    ).any():
        raise ValueError("Performance feature frame contains duplicate keys")
    feature_columns = [
        column for column in frame.columns if column not in PERFORMANCE_KEY_COLUMNS
    ]
    invalid = [column for column in feature_columns if not _SAFE_COLUMN.fullmatch(column)]
    if invalid:
        raise ValueError(f"Unsafe performance feature columns: {invalid}")

    identity = identities.iloc[0]
    with connect(path) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            existing = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info('performance_features')"
                ).fetchall()
            }
            for column in feature_columns:
                if column not in existing:
                    con.execute(
                        f'ALTER TABLE performance_features ADD COLUMN "{column}" '
                        f'{_sql_type(frame[column])}'
                    )
            con.execute(
                "DELETE FROM performance_features "
                "WHERE performance_feature_name = ? "
                "AND performance_feature_version = ?",
                [
                    identity["performance_feature_name"],
                    identity["performance_feature_version"],
                ],
            )
            con.register("incoming_performance", frame)
            columns = list(frame.columns)
            quoted = ", ".join(f'"{column}"' for column in columns)
            con.execute(
                f"INSERT INTO performance_features ({quoted}) "
                f"SELECT {quoted} FROM incoming_performance"
            )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise


def load_performance_features(
    performance_feature_name: str,
    performance_feature_version: str,
    path: Path | None = None,
) -> pd.DataFrame:
    with connect(path) as con:
        return con.execute(
            "SELECT * FROM performance_features "
            "WHERE performance_feature_name = ? AND performance_feature_version = ? "
            "ORDER BY race_date, race_id, horse_id",
            [performance_feature_name, performance_feature_version],
        ).df()


def clear_performance_features(
    performance_feature_name: str,
    performance_feature_version: str,
    path: Path | None = None,
) -> int:
    with connect(path) as con:
        count = int(con.execute(
            "SELECT COUNT(*) FROM performance_features "
            "WHERE performance_feature_name = ? AND performance_feature_version = ?",
            [performance_feature_name, performance_feature_version],
        ).fetchone()[0])
        con.execute(
            "DELETE FROM performance_features "
            "WHERE performance_feature_name = ? AND performance_feature_version = ?",
            [performance_feature_name, performance_feature_version],
        )
        return count


def performance_feature_summary(
    performance_feature_name: str,
    performance_feature_version: str,
    path: Path | None = None,
) -> dict[str, object]:
    with connect(path) as con:
        row = con.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT race_id), MIN(race_date), MAX(race_date),
                   MAX(generated_at), COUNT(speed_index),
                   COUNT(*) FILTER (WHERE speed_index_was_clipped)
            FROM performance_features
            WHERE performance_feature_name = ? AND performance_feature_version = ?
            """,
            [performance_feature_name, performance_feature_version],
        ).fetchone()
        latest = con.execute(
            """
            SELECT feature_run_id, source_data_fingerprint, source_state_token,
                   source_state_json, config_json
            FROM feature_runs
            WHERE feature_set_name = ? AND feature_set_version = ?
              AND status = 'completed'
            ORDER BY generated_at DESC LIMIT 1
            """,
            [performance_feature_name, performance_feature_version],
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "race_count": int(row[1]),
        "start_date": row[2],
        "end_date": row[3],
        "generated_at": row[4],
        "available_count": int(row[5]),
        "clipped_count": int(row[6]),
        "latest_run_id": latest[0] if latest else None,
        "source_data_fingerprint": latest[1] if latest else None,
        "source_state_token": latest[2] if latest else None,
        "source_state_json": latest[3] if latest else None,
        "config_json": latest[4] if latest else None,
    }


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


def feature_store_summary(
    feature_set_name: str,
    feature_set_version: str,
    path: Path | None = None,
) -> dict[str, object]:
    with connect(path) as con:
        row = con.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT race_id), MIN(race_date), MAX(race_date),
                   MAX(generated_at)
            FROM race_features
            WHERE feature_set_name = ? AND feature_set_version = ?
            """,
            [feature_set_name, feature_set_version],
        ).fetchone()
        latest = con.execute(
            """
            SELECT feature_run_id, status, source_data_fingerprint, config_json, message,
                   source_state_token, source_state_json
            FROM feature_runs
            WHERE feature_set_name = ? AND feature_set_version = ?
              AND status = 'completed'
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            [feature_set_name, feature_set_version],
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "race_count": int(row[1]),
        "start_date": row[2],
        "end_date": row[3],
        "generated_at": row[4],
        "latest_run_id": latest[0] if latest else None,
        "latest_status": latest[1] if latest else None,
        "source_data_fingerprint": latest[2] if latest else None,
        "config_json": latest[3] if latest else None,
        "message": latest[4] if latest else None,
        "source_state_token": latest[5] if latest else None,
        "source_state_json": latest[6] if latest else None,
    }
