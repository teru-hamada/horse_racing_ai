from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import duckdb

from .storage import feature_store_summary, performance_feature_summary


PATHS = import_module("src.00_common.config").PATHS


def source_data_state(database: Path | None = None) -> dict[str, object]:
    """Return a lightweight generation marker for historical source records."""

    path = Path(database or PATHS.database)
    if not path.exists():
        return {
            "row_count": 0,
            "race_count": 0,
            "start_date": None,
            "end_date": None,
            "latest_collected_at": None,
            "latest_collection_run_id": None,
        }
    connection = duckdb.connect(str(path), read_only=True)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT race_id), MIN(race_date), MAX(race_date),
                   MAX(collected_at),
                   arg_max(collection_run_id, collected_at)
            FROM race_records
            WHERE dataset_type = 'historical'
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "row_count": int(row[0]),
        "race_count": int(row[1]),
        "start_date": row[2].isoformat() if row[2] is not None else None,
        "end_date": row[3].isoformat() if row[3] is not None else None,
        "latest_collected_at": row[4].isoformat() if row[4] is not None else None,
        "latest_collection_run_id": row[5],
    }


def source_state_token(state: dict[str, object]) -> str:
    canonical = json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_freshness(
    feature_set_name: str,
    feature_set_version: str,
    *,
    source_database: Path | None = None,
    feature_database: Path | None = None,
) -> dict[str, object]:
    current = source_data_state(source_database)
    current_token = source_state_token(current)
    summary = feature_store_summary(
        feature_set_name, feature_set_version, feature_database
    )
    stored_token = summary.get("source_state_token")
    try:
        stored = json.loads(str(summary.get("source_state_json") or "{}"))
    except json.JSONDecodeError:
        stored = {}

    if int(summary["row_count"]) == 0:
        status = "missing"
    elif not stored_token:
        status = "unknown"
    elif stored_token == current_token:
        status = "fresh"
    else:
        status = "stale"
    differences = {
        key: {"generated": stored.get(key), "current": current.get(key)}
        for key in current
        if stored and stored.get(key) != current.get(key)
    }
    return {
        "status": status,
        "current_state": current,
        "stored_state": stored,
        "current_token": current_token,
        "stored_token": stored_token,
        "differences": differences,
    }


def performance_feature_freshness(
    performance_feature_name: str,
    performance_feature_version: str,
    *,
    source_database: Path | None = None,
    feature_database: Path | None = None,
) -> dict[str, object]:
    current = source_data_state(source_database)
    current_token = source_state_token(current)
    summary = performance_feature_summary(
        performance_feature_name, performance_feature_version, feature_database
    )
    stored_token = summary.get("source_state_token")
    try:
        stored = json.loads(str(summary.get("source_state_json") or "{}"))
    except json.JSONDecodeError:
        stored = {}
    if int(summary["row_count"]) == 0:
        status = "missing"
    elif not stored_token:
        status = "unknown"
    elif stored_token == current_token:
        status = "fresh"
    else:
        status = "stale"
    differences = {
        key: {"generated": stored.get(key), "current": current.get(key)}
        for key in current
        if stored and stored.get(key) != current.get(key)
    }
    return {
        "status": status,
        "current_state": current,
        "stored_state": stored,
        "current_token": current_token,
        "stored_token": stored_token,
        "differences": differences,
    }


def recent_speed_freshness(
    feature_set_name: str = "recent_speed",
    feature_set_version: str = "1.0.0",
    *,
    feature_database: Path | None = None,
) -> dict[str, object]:
    result = feature_freshness(
        feature_set_name, feature_set_version, feature_database=feature_database
    )
    if result["status"] != "fresh":
        return result
    summary = feature_store_summary(
        feature_set_name, feature_set_version, feature_database
    )
    dependency = performance_feature_summary(
        "speed_index", "1.0.0", feature_database
    )
    try:
        config = json.loads(str(summary.get("config_json") or "{}"))
    except json.JSONDecodeError:
        config = {}
    stored_run = config.get("dependency_run_id")
    current_run = dependency.get("latest_run_id")
    if stored_run == current_run:
        return result
    return {
        **result,
        "status": "stale",
        "differences": {
            **result.get("differences", {}),
            "dependency_run_id": {
                "generated": stored_run,
                "current": current_run,
            },
        },
    }
