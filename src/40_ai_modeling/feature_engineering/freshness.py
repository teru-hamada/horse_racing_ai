# 特徴量生成時と現在の元DBの状態を比較し、再生成の必要性を判定する。
# 件数・期間・最終収集情報から軽量なトークンを作り、missing／unknown／fresh／staleを区別する。
# recent_speedでは元データに加え、依存するspeed_indexの生成runが変わっていないかも確認する。
#
from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import duckdb

from .storage import feature_store_summary, performance_feature_summary


# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS


# 全データの内容ハッシュではなく、件数・期間・最終収集情報を使う軽量な変更検出用の状態を取得する。
def source_data_state(database: Path | None = None) -> dict[str, object]:
    """Return a lightweight generation marker for historical source records."""

    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    path = Path(database or PATHS.database)
    # 対象ファイルが未生成のケースを扱い、別のファイルを推測して読み込まない。
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


# 辞書キー順を固定したJSONをハッシュ化し、同じ状態から常に同じトークンを作る。
def source_state_token(state: dict[str, object]) -> str:
    canonical = json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# 行なしはmissing、生成元情報なしはunknown、一致ならfresh、差があればstaleとして返す。
def feature_freshness(
    feature_set_name: str,
    feature_set_version: str,
    *,
    source_database: Path | None = None,
    feature_database: Path | None = None,
) -> dict[str, object]:
    # source_database: 鮮度確認などで参照する元レースDBの指定先。
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    current = source_data_state(source_database)
    # 現在の元データ状態から作った比較用ハッシュ。
    current_token = source_state_token(current)
    summary = feature_store_summary(
        feature_set_name, feature_set_version, feature_database
    )
    # 特徴量の生成履歴に記録された、生成当時の元データ状態ハッシュ。
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
    # 生成時と現在、または再現元と今回の違いを項目名ごとに保持する。
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


# 1走単位の特徴量について、元DBの現在状態と生成時の状態を照合する。
def performance_feature_freshness(
    performance_feature_name: str,
    performance_feature_version: str,
    *,
    source_database: Path | None = None,
    feature_database: Path | None = None,
) -> dict[str, object]:
    # source_database: 鮮度確認などで参照する元レースDBの指定先。
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    current = source_data_state(source_database)
    # 現在の元データ状態から作った比較用ハッシュ。
    current_token = source_state_token(current)
    summary = performance_feature_summary(
        performance_feature_name, performance_feature_version, feature_database
    )
    # 特徴量の生成履歴に記録された、生成当時の元データ状態ハッシュ。
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
    # 生成時と現在、または再現元と今回の違いを項目名ごとに保持する。
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


# 元DBが同じでも速度指数の生成runが変わっていれば、依存先を使い直す必要があるためstaleとする。
def recent_speed_freshness(
    feature_set_name: str = "recent_speed",
    feature_set_version: str = "1.0.0",
    *,
    feature_database: Path | None = None,
) -> dict[str, object]:
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
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
