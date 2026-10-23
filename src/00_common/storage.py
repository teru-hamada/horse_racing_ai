# レース、JRAオッズ、収集・学習・予想の実行履歴をracing.duckdbへ保存する。
# 列順・型を正規化してから登録し、再登録時には対象キーの既存データを置き換える。
# 特徴量の保存先features.duckdbとは別の責務を持つ。database引数があるAPIは検証用DBへ切り替えられる。
#
from __future__ import annotations

import json
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS


# レース表の保存列と順序。正規化とINSERTの列指定で同じ定義を使う。
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

# モデル登録表の保存列。タスク・版・評価指標・成果物パスをそろえる。
MODEL_RUN_COLUMNS = [
    "model_run_id", "created_at", "status", "target",
    "train_rows", "validation_rows", "test_rows", "best_epoch",
    "validation_auc", "test_auc", "test_log_loss", "model_path",
    "metrics_json", "task_name", "estimator_name", "model_version",
    "feature_version",
]

# 券種別オッズの保存列。選択番号とオッズ上下限、収集情報を含む。
ODDS_COLUMNS = [
    "race_id", "race_date", "bet_type", "selection_1", "selection_2",
    "selection_3", "odds_min", "odds_max", "source_html",
    "collection_run_id", "registered_at",
    "fetched_at", "source_url", "source_cname", "source_sha256",
]


# 接続時に実行するテーブル作成SQL。既存テーブルは作り直さない。
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

CREATE TABLE IF NOT EXISTS race_odds (
    race_id VARCHAR,
    race_date DATE,
    bet_type VARCHAR,
    selection_1 INTEGER,
    selection_2 INTEGER,
    selection_3 INTEGER,
    odds_min DOUBLE,
    odds_max DOUBLE,
    source_html VARCHAR,
    collection_run_id VARCHAR,
    registered_at TIMESTAMP
);
"""


# 指定先の親ディレクトリとスキーマを作成して接続する。読み取り専用の接続ではない。
def connect(database: Path | None = None) -> duckdb.DuckDBPyConnection:
    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    database = database if database is not None else PATHS.database
    database.parent.mkdir(parents=True, exist_ok=True)
    from .runtime_lock import connect_locked
    con = connect_locked(PATHS.root, database, duckdb.connect)
    try:
        con.execute(SCHEMA_SQL)
        # 既存DBも、登録日時とは別に元HTMLの取得情報を保持できるよう移行する。
        for column in ("fetched_at", "source_url", "source_cname", "source_sha256"):
            con.execute(f"ALTER TABLE race_odds ADD COLUMN IF NOT EXISTS {column} VARCHAR")
    except BaseException:
        con.close()
        raise
    return con


# 保存先と同じ列名を持つ空DataFrameを返し、データなしでも列構成を安定させる。
def empty_race_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RACE_RECORD_COLUMNS)


# 不足列は欠損で補い、余分な列を除いて保存列順へ並べる。不正な数値・日付は欠損へ変換する。
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


# 同じrun_idの収集履歴を置き換え、実行状態や件数を最新の内容に更新する。
def save_collection_run(metadata: dict[str, Any]) -> None:
    row = pd.DataFrame([metadata])
    with connect() as con:
        con.register("run_row", row)
        con.execute("DELETE FROM collection_runs USING run_row WHERE collection_runs.run_id = run_row.run_id")
        con.execute("INSERT INTO collection_runs SELECT * FROM run_row")
        con.unregister("run_row")


# 同じレース・馬・データ種別の行を削除して再登録する。別のデータ種別の行は保持する。
def save_race_records(
    df: pd.DataFrame,
    run_id: str,
    dataset_type: str,
    *, database: Path | None = None,
) -> None:
    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    if df.empty:
        raise ValueError("保存対象のレースデータが0件です。")

    normalized = normalize_race_frame(df)
    normalized["dataset_type"] = dataset_type
    normalized["collection_run_id"] = run_id
    normalized["collected_at"] = datetime.now()

    with connect(database) as con:
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


# 任意のデータ種別で絞り込み、日付・レースID・馬番の順にレース行を読み出す。
def load_records(dataset_type: str | None = None, *, database: Path | None = None) -> pd.DataFrame:
    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    with connect(database) as con:
        if dataset_type:
            return con.execute(
                "SELECT * FROM race_records WHERE dataset_type = ? ORDER BY race_date, race_id, horse_number",
                [dataset_type],
            ).df()
        return con.execute("SELECT * FROM race_records ORDER BY race_date, race_id, horse_number").df()


# 入力に含まれるレースの既存オッズを全券種まとめて置換するため、対象レースの一式を渡す。
def save_race_odds(df: pd.DataFrame, run_id: str, *, database: Path | None = None) -> None:
    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    if df.empty:
        return
    normalized = df.copy()
    for column in ODDS_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["race_date"] = pd.to_datetime(
        normalized["race_date"], errors="coerce"
    ).dt.date
    for column in ("selection_1", "selection_2", "selection_3", "odds_min", "odds_max"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["collection_run_id"] = run_id
    normalized["registered_at"] = datetime.now()
    normalized = normalized[ODDS_COLUMNS]
    with connect(database) as con:
        con.register("incoming_odds", normalized)
        con.execute(
            """
            DELETE FROM race_odds
            USING incoming_odds
            WHERE race_odds.race_id = incoming_odds.race_id
            """
        )
        columns = ", ".join(ODDS_COLUMNS)
        con.execute(
            f"INSERT INTO race_odds ({columns}) SELECT {columns} FROM incoming_odds"
        )
        con.unregister("incoming_odds")


# race_idsの指定を優先し、なければrace_date、両方なければ全件を読み出す。
def load_race_odds(
    race_ids: list[str] | None = None,
    race_date: Any | None = None,
    *, database: Path | None = None,
) -> pd.DataFrame:
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    # database: 読み書き対象のDBパス。テストでは専用DBへ切り替える。
    with connect(database) as con:
        # レースIDが指定された場合は、開催日指定より優先してその集合に絞る。
        if race_ids:
            placeholders = ", ".join("?" for _ in race_ids)
            return con.execute(
                f"SELECT * FROM race_odds WHERE race_id IN ({placeholders}) "
                "ORDER BY race_id, bet_type, selection_1, selection_2, selection_3",
                [str(value) for value in race_ids],
            ).df()
        # ID指定がないとき、日付が渡されていれば対象日のオッズだけを読み出す。
        if race_date is not None:
            return con.execute(
                "SELECT * FROM race_odds WHERE race_date = ? "
                "ORDER BY race_id, bet_type, selection_1, selection_2, selection_3",
                [race_date],
            ).df()
        return con.execute(
            "SELECT * FROM race_odds ORDER BY race_date, race_id, bet_type"
        ).df()


# 指定種別・期間の行数とレース数をDBで集計する。頭数とレース数を区別する。
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


# 指定種別のレースに存在する開催年の集合を返し、画面の期間選択などに使う。
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


# 収集実行履歴を開始日時の降順で読み出す。
def collection_runs() -> pd.DataFrame:
    with connect() as con:
        return con.execute("SELECT * FROM collection_runs ORDER BY started_at DESC").df()


# 既知のモデル登録列へそろえ、同じ実行IDの登録を置き換える。未指定項目はNoneになる。
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


# モデル登録履歴を作成日時の降順で読み出し、画面で最新モデルから選べるようにする。
def model_runs() -> pd.DataFrame:
    with connect() as con:
        return con.execute("SELECT * FROM model_runs ORDER BY created_at DESC").df()


# 予想実行IDに対応する履歴を置き換え、使用モデルと出力先を保存する。
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


# ダッシュボードに使う過去・出馬表の行数、レース数、完了モデル数、最新日を集計する。
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


# 日本語をそのままUTF-8で出力する。日時など標準JSON外の値は文字列へ変換する。
def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
