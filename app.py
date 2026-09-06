from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.public_api import (
    AppLogger,
    collection_runs,
    dashboard_summary,
    load_records,
    model_runs,
    PATHS,
    build_prediction_site,
    latest_prediction_file,
    prediction_date_status,
    compare_prediction_date,
    predict_historical_race,
    predict_race,
    predict_race_date,
    race_record_summary,
    race_record_years,
    cancel_database_job,
    cancel_feature_generation_job,
    cancel_speed_index_job,
    cancel_recent_speed_job,
    cancel_race_entry_job,
    cancel_html_collection_job as cancel_job,
    generate_demo_records,
    feature_runs,
    feature_freshness,
    feature_store_summary,
    fetch_jravan_weather,
    performance_feature_freshness,
    performance_feature_summary,
    recent_speed_freshness,
    clear_features,
    clear_performance_features,
    list_database_jobs,
    list_feature_generation_jobs,
    list_speed_index_jobs,
    list_recent_speed_jobs,
    list_race_entry_jobs,
    list_html_collection_jobs as list_jobs,
    NetkeibaHtmlCollector,
    NetkeibaDatabaseCreator,
    start_database_job,
    start_feature_generation_job,
    start_speed_index_job,
    start_recent_speed_job,
    start_race_entry_job,
    start_html_collection_job as start_job,
    TrainConfig,
    RecentFormRunConfig,
    SpeedIndexRunConfig,
    RecentSpeedRunConfig,
    RaceEntryRunConfig,
    train_model,
    save_collection_run,
    save_race_records,
)


st.set_page_config(page_title="競馬予想AI", page_icon="🏇", layout="wide")
st.title("🏇 競馬予想AIシステム")
st.caption("私的利用向けMVP：データ収集・保存・学習・過学習確認・レース予想を個別実行できます。")
st.markdown(
    """
    <style>
    div.stButton > button,
    div.stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: #1677ff !important;
        border-color: #1677ff !important;
        color: white !important;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0958d9 !important;
        border-color: #0958d9 !important;
        color: white !important;
    }
    div.stButton > button:disabled,
    div.stDownloadButton > button:disabled,
    [data-testid="stFormSubmitButton"] > button:disabled {
        background-color: #1677ff !important;
        border-color: #1677ff !important;
        color: white !important;
        opacity: 0.45;
        cursor: not-allowed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []

if "scraping_urls" not in st.session_state:
    st.session_state.scraping_urls = []


_URL_PATTERN = re.compile(r"https?://[^\s<>'\"\]\)]+", re.IGNORECASE)

_PREDICTION_COLUMN_LABELS = {
    "prediction_rank": "予測順位",
    "race_id": "レースID",
    "race_date": "開催日",
    "course_name": "競馬場",
    "race_number": "レース番号",
    "race_name": "レース名",
    "horse_number": "馬番",
    "horse_id": "競走馬ID",
    "horse_name": "馬名",
    "jockey_name": "騎手",
    "odds": "オッズ",
    "popularity": "人気",
    "top3_probability": "3着以内確率",
    "expected_value_index": "期待値指数",
    "finish_position": "実着順",
    "predicted_top3": "予測上位3頭",
    "actual_top3": "実際の3着以内",
    "top3_hit": "的中",
}


@st.cache_data(ttl=600, show_spinner=False)
def _cached_jravan_weather(
    course_name: str,
    target_date: date,
) -> dict[str, object]:
    """Avoid repeated JRA-VAN requests during Streamlit reruns."""

    return fetch_jravan_weather(course_name, target_date)


def _extract_urls(value: object) -> list[str]:
    """文字列やコレクションからHTTP(S) URLを重複なく抽出する。"""
    if value is None:
        return []

    if isinstance(value, str):
        candidates = _URL_PATTERN.findall(value)
    elif isinstance(value, dict):
        candidates = []
        for item in value.values():
            candidates.extend(_extract_urls(item))
    elif isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            candidates.extend(_extract_urls(item))
    else:
        candidates = _URL_PATTERN.findall(str(value))

    cleaned: list[str] = []
    for url in candidates:
        normalized = url.rstrip(".,;:!?、。】」』")
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _append_scraping_urls(urls: list[str]) -> None:
    """セッション内の取得URL一覧へ、順序を維持して追加する。"""
    current = list(st.session_state.scraping_urls)
    for url in urls:
        if url not in current:
            current.append(url)
    st.session_state.scraping_urls = current[-1000:]


def _frame_source_urls(frame: pd.DataFrame | None) -> list[str]:
    """取得結果DataFrameに含まれるURL列からURLを抽出する。"""
    if frame is None or frame.empty:
        return []

    urls: list[str] = []
    candidate_columns = [
        "source_url",
        "race_url",
        "url",
        "page_url",
        "request_url",
    ]
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna().astype(str).tolist():
            for url in _extract_urls(value):
                if url not in urls:
                    urls.append(url)
    return urls


def _scraper_debug_urls(scraper: object) -> list[str]:
    """スクレイパーが保持している代表的な属性からURLを回収する。"""
    urls: list[str] = []
    candidate_attributes = [
        "current_url",
        "last_url",
        "requested_url",
        "requested_urls",
        "visited_urls",
        "race_urls",
        "source_urls",
        "diagnostics",
        "debug_info",
    ]
    for name in candidate_attributes:
        if not hasattr(scraper, name):
            continue
        try:
            value = getattr(scraper, name)
        except Exception:
            continue
        for url in _extract_urls(value):
            if url not in urls:
                urls.append(url)
    return urls


def ui_log_callback(line: str) -> None:
    st.session_state.ui_logs.append(line)
    if len(st.session_state.ui_logs) > 300:
        st.session_state.ui_logs = st.session_state.ui_logs[-300:]
    if log_placeholder is not None:
        log_placeholder.code(
            "\n".join(st.session_state.ui_logs[-80:]),
            language="text",
        )


def new_logger(callback=None) -> AppLogger:
    return AppLogger(PATHS.logs, callback=callback or ui_log_callback)


def _quote_identifier(name: str) -> str:
    """DuckDBの識別子を安全にクォートする。"""
    return '"' + str(name).replace('"', '""') + '"'


def _database_tables(connection: duckdb.DuckDBPyConnection) -> list[str]:
    try:
        return [str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()]
    except Exception:
        return []


def _table_columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> set[str]:
    try:
        rows = connection.execute(
            f"PRAGMA table_info({_quote_identifier(table_name)})"
        ).fetchall()
        return {str(row[1]) for row in rows}
    except Exception:
        return set()


def _delete_database_rows(
    column_name: str,
    values: list[str],
    *,
    prefix_match: bool = False,
) -> dict[str, int]:
    """
    指定列を持つ全テーブルから一致行を削除する。

    テーブル構造が異なる環境でも動作するよう、対象列が存在する
    テーブルだけを処理する。
    """
    result: dict[str, int] = {}
    if not values or not Path(PATHS.database).exists():
        return result

    connection = duckdb.connect(str(PATHS.database))
    try:
        for table_name in _database_tables(connection):
            columns = _table_columns(connection, table_name)
            if column_name not in columns:
                continue

            table_sql = _quote_identifier(table_name)
            column_sql = _quote_identifier(column_name)
            deleted = 0

            for value in values:
                if prefix_match:
                    before = connection.execute(
                        f"SELECT COUNT(*) FROM {table_sql} "
                        f"WHERE CAST({column_sql} AS VARCHAR) LIKE ?",
                        [f"{value}%"],
                    ).fetchone()[0]
                    connection.execute(
                        f"DELETE FROM {table_sql} "
                        f"WHERE CAST({column_sql} AS VARCHAR) LIKE ?",
                        [f"{value}%"],
                    )
                else:
                    before = connection.execute(
                        f"SELECT COUNT(*) FROM {table_sql} "
                        f"WHERE CAST({column_sql} AS VARCHAR) = ?",
                        [str(value)],
                    ).fetchone()[0]
                    connection.execute(
                        f"DELETE FROM {table_sql} "
                        f"WHERE CAST({column_sql} AS VARCHAR) = ?",
                        [str(value)],
                    )
                deleted += int(before)

            if deleted:
                result[table_name] = deleted
    finally:
        connection.close()

    return result


def _split_output_paths(value: object) -> list[Path]:
    """収集履歴のoutput_pathから複数ファイルパスを取り出す。"""
    if value is None or pd.isna(value):
        return []

    paths: list[Path] = []
    for part in str(value).split(";"):
        cleaned = part.strip()
        if cleaned:
            paths.append(Path(cleaned))
    return paths


def _read_race_ids_from_files(paths: list[Path]) -> list[str]:
    """削除対象ファイルからrace_idを回収する。"""
    race_ids: set[str] = set()

    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                frame = pd.read_parquet(path, columns=["race_id"])
            elif suffix == ".csv":
                frame = pd.read_csv(
                    path,
                    usecols=lambda column: str(column) == "race_id",
                )
            else:
                continue

            if "race_id" in frame.columns:
                race_ids.update(
                    frame["race_id"]
                    .dropna()
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .tolist()
                )
        except Exception:
            continue

    return sorted(race_ids)


def _related_output_files(path: Path) -> list[Path]:
    """Parquet/CSVなど同一名称の関連ファイルも削除対象にする。"""
    candidates = {path}

    if path.suffix:
        for suffix in [".parquet", ".csv", ".json"]:
            candidates.add(path.with_suffix(suffix))

    return sorted(candidates, key=lambda item: str(item))


def _remove_paths(paths: list[Path]) -> tuple[list[str], list[str]]:
    """ファイルまたはフォルダを削除し、成功・失敗を返す。"""
    deleted: list[str] = []
    failed: list[str] = []

    unique_paths = list(dict.fromkeys(paths))
    for path in unique_paths:
        try:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(str(path))
        except Exception as exc:
            failed.append(f"{path}: {exc}")

    return deleted, failed


def _delete_collection_run(
    selected_run: pd.Series,
) -> dict[str, object]:
    """収集実行単位で出力ファイルとDB行を削除する。取得HTMLは削除しない。"""
    run_id = str(selected_run["run_id"])
    output_paths = _split_output_paths(selected_run.get("output_path"))

    # 出力ファイルを消す前にrace_idを回収する。
    race_ids = _read_race_ids_from_files(output_paths)

    paths_to_delete: list[Path] = []
    for output_path in output_paths:
        paths_to_delete.extend(_related_output_files(output_path))

    raw_run_dir = Path(PATHS.raw_html) / run_id
    # 取得HTMLは「取得HTML」区分からのみ削除する。

    deleted_paths, failed_paths = _remove_paths(paths_to_delete)

    database_result: dict[str, dict[str, int]] = {}

    # レースデータはrace_idを基準に削除する。
    if race_ids:
        race_delete_result = _delete_database_rows(
            "race_id",
            race_ids,
        )
        if race_delete_result:
            database_result["race_id"] = race_delete_result

    # 実行履歴・run_id付きデータを削除する。
    for column in [
        "run_id",
        "collection_run_id",
        "source_run_id",
    ]:
        result = _delete_database_rows(
            column,
            [run_id],
            prefix_match=True,
        )
        if result:
            database_result[column] = result

    return {
        "run_id": run_id,
        "html_path": str(raw_run_dir),
        "html_deleted": False,
        "race_ids": race_ids,
        "deleted_paths": deleted_paths,
        "failed_paths": failed_paths,
        "database_result": database_result,
    }


def _delete_model_run(selected_model: pd.Series) -> dict[str, object]:
    """学習結果をモデル実行単位で削除する。"""
    model_run_id = str(selected_model["model_run_id"])
    model_path = Path(str(selected_model.get("model_path", "")))

    deleted_paths, failed_paths = _remove_paths(
        [model_path] if str(model_path) not in {"", "."} else []
    )

    database_result: dict[str, dict[str, int]] = {}
    for column in ["model_run_id", "run_id"]:
        result = _delete_database_rows(
            column,
            [model_run_id],
            prefix_match=False,
        )
        if result:
            database_result[column] = result

    return {
        "model_run_id": model_run_id,
        "deleted_paths": deleted_paths,
        "failed_paths": failed_paths,
        "database_result": database_result,
    }


def _clear_ui_state() -> None:
    st.session_state.ui_logs = []
    st.session_state.scraping_urls = []
    st.cache_data.clear()


def _cached_race_html_count(
    dataset_type: str,
    start_date: date,
    end_date: date,
) -> int:
    if dataset_type == "historical":
        return sum(
            1
            for year in range(
                start_date.year,
                end_date.year + 1,
            )
            for path in (
                PATHS.historical_html / str(year) / "result"
            ).glob("*.html")
            if path.is_file()
        )

    race_ids: set[str] = set()
    current_date = start_date
    while current_date <= end_date:
        date_text = current_date.strftime("%Y%m%d")
        list_path = (
            PATHS.upcoming_html
            / str(current_date.year)
            / "race_list_db"
            / f"{date_text}.html"
        )
        if list_path.exists():
            html = list_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            race_ids.update(
                NetkeibaHtmlCollector._extract_race_ids(
                    html,
                    date_text,
                )
            )
        current_date += timedelta(days=1)

    return sum(
        1
        for race_id in race_ids
        if any(
            (
                PATHS.upcoming_html
                / race_id[:4]
                / "card"
            ).glob(f"{race_id}*.html")
        )
    )


def _available_upcoming_html_dates() -> list[date]:
    """Return dates having both a cached race list and at least one race card."""

    available: list[date] = []
    for list_path in PATHS.upcoming_html.glob("*/race_list_db/*.html"):
        try:
            target_date = datetime.strptime(list_path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if _cached_race_html_count("upcoming", target_date, target_date) > 0:
            available.append(target_date)
    return sorted(set(available), reverse=True)


def _render_collection_status(dataset_type: str) -> bool:
    st.subheader("実行状況")
    jobs = list_jobs(dataset_type)
    if not jobs:
        st.info(
            "この画面から開始したHTML収集はありません。"
        )
        return False

    latest_job = jobs[0]
    status_labels = {
        "running": "実行中",
        "cancelling": "中止処理中",
        "cancelled": "中止",
        "completed": "完了",
        "failed": "失敗",
    }
    st.progress(
        float(latest_job["progress"]),
        text=(
            f"{status_labels.get(latest_job['status'], latest_job['status'])} "
            f"{float(latest_job['progress']):.0%}"
        ),
    )
    st.write(
        f"実行ID: `{latest_job['job_id']}`  "
        f"対象: {latest_job['start_date']} ～ "
        f"{latest_job['end_date']}"
    )
    if latest_job["result"]:
        st.write(
            f"確認日数: "
            f"{latest_job['result'].get('date_count', 0):,} / "
            f"取得レースHTML: "
            f"{latest_job['result'].get('race_count', 0):,}"
            + (
                " / 取得競走馬HTML: "
                f"{latest_job['result'].get('horse_count', 0):,}"
                if "horse_count" in latest_job["result"]
                else ""
            )
        )
    if latest_job["error"]:
        st.error(latest_job["error"])
    with st.expander(
        "実行ログ",
        expanded=latest_job["status"] == "running",
    ):
        st.code(
            "\n".join(latest_job["logs"][-100:]),
            language="text",
        )
    if st.button(
        "状況を更新",
        key="refresh_html_collection",
    ):
        st.rerun()
    return latest_job["status"] in {
        "running",
        "cancelling",
    }


@st.fragment(run_every=3)
def _render_active_collection_status(
    dataset_type: str,
) -> None:
    if not _render_collection_status(dataset_type):
        # 完了・中止・失敗時は画面全体を1回更新し、
        # 開始・中止ボタンの活性状態も最新化する。
        st.rerun()


def _render_database_creation_status(
    dataset_type: str | None = None,
) -> bool:
    st.subheader("データベース作成状況")
    jobs = list_database_jobs(dataset_type)
    if not jobs:
        st.info(
            "この画面から開始したデータベース作成は"
            "ありません。"
        )
        return False

    latest_job = jobs[0]
    status_labels = {
        "running": "実行中",
        "cancelling": "中止処理中",
        "cancelled": "中止",
        "completed": "完了",
        "failed": "失敗",
    }
    st.progress(
        float(latest_job["progress"]),
        text=(
            f"{status_labels.get(latest_job['status'], latest_job['status'])} "
            f"{float(latest_job['progress']):.0%}"
        ),
    )
    st.write(
        f"実行ID: `{latest_job['job_id']}`  "
        f"対象: {latest_job['start_date']} ～ "
        f"{latest_job['end_date']}"
    )
    if latest_job["result"]:
        st.write(
            f"登録レース: "
            f"{latest_job['result'].get('race_count', 0):,} / "
            f"登録頭数: "
            f"{latest_job['result'].get('row_count', 0):,}"
        )
    if latest_job["error"]:
        st.error(latest_job["error"])
    with st.expander(
        "実行ログ",
        expanded=latest_job["status"] == "running",
    ):
        st.code(
            "\n".join(latest_job["logs"][-100:]),
            language="text",
        )
    if st.button(
        "状況を更新",
        key="refresh_database_creation",
    ):
        st.rerun()
    return latest_job["status"] in {
        "running",
        "cancelling",
    }


@st.fragment(run_every=3)
def _render_active_database_creation_status(
    dataset_type: str | None = None,
) -> None:
    if not _render_database_creation_status(dataset_type):
        st.rerun()


def _render_feature_generation_status() -> bool:
    st.subheader("特徴量生成状況")
    jobs = list_feature_generation_jobs()
    if not jobs:
        st.info("この画面から開始した特徴量生成はありません。")
        return False

    latest_job = jobs[0]
    status_labels = {
        "running": "生成中",
        "cancelling": "中止処理中",
        "cancelled": "中止",
        "completed": "完了",
        "failed": "失敗",
    }
    progress_value = float(latest_job["progress"])
    st.progress(
        progress_value,
        text=(
            f"{status_labels.get(latest_job['status'], latest_job['status'])} "
            f"{progress_value:.0%}"
        ),
    )
    st.write(f"実行ID: `{latest_job['job_id']}`")
    if latest_job["result"]:
        st.success(
            f"{int(latest_job['result'].get('row_count', 0)):,}行の特徴量を保存しました。"
        )
    if latest_job["error"]:
        st.error(latest_job["error"])
    with st.expander(
        "実行ログ",
        expanded=latest_job["status"] in {"running", "cancelling"},
    ):
        st.code("\n".join(latest_job["logs"][-100:]), language="text")
    if st.button("状態を更新", key="refresh_feature_generation"):
        st.rerun()
    return latest_job["status"] in {"running", "cancelling"}


@st.fragment(run_every=3)
def _render_active_feature_generation_status() -> None:
    if not _render_feature_generation_status():
        st.rerun()


def _render_speed_index_status() -> bool:
    st.subheader("スピード指数生成状況")
    jobs = list_speed_index_jobs()
    if not jobs:
        st.info("この画面から開始したスピード指数生成はありません。")
        return False
    latest_job = jobs[0]
    labels = {
        "running": "生成中", "cancelling": "中止処理中",
        "cancelled": "中止", "completed": "完了", "failed": "失敗",
    }
    value = float(latest_job["progress"])
    st.progress(
        value,
        text=f"{labels.get(latest_job['status'], latest_job['status'])} {value:.0%}",
    )
    st.write(f"実行ID: `{latest_job['job_id']}`")
    if latest_job["result"]:
        st.success(
            f"{int(latest_job['result'].get('row_count', 0)):,}走の指数を保存しました。"
        )
    if latest_job["error"]:
        st.error(latest_job["error"])
    with st.expander(
        "実行ログ",
        expanded=latest_job["status"] in {"running", "cancelling"},
    ):
        st.code("\n".join(latest_job["logs"][-100:]), language="text")
    if st.button("状態を更新", key="refresh_speed_index"):
        st.rerun()
    return latest_job["status"] in {"running", "cancelling"}


@st.fragment(run_every=3)
def _render_active_speed_index_status() -> None:
    if not _render_speed_index_status():
        st.rerun()


def _render_speed_index_management() -> None:
    speed_name = "speed_index"
    speed_version = "1.0.0"
    summary = performance_feature_summary(speed_name, speed_version)
    freshness = performance_feature_freshness(speed_name, speed_version)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("特徴量セット", "1走単位スピード指数")
    metric1.caption(f"識別子: {speed_name}:{speed_version}")
    metric2.metric("保存走数", f"{int(summary['row_count']):,}")
    metric3.metric("指数算出済み", f"{int(summary['available_count']):,}")
    if summary["row_count"]:
        st.success(
            f"生成済み期間: {summary['start_date']} ～ {summary['end_date']}　"
            f"実行ID: {summary['latest_run_id']}"
        )
        st.caption(
            f"異常値クリップ: {int(summary['clipped_count']):,}走 / "
            f"元データ指紋: {summary['source_data_fingerprint']}"
        )

    if freshness["status"] == "fresh":
        st.success("同期状態: 最新の元データと同期済みです。")
    elif freshness["status"] == "stale":
        st.warning("同期状態: 元データが更新されています。再生成してください。")
        if freshness["differences"]:
            with st.expander("元データの変更内容"):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "項目": key,
                            "生成時": values.get("generated"),
                            "現在": values.get("current"),
                        }
                        for key, values in freshness["differences"].items()
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
    elif freshness["status"] == "missing":
        st.warning("同期状態: スピード指数が未生成です。")
    else:
        st.warning("同期状態を判定できません。一度再生成してください。")

    st.subheader("生成設定")
    setting1, setting2 = st.columns(2)
    half_life = setting1.number_input(
        "標準タイムの半減期（日）",
        min_value=30, max_value=3650, value=730, step=30,
        key="selected_speed_half_life",
        help="標準タイムに使う過去レースの時間減衰です。初期値730日。原則変更不要です。",
    )
    lookback = setting2.number_input(
        "標準タイムの最大参照期間（日）",
        min_value=365, max_value=7300, value=3650, step=365,
        key="selected_speed_lookback",
        help="標準タイムに使う最長期間です。初期値3650日。原則変更不要です。",
    )
    setting3, setting4 = st.columns(2)
    seconds_scale = setting3.number_input(
        "1600mの1秒差あたりの点数",
        min_value=1.0, max_value=30.0, value=10.0, step=0.5,
        key="selected_speed_seconds_scale",
        help="1600mで標準より1秒速い場合の加点です。初期値10点。原則変更不要です。",
    )
    weight_points = setting4.number_input(
        "斤量1kgあたりの補正点",
        min_value=0.0, max_value=5.0, value=1.5, step=0.1,
        key="selected_speed_weight_points",
        help="55kgを基準にした斤量補正です。初期値1.5点。原則変更不要です。",
    )

    jobs = list_speed_index_jobs()
    active_job = next((
        job for job in jobs if job["status"] in {"running", "cancelling"}
    ), None)
    start_column, cancel_column = st.columns(2)
    with start_column:
        start_clicked = st.button(
            "スピード指数生成を開始",
            type="primary",
            disabled=active_job is not None,
            use_container_width=True,
            key="selected_start_speed_index",
        )
    with cancel_column:
        cancel_clicked = st.button(
            "スピード指数生成を中止",
            disabled=active_job is None,
            use_container_width=True,
            key="selected_cancel_speed_index",
        )
    if start_clicked:
        try:
            job_id = start_speed_index_job(SpeedIndexRunConfig(
                half_life_days=float(half_life),
                max_lookback_days=int(lookback),
                seconds_scale_at_1600m=float(seconds_scale),
                weight_points_per_kg=float(weight_points),
            ))
            st.success(f"スピード指数生成を開始しました。実行ID: {job_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if cancel_clicked and active_job is not None:
        if cancel_speed_index_job(str(active_job["job_id"])):
            st.warning("中止を要求しました。完成済みデータは維持されます。")
            st.rerun()
    if active_job is not None:
        _render_active_speed_index_status()
    else:
        _render_speed_index_status()

    st.subheader("データのクリア")
    confirmed = st.checkbox(
        f"1走単位スピード指数（{speed_name}:{speed_version}）をクリアする",
        disabled=active_job is not None,
        key="selected_confirm_clear_speed",
    )
    if st.button(
        "スピード指数をクリア",
        disabled=not confirmed or active_job is not None,
        key="selected_clear_speed_index",
    ):
        deleted = clear_performance_features(speed_name, speed_version)
        st.success(f"{deleted:,}走をクリアしました。元レースデータは変更していません。")
        st.rerun()

    runs = feature_runs()
    speed_runs = runs[runs["feature_set_name"].eq(speed_name)] if not runs.empty else runs
    if not speed_runs.empty:
        st.subheader("生成履歴")
        st.dataframe(speed_runs.head(20), use_container_width=True, hide_index=True)


def _render_recent_speed_status() -> bool:
    jobs = list_recent_speed_jobs()
    st.subheader("近走スピード成績の生成状況")
    if not jobs:
        st.info("この画面から開始した近走スピード成績の生成はありません。")
        return False
    job = jobs[0]
    labels = {
        "running": "生成中", "cancelling": "中止処理中",
        "cancelled": "中止", "completed": "完了", "failed": "失敗",
    }
    value = float(job["progress"])
    st.progress(value, text=f"{labels.get(job['status'], job['status'])} {value:.0%}")
    st.write(f"実行ID: `{job['job_id']}`")
    if job["result"]:
        st.success(f"{int(job['result'].get('row_count', 0)):,}行を保存しました。")
    if job["error"]:
        st.error(job["error"])
    with st.expander("実行ログ", expanded=job["status"] in {"running", "cancelling"}):
        st.code("\n".join(job["logs"][-100:]), language="text")
    return job["status"] in {"running", "cancelling"}


@st.fragment(run_every=3)
def _render_active_recent_speed_status() -> None:
    if not _render_recent_speed_status():
        st.rerun()


def _render_recent_speed_management() -> None:
    name = "recent_speed"
    version = "1.0.0"
    summary = feature_store_summary(name, version)
    freshness = recent_speed_freshness(name, version)
    dependency = performance_feature_summary("speed_index", "1.0.0")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("特徴量セット", "近走スピード成績")
    metric1.caption(f"識別子: {name}:{version}")
    metric2.metric("保存行数", f"{int(summary['row_count']):,}")
    metric3.metric("依存する指数", "1走単位スピード指数")
    metric3.caption("依存識別子: speed_index:1.0.0")
    if summary["row_count"]:
        st.success(
            f"生成済み期間: {summary['start_date']} ～ {summary['end_date']}　"
            f"実行ID: {summary['latest_run_id']}"
        )
    if freshness["status"] == "fresh":
        st.success("同期状態: 元データ・1走単位スピード指数ともに最新です。")
    elif freshness["status"] == "stale":
        st.warning("同期状態: 元データまたはスピード指数が更新されています。再生成してください。")
    elif freshness["status"] == "missing":
        st.warning("同期状態: 近走スピード成績が未生成です。")
    else:
        st.warning("同期状態を判定できません。一度再生成してください。")
    if dependency["row_count"] == 0:
        st.error("先に1走単位スピード指数（speed_index:1.0.0）を生成してください。")

    st.subheader("生成設定")
    setting1, setting2 = st.columns(2)
    half_life = setting1.number_input(
        "時間減衰の半減期（日）",
        min_value=1, max_value=3650, value=180, step=30,
        key="recent_speed_half_life",
        help="最近のスピード指数を重くする半減期です。初期値180日。原則変更不要です。",
    )
    lookback = setting2.number_input(
        "最大参照期間（日）",
        min_value=30, max_value=7300, value=1095, step=30,
        key="recent_speed_lookback",
        help="集約対象とする過去指数の最長期間です。初期値1095日。原則変更不要です。",
    )
    jobs = list_recent_speed_jobs()
    active = next((job for job in jobs if job["status"] in {"running", "cancelling"}), None)
    start_column, cancel_column = st.columns(2)
    with start_column:
        start_clicked = st.button(
            "近走スピード成績の生成を開始",
            type="primary",
            disabled=active is not None or dependency["row_count"] == 0,
            use_container_width=True,
        )
    with cancel_column:
        cancel_clicked = st.button(
            "近走スピード成績の生成を中止",
            disabled=active is None,
            use_container_width=True,
        )
    if start_clicked:
        try:
            job_id = start_recent_speed_job(RecentSpeedRunConfig(
                half_life_days=float(half_life),
                max_lookback_days=int(lookback),
            ))
            st.success(f"生成を開始しました。実行ID: {job_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if cancel_clicked and active is not None:
        if cancel_recent_speed_job(str(active["job_id"])):
            st.warning("中止を要求しました。完成済みデータは維持されます。")
            st.rerun()
    if active is not None:
        _render_active_recent_speed_status()
    else:
        _render_recent_speed_status()

    st.subheader("データのクリア")
    confirmed = st.checkbox(
        f"近走スピード成績（{name}:{version}）をクリアする",
        disabled=active is not None,
    )
    if st.button("近走スピード成績をクリア", disabled=not confirmed or active is not None):
        deleted = clear_features(name, version)
        st.success(f"{deleted:,}行をクリアしました。元レース・1走指数は変更していません。")
        st.rerun()
    runs = feature_runs()
    selected_runs = runs[runs["feature_set_name"].eq(name)] if not runs.empty else runs
    if not selected_runs.empty:
        st.subheader("生成履歴")
        st.dataframe(selected_runs.head(20), use_container_width=True, hide_index=True)


def _render_race_entry_status() -> bool:
    jobs = list_race_entry_jobs()
    st.subheader("出馬表基本条件の生成状況")
    if not jobs:
        st.info("この画面から開始した出馬表基本条件の生成はありません。")
        return False
    job = jobs[0]
    labels = {
        "running": "生成中", "cancelling": "中止処理中",
        "cancelled": "中止", "completed": "完了", "failed": "失敗",
    }
    value = float(job["progress"])
    st.progress(value, text=f"{labels.get(job['status'], job['status'])} {value:.0%}")
    st.write(f"実行ID: `{job['job_id']}`")
    if job["result"]:
        st.success(f"{int(job['result'].get('row_count', 0)):,}行を保存しました。")
    if job["error"]:
        st.error(job["error"])
    with st.expander("実行ログ", expanded=job["status"] in {"running", "cancelling"}):
        st.code("\n".join(job["logs"][-100:]), language="text")
    return job["status"] in {"running", "cancelling"}


@st.fragment(run_every=3)
def _render_active_race_entry_status() -> None:
    if not _render_race_entry_status():
        st.rerun()


def _render_race_entry_management() -> None:
    name = "race_entry"
    version = "1.0.0"
    summary = feature_store_summary(name, version)
    freshness = feature_freshness(name, version)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("特徴量セット", "出馬表基本条件")
    metric1.caption(f"識別子: {name}:{version}")
    metric2.metric("保存行数", f"{int(summary['row_count']):,}")
    metric3.metric("生成項目数", "14")
    if summary["row_count"]:
        st.success(
            f"生成済み期間: {summary['start_date']} ～ {summary['end_date']}　"
            f"実行ID: {summary['latest_run_id']}"
        )
    if freshness["status"] == "fresh":
        st.success("同期状態: 元データと同期しています。")
    elif freshness["status"] == "stale":
        st.warning("同期状態: 元データが更新されています。再生成してください。")
    elif freshness["status"] == "missing":
        st.warning("同期状態: 出馬表基本条件が未生成です。")
    else:
        st.warning("同期状態を判定できません。一度再生成してください。")

    st.subheader("生成設定")
    st.text_input(
        "対象情報",
        value="競馬場・レース番号・芝ダート・距離・馬場状態・馬齢・性別・斤量・枠馬番・相対値・前走距離差",
        disabled=True,
        help=(
            "出馬表情報と馬場状態を生成します。天気・オッズ・人気・馬体重は"
            "含みません。予想時の馬場状態は競馬場ごとに画面で指定します。"
        ),
    )
    jobs = list_race_entry_jobs()
    active = next((job for job in jobs if job["status"] in {"running", "cancelling"}), None)
    start_column, cancel_column = st.columns(2)
    with start_column:
        start_clicked = st.button(
            "出馬表基本条件の生成を開始",
            type="primary",
            disabled=active is not None,
            use_container_width=True,
        )
    with cancel_column:
        cancel_clicked = st.button(
            "出馬表基本条件の生成を中止",
            disabled=active is None,
            use_container_width=True,
        )
    if start_clicked:
        try:
            job_id = start_race_entry_job(RaceEntryRunConfig())
            st.success(f"生成を開始しました。実行ID: {job_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if cancel_clicked and active is not None:
        if cancel_race_entry_job(str(active["job_id"])):
            st.warning("中止を要求しました。完成済みデータは維持されます。")
            st.rerun()
    if active is not None:
        _render_active_race_entry_status()
    else:
        _render_race_entry_status()

    st.subheader("データのクリア")
    confirmed = st.checkbox(
        f"出馬表基本条件（{name}:{version}）をクリアする",
        disabled=active is not None,
    )
    if st.button("出馬表基本条件をクリア", disabled=not confirmed or active is not None):
        deleted = clear_features(name, version)
        st.success(f"{deleted:,}行をクリアしました。元レースデータは変更していません。")
        st.rerun()
    runs = feature_runs()
    selected_runs = runs[runs["feature_set_name"].eq(name)] if not runs.empty else runs
    if not selected_runs.empty:
        st.subheader("生成履歴")
        st.dataframe(selected_runs.head(20), use_container_width=True, hide_index=True)


with st.sidebar:
    page = st.radio(
        "メニュー",
        ["ダッシュボード", "HTML収集（学習用）", "HTML収集（予想用）", "データベース作成", "前準備（特徴量エンジニアリング）", "モデル学習", "レース予想", "ログ・保存結果", "メンテナンス"],
    )
    st.divider()
    st.caption(f"DB: {PATHS.database}")
    st.caption(f"特徴量DB: {PATHS.feature_database}")
    st.caption("推奨Python: 3.11")

log_placeholder = None


if page == "ダッシュボード":
    summary = dashboard_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("過去データ行数", f"{summary['history_rows']:,}")
    c2.metric("予想用データ行数", f"{summary['upcoming_rows']:,}")
    c3.metric("登録レース数", f"{summary['races']:,}")
    c4.metric("学習済みモデル", f"{summary['model_count']:,}")
    st.info(
        "最初は［データベース作成］のデモデータ生成を実行してください。HTML収集前でも、学習から予想まで動作確認できます。"
    )
    runs = collection_runs()
    if not runs.empty:
        st.subheader("直近のデータ収集")
        st.dataframe(runs.head(10), use_container_width=True, hide_index=True)

elif page in {"HTML収集（学習用）", "HTML収集（予想用）"}:
    dataset_type = "historical" if page == "HTML収集（学習用）" else "upcoming"
    st.header(page)
    st.markdown(
        """
        <style>
        .st-key-start_html_collection button {
            background-color: #1677ff !important;
            border-color: #1677ff !important;
            color: white !important;
        }
        .st-key-start_html_collection button:hover {
            background-color: #0958d9 !important;
            border-color: #0958d9 !important;
        }
        .st-key-cancel_html_collection button {
            background-color: #1677ff !important;
            border-color: #1677ff !important;
            color: white !important;
        }
        .st-key-cancel_html_collection button:hover {
            background-color: #0958d9 !important;
            border-color: #0958d9 !important;
        }
        .st-key-refresh_html_collection button {
            background-color: #1677ff !important;
            border-color: #1677ff !important;
            color: white !important;
        }
        .st-key-refresh_html_collection button:hover {
            background-color: #0958d9 !important;
            border-color: #0958d9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if dataset_type == "historical":
        current_year = date.today().year
        available_years = list(
            range(current_year, 1985, -1)
        )
        years_with_data = {
            year
            for year in available_years
            if any(
                (
                    PATHS.historical_html / str(year)
                ).rglob("*.html")
            )
        }
        selected_year = st.selectbox(
            "取得年",
            options=available_years,
            index=0,
            format_func=lambda year: (
                f"{year}年 ※データあり"
                if year in years_with_data
                else f"{year}年"
            ),
        )
        start_date = date(int(selected_year), 1, 1)
        end_date = min(
            date(int(selected_year), 12, 31),
            date.today(),
        )
        st.caption(
            f"{selected_year}年の過去レース結果HTMLを取得します。"
        )
    else:
        system_now = datetime.now()
        system_date = system_now.date()
        prediction_start_date = (
            system_date
            if system_now.hour < 9
            else system_date + timedelta(days=1)
        )
        prediction_date_limit = system_date + timedelta(
            days=7
        )
        stored_prediction_date = st.session_state.get(
            "upcoming_collection_date"
        )
        if stored_prediction_date is not None:
            stored_prediction_date = pd.to_datetime(
                stored_prediction_date,
                errors="coerce",
            )
            if (
                pd.isna(stored_prediction_date)
                or stored_prediction_date.date()
                < prediction_start_date
                or stored_prediction_date.date()
                > prediction_date_limit
            ):
                st.session_state[
                    "upcoming_collection_date"
                ] = prediction_start_date
        target_date = st.date_input(
            "取得日",
            value=prediction_start_date,
            min_value=prediction_start_date,
            max_value=prediction_date_limit,
            key="upcoming_collection_date",
        )
        start_date = target_date
        end_date = target_date
        st.caption(
            f"{target_date}の予想用出馬表HTMLを取得します。"
            f"選択可能期間: {prediction_start_date} ～ "
            f"{prediction_date_limit}。"
            "システム日付当日は午前9:00より前まで"
            "選択できます。"
        )

    force = st.checkbox("保存済みHTMLを再取得する", value=False)
    html_storage_path = (
        "data/raw_html/historical/<年>/"
        if dataset_type == "historical"
        else "data/raw_html/upcoming/<年>/"
    )
    st.caption(
        "ページ取得間隔は2秒固定です。"
        "処理はバックグラウンドで継続し、"
        f"HTMLは {html_storage_path} 以下へ保存します。"
    )
    current_jobs = list_jobs(dataset_type)
    active_job = next(
        (
            job
            for job in current_jobs
            if job["status"] in {"running", "cancelling"}
        ),
        None,
    )
    start_column, cancel_column = st.columns(2)
    with start_column:
        if st.button(
            "HTML収集を開始",
            type="primary",
            key="start_html_collection",
            disabled=active_job is not None,
            use_container_width=True,
        ):
            try:
                job_id = start_job(dataset_type, start_date, end_date, force)
                st.success(
                    f"バックグラウンド収集を開始しました。実行ID: {job_id}"
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with cancel_column:
        if st.button(
            "処理を中止",
            key="cancel_html_collection",
            disabled=active_job is None,
            use_container_width=True,
        ):
            if active_job and cancel_job(str(active_job["job_id"])):
                st.warning("中止を要求しました。現在のページ処理後に停止します。")
                st.rerun()
            else:
                st.info("中止できるHTML収集はありません。")

    if active_job is not None:
        _render_active_collection_status(dataset_type)
    else:
        _render_collection_status(dataset_type)

elif page == "データベース作成":
    st.header("データベース作成")
    mode = st.radio(
        "作成方法",
        [
            "取得済みHTMLから作成",
            "デモデータを作成",
            "登録内容確認",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "デモデータを作成":
        st.info(
            "デモデータは既存データを残したまま追加登録されます。"
            "デモデータを削除する場合は、メンテナンス画面の"
            "「ダミーデータ」から削除してください。"
        )
        st.write("実サイトへアクセスせず、学習・評価・予想を試せる合成データを保存します。")
        historical_races = st.slider("過去レース数", min_value=100, max_value=1000, value=360, step=20)
        upcoming_races = st.slider("週末レース数", min_value=1, max_value=12, value=6)
        if st.button("デモデータ生成・保存", type="primary"):
            logger = new_logger()
            run_id = f"demo_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            started = datetime.now()
            try:
                logger.info("デモデータ生成を開始")
                historical, upcoming = generate_demo_records(
                    historical_races=historical_races,
                    upcoming_races=upcoming_races,
                )
                save_race_records(
                    historical, run_id + "_history", "historical"
                )
                save_race_records(
                    upcoming, run_id + "_upcoming", "upcoming"
                )
                save_collection_run(
                    {
                        "run_id": run_id,
                        "started_at": started,
                        "completed_at": datetime.now(),
                        "status": "completed",
                        "dataset_type": "demo",
                        "source": "generator",
                        "start_date": historical["race_date"].min(),
                        "end_date": upcoming["race_date"].max(),
                        "race_count": historical["race_id"].nunique() + upcoming["race_id"].nunique(),
                        "row_count": len(historical) + len(upcoming),
                        "output_path": "",
                        "message": "デモデータ生成完了",
                    }
                )
                logger.info(f"保存完了: 過去={len(historical)}行、予想用={len(upcoming)}行")
                st.success("デモデータを保存しました。次に［モデル学習］へ進んでください。")
            except Exception as exc:
                logger.exception(f"デモデータ生成失敗: {exc}")
                st.exception(exc)
    elif mode == "取得済みHTMLから作成":
        st.info(
            "別の年・日付を指定して作成したデータは、"
            "既存データを残したまま追加登録されます。"
            "同じレースID・競走馬ID・データ種別のデータを"
            "再登録した場合だけ、該当データを最新内容へ更新します。"
            "既に登録されている別年のデータは削除されません。"
        )
        dataset_type_label = st.radio(
            "対象", ["学習用の過去結果", "予想用の出馬表"], horizontal=True
        )
        dataset_type = (
            "historical"
            if dataset_type_label == "学習用の過去結果"
            else "upcoming"
        )
        if dataset_type == "historical":
            database_creation_available = True
            database_years = race_record_years("historical")
            historical_years = list(
                range(date.today().year, 1985, -1)
            )
            html_years = {
                year
                for year in historical_years
                if any(
                    (
                        PATHS.historical_html
                        / str(year)
                        / "result"
                    ).glob("*.html")
                )
            }

            def database_year_label(year: int) -> str:
                statuses: list[str] = []
                if year in html_years:
                    statuses.append("※HTML取得済み")
                if year in database_years:
                    statuses.append("※データベース作成済み")
                suffix = (
                    " " + " ".join(statuses)
                    if statuses
                    else ""
                )
                return f"{year}年{suffix}"

            selected_year = st.selectbox(
                "対象年",
                historical_years,
                format_func=database_year_label,
                key="database_year",
            )
            start_date = date(int(selected_year), 1, 1)
            end_date = min(date(int(selected_year), 12, 31), date.today())
        else:
            upcoming_html_dates = _available_upcoming_html_dates()
            database_creation_available = bool(upcoming_html_dates)
            if upcoming_html_dates:
                date_statuses: dict[date, str] = {}
                for available_date in upcoming_html_dates:
                    race_count = _cached_race_html_count(
                        "upcoming", available_date, available_date
                    )
                    registered = race_record_summary(
                        "upcoming", available_date, available_date
                    )
                    registered_text = (
                        f"・DB作成済み {registered['race_count']:,}レース"
                        if registered["race_count"] > 0 else ""
                    )
                    date_statuses[available_date] = (
                        f"{available_date:%Y-%m-%d}（HTML {race_count:,}レース"
                        f"{registered_text}）"
                    )
                target_date = st.selectbox(
                    "対象日",
                    upcoming_html_dates,
                    format_func=lambda value: date_statuses[value],
                    key="database_upcoming_html_date",
                    help=(
                        "収集済みのレース一覧HTMLと出馬表HTMLが揃っている日付だけを"
                        "表示します。通常は最新の取得日が先頭です。"
                    ),
                )
                start_date = target_date
                end_date = target_date
            else:
                start_date = date.today()
                end_date = date.today()
                st.warning(
                    "予想用の取得済みHTMLがありません。先にHTML収集（予想用）を実行してください。"
                )

        if dataset_type == "upcoming":
            html_race_count = _cached_race_html_count(
                dataset_type,
                start_date,
                end_date,
            )
            database_summary = race_record_summary(
                dataset_type,
                start_date,
                end_date,
            )
            status_messages: list[str] = []
            if html_race_count > 0:
                status_messages.append(
                    f"※HTML取得済み（{html_race_count:,}レース）"
                )
            if database_summary["race_count"] > 0:
                status_messages.append(
                    "※データベース作成済み"
                    f"（{database_summary['race_count']:,}レース・"
                    f"{database_summary['row_count']:,}頭）"
                )
            if status_messages:
                st.success("　".join(status_messages))

        st.caption(
            "ネットワークアクセスは行わず、取得済みHTMLだけを"
            "バックグラウンドで解析・登録します。"
            "画面を移動しても処理は継続します。"
        )
        database_jobs = list_database_jobs()
        active_database_job = next(
            (
                job
                for job in database_jobs
                if job["status"] in {
                    "running",
                    "cancelling",
                }
            ),
            None,
        )
        start_column, cancel_column = st.columns(2)
        with start_column:
            start_database = st.button(
                "データベース作成を開始",
                type="primary",
                disabled=(
                    active_database_job is not None
                    or not database_creation_available
                ),
                use_container_width=True,
            )
        with cancel_column:
            cancel_database = st.button(
                "処理を中止",
                key="cancel_database_creation",
                disabled=active_database_job is None,
                use_container_width=True,
            )

        if start_database:
            try:
                job_id = start_database_job(
                    dataset_type=dataset_type,
                    start_date=start_date,
                    end_date=end_date,
                )
                st.success(
                    "バックグラウンドでデータベース作成を"
                    f"開始しました。実行ID: {job_id}"
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if cancel_database:
            if (
                active_database_job
                and cancel_database_job(
                    str(active_database_job["job_id"])
                )
            ):
                st.warning(
                    "中止を要求しました。現在の解析単位が"
                    "完了した後に停止します。"
                )
                st.rerun()
            else:
                st.info(
                    "中止できるデータベース作成はありません。"
                )

        if active_database_job is not None:
            _render_active_database_creation_status()
        else:
            _render_database_creation_status()

    else:
        confirmation_type_label = st.radio(
            "確認対象",
            ["学習用の過去結果", "予想用の出馬表"],
            horizontal=True,
            key="registered_data_type",
        )
        confirmation_dataset_type = (
            "historical"
            if confirmation_type_label == "学習用の過去結果"
            else "upcoming"
        )
        records = load_records(confirmation_dataset_type)
        if records.empty:
            st.info(
                f"{confirmation_type_label}としてデータベースに"
                "登録されているレースデータはありません。"
            )
        else:
            records = records.copy()
            records["race_date"] = pd.to_datetime(
                records["race_date"],
                errors="coerce",
            )
            records = records[
                records["race_date"].notna()
            ].copy()

            if records.empty:
                st.info(
                    "開催日を確認できる登録データがありません。"
                )
            else:
                records["race_year"] = (
                    records["race_date"].dt.year.astype(int)
                )
                registered_years = sorted(
                    records["race_year"].unique().tolist(),
                    reverse=True,
                )
                selected_registered_year = st.selectbox(
                    "開催年",
                    registered_years,
                    key="registered_data_year",
                )
                year_records = records[
                    records["race_year"]
                    == int(selected_registered_year)
                ].copy()

                race_options = (
                    year_records.sort_values(
                        [
                            "race_date",
                            "course_name",
                            "race_number",
                            "race_id",
                        ]
                    )
                    .drop_duplicates(
                        ["dataset_type", "race_id"]
                    )
                    .reset_index(drop=True)
                )
                dataset_labels = {
                    "historical": "学習用",
                    "upcoming": "予想用",
                }

                def registered_race_label(index: int) -> str:
                    row = race_options.iloc[index]
                    race_number = (
                        f"{int(row['race_number'])}R"
                        if pd.notna(row["race_number"])
                        else "レース番号不明"
                    )
                    return (
                        f"{row['race_date']:%Y-%m-%d} "
                        f"{row['course_name']} "
                        f"{race_number} "
                        f"{row['race_name']} "
                        f"[{dataset_labels.get(str(row['dataset_type']), str(row['dataset_type']))}]"
                    )

                race_indices = list(range(len(race_options)))
                selected_race_index = st.selectbox(
                    "開催レース",
                    race_indices,
                    format_func=registered_race_label,
                    key="registered_data_race",
                )
                selected_race = race_options.iloc[
                    int(selected_race_index)
                ]
                selected_rows = year_records[
                    (
                        year_records["race_id"].astype(str)
                        == str(selected_race["race_id"])
                    )
                    & (
                        year_records["dataset_type"].astype(str)
                        == str(selected_race["dataset_type"])
                    )
                ].sort_values(
                    ["finish_position", "horse_number"],
                    na_position="last",
                )

                st.subheader("レース情報")
                race_columns = [
                    "race_id",
                    "race_date",
                    "dataset_type",
                    "course_name",
                    "race_number",
                    "race_name",
                    "surface",
                    "distance",
                    "weather",
                    "track_condition",
                ]
                race_display = selected_rows[
                    [
                        column
                        for column in race_columns
                        if column in selected_rows.columns
                    ]
                ].head(1).copy()
                race_display["dataset_type"] = (
                    race_display["dataset_type"]
                    .astype(str)
                    .map(dataset_labels)
                    .fillna(race_display["dataset_type"])
                )
                st.dataframe(
                    race_display,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    f"出走馬情報（{len(selected_rows):,}頭）"
                )
                registered_columns = [
                    column
                    for column in selected_rows.columns
                    if column != "race_year"
                ]
                st.dataframe(
                    selected_rows[registered_columns],
                    use_container_width=True,
                    hide_index=True,
                )

elif page == "前準備（特徴量エンジニアリング）":
    st.header("前準備（特徴量エンジニアリング）")
    st.info(
        "学習用の過去レース特徴量を事前生成します。モデル学習時には自動生成されません。"
        "元データを更新した場合や特徴量設定を変更した場合は再生成してください。"
    )

    selected_feature_set = st.selectbox(
        "特徴量セット",
        [
            "baseline:1.1.0", "speed_index:1.0.0", "recent_speed:1.0.0",
            "race_entry:1.1.0",
        ],
        format_func=lambda value: {
            "baseline:1.1.0": "基礎近走成績（baseline:1.1.0）",
            "speed_index:1.0.0": "1走単位スピード指数（speed_index:1.0.0）",
            "recent_speed:1.0.0": "近走スピード成績（recent_speed:1.0.0）",
            "race_entry:1.1.0": "出馬表基本条件・馬場状態（race_entry:1.1.0）",
        }[value],
        help="表示・生成・中止・クリアする特徴量セットを選択します。",
    )
    if selected_feature_set == "speed_index:1.0.0":
        _render_speed_index_management()
        st.stop()
    if selected_feature_set == "recent_speed:1.0.0":
        _render_recent_speed_management()
        st.stop()
    if selected_feature_set == "race_entry:1.1.0":
        _render_race_entry_management()
        st.stop()

    feature_set_name = "baseline"
    feature_set_version = "1.1.0"
    summary = feature_store_summary(feature_set_name, feature_set_version)
    freshness = feature_freshness(feature_set_name, feature_set_version)
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("特徴量セット", "基礎近走成績")
    metric1.caption(f"識別子: {feature_set_name}:{feature_set_version}")
    metric2.metric("保存行数", f"{int(summary['row_count']):,}")
    metric3.metric("対象レース数", f"{int(summary['race_count']):,}")
    generated_at = pd.to_datetime(summary["generated_at"], errors="coerce")
    st.markdown("#### 最終生成日時")
    if pd.notna(generated_at):
        generated_date_column, generated_time_column = st.columns(2)
        generated_date_column.metric("日付", generated_at.strftime("%Y-%m-%d"))
        generated_time_column.metric("時刻", generated_at.strftime("%H:%M:%S"))
    else:
        st.info("まだ特徴量は生成されていません。")
    if summary["row_count"]:
        fingerprint = str(summary.get("source_data_fingerprint") or "")
        st.success(
            f"生成済み期間: {summary['start_date']} ～ {summary['end_date']}　"
            f"実行ID: {summary['latest_run_id']}"
        )
        if fingerprint:
            st.caption(f"元データ指紋: {fingerprint}")

    freshness_status = str(freshness["status"])
    if freshness_status == "fresh":
        st.success("同期状態: 最新の元データと同期済みです。")
    elif freshness_status == "stale":
        st.warning(
            "同期状態: 元データが更新されています。モデル学習前に特徴量を再生成してください。"
        )
        differences = freshness.get("differences", {})
        if differences:
            difference_rows = [
                {
                    "項目": key,
                    "特徴量生成時": values.get("generated"),
                    "現在": values.get("current"),
                }
                for key, values in differences.items()
            ]
            with st.expander("元データの変更内容"):
                st.dataframe(
                    pd.DataFrame(difference_rows),
                    use_container_width=True,
                    hide_index=True,
                )
    elif freshness_status == "missing":
        st.warning("同期状態: 特徴量が未生成です。特徴量生成を実行してください。")
    else:
        st.warning(
            "同期状態: 既存の特徴量には元データ世代情報がないため判定できません。"
            "一度再生成してください。"
        )

    st.subheader("生成設定")
    setting1, setting2 = st.columns(2)
    half_life_days = setting1.number_input(
        "時間減衰の半減期（日）",
        min_value=1,
        max_value=3650,
        value=180,
        step=30,
        help=(
            "過去成績の影響が半分になるまでの日数です。短くすると最近の成績をより強く"
            "評価し、長くすると古い成績も残りやすくなります。初期値は180日です。"
            "モデル比較など明確な目的がない限り、原則変更不要です。"
        ),
    )
    max_lookback_days = setting2.number_input(
        "最大参照期間（日）",
        min_value=30,
        max_value=7300,
        value=1095,
        step=30,
        help=(
            "時間減衰集計で参照する過去成績の最長期間です。初期値1095日は約3年間です。"
            "これより古い成績は集計対象から除外します。データ期間を変更する明確な目的が"
            "ない限り、原則変更不要です。"
        ),
    )
    st.caption(
        "全期間の履歴を対象に生成します。対象レース当日および未来の結果は使用しません。"
    )

    feature_jobs = list_feature_generation_jobs()
    active_feature_job = next(
        (
            job for job in feature_jobs
            if job["status"] in {"running", "cancelling"}
        ),
        None,
    )
    start_column, cancel_column = st.columns(2)
    with start_column:
        start_features = st.button(
            "特徴量生成を開始",
            type="primary",
            disabled=active_feature_job is not None,
            use_container_width=True,
        )
    with cancel_column:
        cancel_features = st.button(
            "生成を中止",
            disabled=active_feature_job is None,
            use_container_width=True,
        )

    if start_features:
        try:
            job_id = start_feature_generation_job(
                RecentFormRunConfig(
                    feature_set_name=feature_set_name,
                    feature_set_version=feature_set_version,
                    half_life_days=float(half_life_days),
                    max_lookback_days=int(max_lookback_days),
                )
            )
            st.success(f"バックグラウンド生成を開始しました。実行ID: {job_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if cancel_features and active_feature_job is not None:
        if cancel_feature_generation_job(str(active_feature_job["job_id"])):
            st.warning("中止を要求しました。完成済みの特徴量は維持されます。")
            st.rerun()

    if active_feature_job is not None:
        _render_active_feature_generation_status()
    else:
        _render_feature_generation_status()

    runs = feature_runs()
    recent_runs = runs[runs["feature_set_name"].eq(feature_set_name)] if not runs.empty else runs
    if not recent_runs.empty:
        st.subheader("基礎近走成績の生成履歴")
        st.dataframe(recent_runs.head(20), use_container_width=True, hide_index=True)

    # 選択中の基礎近走成績だけを表示し、下に残る旧スピード指数パネルは実行しません。
    st.stop()

    st.divider()
    st.header("1走単位スピード指数")
    speed_name = "speed_index"
    speed_version = "1.0.0"
    speed_summary = performance_feature_summary(speed_name, speed_version)
    speed_freshness = performance_feature_freshness(speed_name, speed_version)
    speed_metric1, speed_metric2, speed_metric3 = st.columns(3)
    speed_metric1.metric("特徴量セット", "1走単位スピード指数")
    speed_metric1.caption(f"識別子: {speed_name}:{speed_version}")
    speed_metric2.metric("保存走数", f"{int(speed_summary['row_count']):,}")
    speed_metric3.metric("指数算出済み", f"{int(speed_summary['available_count']):,}")
    if speed_summary["row_count"]:
        st.success(
            f"生成済み期間: {speed_summary['start_date']} ～ {speed_summary['end_date']}　"
            f"実行ID: {speed_summary['latest_run_id']}"
        )
        st.caption(
            f"異常値クリップ: {int(speed_summary['clipped_count']):,}走 / "
            f"元データ指紋: {speed_summary['source_data_fingerprint']}"
        )
    if speed_freshness["status"] == "fresh":
        st.success("同期状態: 最新の元データと同期済みです。")
    elif speed_freshness["status"] == "stale":
        st.warning("同期状態: 元データが更新されています。スピード指数を再生成してください。")
        if speed_freshness["differences"]:
            with st.expander("元データの変更内容"):
                st.dataframe(pd.DataFrame([
                    {
                        "項目": key,
                        "生成時": values.get("generated"),
                        "現在": values.get("current"),
                    }
                    for key, values in speed_freshness["differences"].items()
                ]), use_container_width=True, hide_index=True)
    elif speed_freshness["status"] == "missing":
        st.warning("同期状態: スピード指数が未生成です。")
    else:
        st.warning("同期状態を判定できません。一度再生成してください。")

    st.subheader("スピード指数の生成設定")
    speed_setting1, speed_setting2 = st.columns(2)
    speed_half_life = speed_setting1.number_input(
        "標準タイムの半減期（日）",
        min_value=30, max_value=3650, value=730, step=30,
        help=(
            "標準タイム算出で古いレースの影響が半分になる日数です。初期値は730日です。"
            "検証目的がない限り、原則変更不要です。"
        ),
    )
    speed_lookback = speed_setting2.number_input(
        "標準タイムの最大参照期間（日）",
        min_value=365, max_value=7300, value=3650, step=365,
        help=(
            "標準タイムの算出対象とする最長期間です。初期値3650日は約10年です。"
            "検証目的がない限り、原則変更不要です。"
        ),
    )
    speed_setting3, speed_setting4 = st.columns(2)
    seconds_scale = speed_setting3.number_input(
        "1600mの1秒差あたりの点数",
        min_value=1.0, max_value=30.0, value=10.0, step=0.5,
        help=(
            "1600mで標準タイムより1秒速い場合に加算する点数です。距離に応じて補正されます。"
            "初期値は10点です。原則変更不要です。"
        ),
    )
    weight_points = speed_setting4.number_input(
        "斤量1kgあたりの補正点",
        min_value=0.0, max_value=5.0, value=1.5, step=0.1,
        help=(
            "55kgを基準に、重い斤量で走った実績を評価する補正です。初期値は1.5点です。"
            "原則変更不要です。"
        ),
    )

    speed_jobs = list_speed_index_jobs()
    active_speed_job = next((
        job for job in speed_jobs if job["status"] in {"running", "cancelling"}
    ), None)
    speed_start_column, speed_cancel_column = st.columns(2)
    with speed_start_column:
        start_speed = st.button(
            "スピード指数生成を開始",
            type="primary",
            disabled=active_speed_job is not None,
            use_container_width=True,
        )
    with speed_cancel_column:
        cancel_speed = st.button(
            "スピード指数生成を中止",
            disabled=active_speed_job is None,
            use_container_width=True,
        )
    if start_speed:
        try:
            job_id = start_speed_index_job(SpeedIndexRunConfig(
                half_life_days=float(speed_half_life),
                max_lookback_days=int(speed_lookback),
                seconds_scale_at_1600m=float(seconds_scale),
                weight_points_per_kg=float(weight_points),
            ))
            st.success(f"スピード指数生成を開始しました。実行ID: {job_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if cancel_speed and active_speed_job is not None:
        if cancel_speed_index_job(str(active_speed_job["job_id"])):
            st.warning("中止を要求しました。完成済みの指数は維持されます。")
            st.rerun()
    if active_speed_job is not None:
        _render_active_speed_index_status()
    else:
        _render_speed_index_status()

    speed_runs = runs[runs["feature_set_name"].eq(speed_name)] if not runs.empty else runs
    if not speed_runs.empty:
        st.subheader("1走単位スピード指数の生成履歴")
        st.dataframe(speed_runs.head(20), use_container_width=True, hide_index=True)

elif page == "モデル学習":
    st.header("ディープラーニングモデル学習")
    pending_training_parameters = st.session_state.pop(
        "pending_training_parameters",
        None,
    )
    if pending_training_parameters:
        for parameter_key, parameter_value in (
            pending_training_parameters.items()
        ):
            st.session_state[parameter_key] = parameter_value
        st.success(
            "学習評価画面の調整案を入力欄へ反映しました。"
        )
    model_type = st.selectbox(
        "学習モデル",
        ["各馬が3着以内に入る確率を予測する"],
        key="training_model_type",
        help=(
            "今後、予測目的の異なるモデルを追加した場合に、"
            "ここから学習するモデルを選択します。"
        ),
    )
    st.caption(
        f"選択中: {model_type}モデル。"
        "レース条件、馬の過去成績、騎手・調教師、"
        "父・母・母父の情報を使って学習します。"
    )
    historical = load_records("historical")
    st.write(f"利用可能な過去データ: **{len(historical):,}行 / {historical['race_id'].nunique() if not historical.empty else 0:,}レース**")
    training_feature_states = [
        ("基礎近走成績", "baseline:1.1.0", feature_freshness("baseline", "1.1.0")),
        ("近走スピード成績", "recent_speed:1.0.0", recent_speed_freshness()),
        ("出馬表基本条件・馬場状態", "race_entry:1.1.0", feature_freshness("race_entry", "1.1.0")),
    ]
    st.subheader("学習に使用する特徴量")
    st.dataframe(
        pd.DataFrame([
            {
                "特徴量セット": label,
                "識別子": identifier,
                "鮮度": state["status"],
            }
            for label, identifier, state in training_feature_states
        ]),
        use_container_width=True,
        hide_index=True,
    )
    features_ready = all(
        state["status"] == "fresh" for _, _, state in training_feature_states
    )
    if not features_ready:
        st.error(
            "未生成または古い特徴量があります。前準備（特徴量エンジニアリング）で"
            "対象セットを生成してから学習してください。"
        )
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input(
        "最大エポック数",
        min_value=10,
        max_value=500,
        value=80,
        step=10,
        key="training_epochs",
        help=(
            "学習データ全体を繰り返し学習する上限回数です。"
            "少ないと学習不足、多すぎると時間増加や過学習につながります。"
            "Early Stoppingを併用するため、まずは80程度が目安です。"
        ),
    )
    batch_size = c2.selectbox(
        "バッチサイズ",
        [32, 64, 128, 256],
        index=2,
        key="training_batch_size",
        help=(
            "1回の重み更新に使うデータ数です。小さいほど細かく学習しますが"
            "時間がかかり、大きいほど高速ですがメモリを多く使います。"
            "通常は64～128が目安です。"
        ),
    )
    learning_rate = c3.selectbox(
        "学習率",
        [0.0001, 0.0003, 0.001, 0.003],
        index=2,
        key="training_learning_rate",
        help=(
            "1回の更新でモデルを変化させる大きさです。"
            "大きすぎると学習が不安定になり、小さすぎると収束が遅くなります。"
            "まずは0.001を基準に調整します。"
        ),
    )
    c4, c5, c6 = st.columns(3)
    hidden_dim = c4.selectbox(
        "隠れ層サイズ",
        [32, 64, 128, 256],
        index=1,
        key="training_hidden_dim",
        help=(
            "ニューラルネットワークが学習できる表現の大きさです。"
            "大きいほど複雑な関係を学べますが、処理負荷と過学習の可能性が"
            "高まります。まずは64が目安です。"
        ),
    )
    dropout = c5.slider(
        "Dropout",
        0.0,
        0.7,
        0.25,
        0.05,
        key="training_dropout",
        help=(
            "学習中に一部のニューロンを無効化して過学習を抑える割合です。"
            "過学習時は増やし、学習不足なら減らします。0.2～0.4が目安です。"
        ),
    )
    patience = c6.number_input(
        "Early Stopping待機",
        3,
        50,
        12,
        key="training_patience",
        help=(
            "検証結果が改善しなくなってから学習を止めるまでの待機回数です。"
            "小さいほど早く停止し、大きいほど改善を長く待ちます。"
            "まずは10～15が目安です。"
        ),
    )
    st.caption("開催日順に70%／15%／15%へ分割し、未来データが学習側へ混ざらないようにします。")
    if st.button(
        "モデル学習を開始",
        type="primary",
        disabled=historical.empty or not features_ready,
    ):
        logger = new_logger()
        progress_bar = st.progress(0.0, text="学習中")
        try:
            metrics = train_model(
                historical,
                TrainConfig(
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    patience=int(patience),
                ),
                log=logger.info,
                progress=lambda value: progress_bar.progress(value, text=f"学習中 {value:.0%}"),
            )
            progress_bar.progress(1.0, text="学習完了")
            st.success(f"モデルを保存しました: {metrics['model_run_id']}")
            st.json(metrics)
        except Exception as exc:
            logger.exception(f"学習失敗: {exc}")
            st.exception(exc)

if page == "モデル学習":
    st.divider()
    st.header("学習結果・過学習チェック")
    runs = model_runs()
    if runs.empty:
        st.warning("学習済みモデルがありません。")
    else:
        model_type_labels = {
            "top3": "各馬が3着以内に入る確率を予測するモデル",
        }

        def evaluation_model_label(model_run_id: str) -> str:
            row = runs[
                runs["model_run_id"] == model_run_id
            ].iloc[0]
            model_type = model_type_labels.get(
                str(row["target"]),
                f"モデル種別: {row['target']}",
            )
            created_at = pd.to_datetime(
                row["created_at"],
                errors="coerce",
            )
            created_text = (
                created_at.strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(created_at)
                else "作成日時不明"
            )
            test_auc = (
                f"{float(row['test_auc']):.3f}"
                if pd.notna(row["test_auc"])
                else "-"
            )
            return (
                f"{model_type}｜{created_text}｜"
                f"テストAUC {test_auc}｜{model_run_id}"
            )

        selected_id = st.selectbox(
            "評価するモデル",
            runs["model_run_id"].tolist(),
            format_func=evaluation_model_label,
        )
        selected = runs[runs["model_run_id"] == selected_id].iloc[0]
        selected_model_type = model_type_labels.get(
            str(selected["target"]),
            f"モデル種別: {selected['target']}",
        )
        st.info(
            f"選択中：{selected_model_type}\n\n"
            f"モデルID：`{selected_id}`"
        )
        model_dir = Path(selected["model_path"])
        history = pd.read_csv(model_dir / "training_history.csv")
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "検証AUC", f"{metrics['validation_auc']:.3f}",
            help="検証期間で3着以内馬を上位に並べる性能です。0.5はランダム相当、1.0に近いほど良好です。0.7以上を一つの目安とし、学習AUCとの差も確認します。",
        )
        c2.metric(
            "テストAUC", f"{metrics['test_auc']:.3f}",
            help="学習にも調整にも使わなかった将来期間の順位判別性能です。検証AUCと同程度で、0.7以上が一つの目安です。",
        )
        c3.metric(
            "テストLog Loss", f"{metrics['test_log_loss']:.3f}",
            help="予測確率の正確さです。小さいほど良く、自信を持った誤予測を強く減点します。同じテスト期間のモデル同士で比較します。",
        )
        c4.metric(
            "最良エポック", str(metrics["best_epoch"]),
            help="検証Lossが最小になった学習回数です。極端に早ければ学習率やモデルの複雑さ、上限付近なら最大エポック数を確認します。",
        )
        c5, c6, c7 = st.columns(3)
        c5.metric(
            "テスト正解率",
            f"{metrics.get('test_accuracy', float('nan')):.3f}",
            help="3着以内／圏外の判定が一致した割合です。圏外馬が多いだけでも高くなるため、他の指標と合わせて評価します。",
        )
        c6.metric(
            "テスト適合率",
            f"{metrics.get('test_precision', float('nan')):.3f}",
            help="モデルが3着以内と判定した馬のうち、実際に3着以内だった割合です。高いほど候補馬の無駄が少ないことを示します。",
        )
        c7.metric(
            "テスト再現率",
            f"{metrics.get('test_recall', float('nan')):.3f}",
            help="実際の3着以内馬をモデルが拾えた割合です。高いほど有力馬の見逃しが少なく、適合率とのバランスを確認します。",
        )

        loss_long = history.melt(
            id_vars="epoch", value_vars=["train_loss", "validation_loss"],
            var_name="series", value_name="loss"
        )
        auc_long = history.melt(
            id_vars="epoch", value_vars=["train_auc", "validation_auc"],
            var_name="series", value_name="auc"
        )
        st.subheader(
            "学習Lossと検証Loss",
            help="両方が下がり近い値で安定する状態が理想です。学習Lossだけ下がり検証Lossが上がる場合は過学習、両方が高止まりする場合は学習不足、激しく上下する場合は学習率過大を疑います。",
        )
        st.plotly_chart(px.line(loss_long, x="epoch", y="loss", color="series"), use_container_width=True)
        st.subheader(
            "学習AUCと検証AUC",
            help="両方が上昇し差が0.08未満なら良好な目安です。差が0.08以上なら軽度、0.15以上なら強い過学習の可能性があります。",
        )
        st.plotly_chart(px.line(auc_long, x="epoch", y="auc", color="series"), use_container_width=True)
        if False:  # 詳細説明は見出しの「？」ヘルプへ移行済み。
            st.markdown(
                """
- **横軸（epoch）**は学習回数、**縦軸（AUC）**は3着以内の馬を圏外の馬より上位に評価できる性能です。AUCは1.0に近いほど良く、0.5はランダム相当です。
- **train_auc（学習AUC）**は学習データでの性能、**validation_auc（検証AUC）**は学習に直接使用していないデータでの性能です。
- **理想的な状態**は、両方のAUCが上昇し、差が小さいまま高い値で安定することです。一つの目安として検証AUCが0.7以上で、学習AUCとの差が0.08未満の状態を確認します。
- 学習AUCだけが上昇し、検証AUCが伸びない、または低下する場合は**過学習**です。特に差が0.15以上ある場合は強い過学習の可能性があります。
- 両方のAUCが0.5付近に留まる場合は、モデルが有効な傾向を十分に学習できていません。特徴量、データ量、学習率、モデルの大きさを見直します。
- 検証AUCは開催時期やデータ件数でも変動するため、1回の上下だけでなく、全体の傾向とテストAUCも合わせて判断します。
                """
            )

        best = history.iloc[int(metrics["best_epoch"]) - 1]
        auc_gap = float(best["train_auc"] - best["validation_auc"])
        last_val_loss = float(history.iloc[-1]["validation_loss"])
        best_val_loss = float(history["validation_loss"].min())
        best_train_auc = float(best["train_auc"])
        config = metrics.get("config", {})
        current_parameters = {
            "最大エポック数": int(config.get("epochs", 80)),
            "バッチサイズ": int(config.get("batch_size", 128)),
            "学習率": float(config.get("learning_rate", 0.001)),
            "隠れ層サイズ": int(config.get("hidden_dim", 64)),
            "Dropout": float(config.get("dropout", 0.25)),
            "Early Stopping待機": int(config.get("patience", 12)),
        }
        recommended_parameters = current_parameters.copy()
        recommendation_reasons: list[str] = []
        if auc_gap >= 0.15 or last_val_loss > best_val_loss * 1.15:
            st.error(
                f"過学習の可能性があります。最良時点の学習AUC−検証AUC={auc_gap:.3f}。"
                "Dropout増加、特徴量削減、データ追加を検討してください。"
            )
            recommended_parameters["Dropout"] = min(
                round(current_parameters["Dropout"] + 0.10, 2),
                0.70,
            )
            hidden_sizes = [32, 64, 128, 256]
            current_hidden_index = hidden_sizes.index(
                current_parameters["隠れ層サイズ"]
            )
            recommended_parameters["隠れ層サイズ"] = hidden_sizes[
                max(0, current_hidden_index - 1)
            ]
            recommended_parameters["Early Stopping待機"] = max(
                5,
                current_parameters["Early Stopping待機"] - 3,
            )
            recommendation_reasons.append(
                "過学習を抑えるため、Dropoutを増やし、"
                "隠れ層と待機回数を小さくします。"
            )
        elif auc_gap >= 0.08:
            st.warning(f"軽度の過学習傾向があります。AUC差={auc_gap:.3f}。")
            recommended_parameters["Dropout"] = min(
                round(current_parameters["Dropout"] + 0.05, 2),
                0.70,
            )
            recommendation_reasons.append(
                "軽度の過学習を抑えるため、Dropoutを少し増やします。"
            )
        else:
            st.success(f"大きな過学習は確認されません。AUC差={auc_gap:.3f}。")

        if best_train_auc < 0.70:
            hidden_sizes = [32, 64, 128, 256]
            current_hidden_index = hidden_sizes.index(
                recommended_parameters["隠れ層サイズ"]
            )
            recommended_parameters["隠れ層サイズ"] = hidden_sizes[
                min(len(hidden_sizes) - 1, current_hidden_index + 1)
            ]
            recommended_parameters["Dropout"] = max(
                round(recommended_parameters["Dropout"] - 0.05, 2),
                0.0,
            )
            recommendation_reasons.append(
                "学習AUCが低いため、隠れ層を増やし、"
                "Dropoutを少し下げて表現力を上げます。"
            )

        best_epoch = int(metrics["best_epoch"])
        if best_epoch <= 3:
            learning_rates = [0.0001, 0.0003, 0.001, 0.003]
            current_rate_index = learning_rates.index(
                current_parameters["学習率"]
            )
            recommended_parameters["学習率"] = learning_rates[
                max(0, current_rate_index - 1)
            ]
            recommendation_reasons.append(
                "最良エポックが非常に早いため、学習率を一段下げて"
                "より緩やかに学習します。"
            )
        elif (
            len(history) >= current_parameters["最大エポック数"]
            and best_epoch >= len(history) * 0.9
        ):
            recommended_parameters["最大エポック数"] = min(
                current_parameters["最大エポック数"] + 20,
                500,
            )
            recommendation_reasons.append(
                "学習終了付近まで改善しているため、"
                "最大エポック数を増やします。"
            )

        if not recommendation_reasons:
            recommendation_reasons.append(
                "大きな問題が見られないため、現在値を維持します。"
            )

        st.subheader("結果を踏まえたパラメータ調整案")
        st.write(" ".join(recommendation_reasons))
        recommendation_rows = [
            {
                "パラメータ": name,
                "現在値": current_parameters[name],
                "推奨値": recommended_parameters[name],
            }
            for name in current_parameters
        ]
        st.dataframe(
            pd.DataFrame(recommendation_rows),
            use_container_width=True,
            hide_index=True,
        )
        if st.button(
            "調整案を上の学習設定に反映",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["pending_training_parameters"] = {
                "training_model_type": (
                    "各馬が3着以内に入る確率を予測する"
                ),
                "training_epochs": int(
                    recommended_parameters["最大エポック数"]
                ),
                "training_batch_size": int(
                    recommended_parameters["バッチサイズ"]
                ),
                "training_learning_rate": float(
                    recommended_parameters["学習率"]
                ),
                "training_hidden_dim": int(
                    recommended_parameters["隠れ層サイズ"]
                ),
                "training_dropout": float(
                    recommended_parameters["Dropout"]
                ),
                "training_patience": int(
                    recommended_parameters["Early Stopping待機"]
                ),
            }
            st.rerun()
        with st.expander("全評価指標"):
            st.json(metrics)

elif page == "レース予想":
    st.header("レース予想")

    historical = load_records("historical")
    runs = model_runs()

    if historical.empty or runs.empty:
        st.warning(
            "過去データと学習済みモデルを用意してください。"
        )
    else:
        prediction_mode = st.radio(
            "予想モード",
            [
                "今後レースを予想",
                "過去レースで予想を検証",
            ],
            horizontal=True,
        )

        model_id = st.selectbox(
            "使用モデル",
            runs["model_run_id"].tolist(),
        )
        selected_model = runs[
            runs["model_run_id"] == model_id
        ].iloc[0]
        model_path = Path(selected_model["model_path"])

        if prediction_mode == "今後レースを予想":
            upcoming = load_records("upcoming")

            if upcoming.empty:
                st.warning(
                    "予想用の出馬表データを取得してください。"
                )
            else:
                available_dates = sorted(
                    pd.to_datetime(upcoming["race_date"], errors="coerce")
                    .dropna().dt.date.unique(),
                    reverse=True,
                )
                prediction_date_labels: dict[date, str] = {}
                for available_date in available_dates:
                    prediction_created, comparison_status = (
                        prediction_date_status(
                            available_date,
                            PATHS.predictions,
                            PATHS.root / "docs",
                        )
                    )
                    statuses: list[str] = []
                    if prediction_created:
                        statuses.append("※予想作成済み")
                    if comparison_status == "completed":
                        statuses.append("※結果照合済み")
                    elif comparison_status == "partial":
                        statuses.append("※一部結果照合済み")
                    suffix = " " + " ".join(statuses) if statuses else ""
                    prediction_date_labels[available_date] = (
                        f"{available_date:%Y年%m月%d日}{suffix}"
                    )
                selected_prediction_date = st.selectbox(
                    "予想対象日",
                    available_dates,
                    format_func=lambda value: prediction_date_labels[value],
                )
                comparison_notice = st.session_state.pop(
                    "prediction_comparison_notice",
                    None,
                )
                if comparison_notice:
                    st.success(comparison_notice)
                races_on_date = upcoming[
                    pd.to_datetime(upcoming["race_date"], errors="coerce")
                    .dt.date.eq(selected_prediction_date)
                ]["race_id"].nunique()
                st.caption(f"対象レース: {races_on_date}レース")
                prediction_date_mask = (
                    pd.to_datetime(upcoming["race_date"], errors="coerce")
                    .dt.date.eq(selected_prediction_date)
                )
                courses_on_date = sorted(
                    upcoming.loc[prediction_date_mask, "course_name"]
                    .dropna().astype(str).unique().tolist()
                )
                st.markdown("#### 競馬場別の馬場状態")
                st.info(
                    "馬場状態の初期値は、JRA-VAN「競馬場別 天気予報」の"
                    "開催当日予報を参考に設定しています。"
                    "予報や実際の馬場発表に応じて手動で変更できます。"
                )
                forecast_by_course: dict[str, dict[str, object]] = {}
                forecast_errors: dict[str, str] = {}
                for course in courses_on_date:
                    try:
                        forecast_by_course[course] = _cached_jravan_weather(
                            course,
                            selected_prediction_date,
                        )
                    except Exception as exc:
                        forecast_errors[course] = str(exc)

                track_condition_options = ["良", "稍重", "重", "不良"]
                condition_columns = st.columns(min(len(courses_on_date), 3))
                track_conditions: dict[str, str] = {}
                for index, course in enumerate(courses_on_date):
                    forecast = forecast_by_course.get(course)
                    default_condition = (
                        str(forecast["track_condition"])
                        if forecast is not None
                        else "良"
                    )
                    column = condition_columns[index % len(condition_columns)]
                    track_conditions[course] = column.selectbox(
                        f"{course}の馬場状態",
                        track_condition_options,
                        index=track_condition_options.index(default_condition),
                        key=(
                            f"track_condition_{selected_prediction_date}_{course}"
                        ),
                        help=(
                            "JRA-VANの開催当日天気予報を「晴・曇→良、"
                            "小雨・小雪→稍重、雨→重、雪→不良」で変換した"
                            "初期値です。必要に応じて変更してください。"
                        ),
                    )
                    if forecast is not None:
                        column.caption(
                            f"JRA-VAN予報: {forecast['weather']} → "
                            f"初期値: {forecast['track_condition']}"
                        )
                if forecast_errors:
                    st.warning(
                        "天気予報を取得できなかった競馬場は「良」を"
                        "初期表示しています: "
                        + "、".join(forecast_errors)
                    )

                if st.button(
                    "この日の全レースを予想",
                    type="primary",
                    key="upcoming_prediction_button",
                ):
                    logger = new_logger()
                    try:
                        logger.info(
                            f"日付一括予想開始: date={selected_prediction_date}, "
                            f"model={model_id}"
                        )
                        with st.spinner(
                            "予想特徴量を生成し、モデルで推論しています。しばらくお待ちください。",
                            show_time=True,
                        ):
                            prediction_input = upcoming.copy()
                            for course, condition in track_conditions.items():
                                course_mask = (
                                    prediction_date_mask
                                    & prediction_input["course_name"].astype(str).eq(course)
                                )
                                prediction_input.loc[
                                    course_mask, "track_condition"
                                ] = condition
                            result, output_path = predict_race_date(
                                historical,
                                prediction_input,
                                selected_prediction_date,
                                model_path,
                            )
                        page_path = None
                        try:
                            page_path = build_prediction_site(
                                result,
                                selected_prediction_date,
                                str(model_id),
                                PATHS.root / "docs",
                            )
                        except Exception as site_exc:
                            logger.exception(f"GitHub Pages用HTML生成失敗: {site_exc}")
                            st.warning(
                                "予想結果は保存しましたが、GitHub Pages用HTMLを"
                                f"生成できませんでした: {site_exc}"
                            )
                        if page_path is not None:
                            st.success(
                                f"{races_on_date}レースの予想結果を保存しました: "
                                f"{output_path}。GitHub Pages用HTMLも生成しました: "
                                f"{page_path}"
                            )
                        st.session_state["latest_upcoming_prediction"] = {
                            "model_id": str(model_id),
                            "race_date": selected_prediction_date,
                            "result": result,
                            "output_path": str(output_path),
                            "track_conditions": track_conditions,
                        }
                        st.session_state.pop("latest_prediction_comparison", None)
                        logger.info("予想完了")
                    except Exception as exc:
                        logger.exception(
                            f"予想失敗: {exc}"
                        )
                        st.exception(exc)

                latest_prediction = st.session_state.get(
                    "latest_upcoming_prediction"
                )
                if not (
                    latest_prediction
                    and latest_prediction.get("race_date")
                    == selected_prediction_date
                    and latest_prediction.get("model_id") == str(model_id)
                ):
                    saved_prediction_path = latest_prediction_file(
                        selected_prediction_date,
                        PATHS.predictions,
                    )
                    if saved_prediction_path is not None:
                        try:
                            saved_prediction = pd.read_parquet(
                                saved_prediction_path
                            )
                            latest_prediction = {
                                "model_id": str(model_id),
                                "race_date": selected_prediction_date,
                                "result": saved_prediction,
                                "output_path": str(saved_prediction_path),
                                "restored": True,
                            }
                            st.session_state[
                                "latest_upcoming_prediction"
                            ] = latest_prediction
                        except Exception as exc:
                            st.warning(
                                "保存済みの予想結果を読み込めませんでした: "
                                f"{exc}"
                            )
                if (
                    latest_prediction
                    and latest_prediction.get("race_date") == selected_prediction_date
                    and latest_prediction.get("model_id") == str(model_id)
                ):
                    result = latest_prediction["result"]
                    top3_list = result[result["prediction_rank"] <= 3].copy()
                    display = top3_list.copy()
                    display["top3_probability"] = display["top3_probability"].map(
                        lambda value: f"{value:.1%}"
                    )
                    display["expected_value_index"] = display[
                        "expected_value_index"
                    ].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
                    list_columns = [
                        "course_name", "race_number", "race_name",
                        "prediction_rank", "horse_number", "horse_name",
                        "top3_probability", "odds", "expected_value_index",
                        "jockey_name",
                    ]
                    st.subheader("レース別 予測上位3頭")
                    st.dataframe(
                        display[list_columns].rename(columns=_PREDICTION_COLUMN_LABELS),
                        use_container_width=True,
                        hide_index=True,
                    )
                    with st.expander("全出走馬の予想を見る"):
                        full_display = result.copy()
                        full_display["top3_probability"] = full_display[
                            "top3_probability"
                        ].map(lambda value: f"{value:.1%}")
                        full_display["expected_value_index"] = full_display[
                            "expected_value_index"
                        ].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
                        st.dataframe(
                            full_display[list_columns].rename(
                                columns=_PREDICTION_COLUMN_LABELS
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                    if st.button(
                        "この日の全レースを確定結果と比較",
                        type="secondary",
                        key=f"compare_prediction_date_{selected_prediction_date}_{model_id}",
                    ):
                        logger = new_logger()
                        try:
                            with st.spinner(
                                "この日の全レース結果を取得して比較しています。",
                                show_time=True,
                            ):
                                result_scraper = NetkeibaDatabaseCreator(logger=logger)
                                comparison, comparison_summary = compare_prediction_date(
                                    result,
                                    selected_prediction_date,
                                    lambda race_id, race_date: (
                                        result_scraper.fetch_result_for_comparison(
                                            race_id,
                                            race_date,
                                            force=True,
                                        )
                                    ),
                                )
                            comparison_page_path = None
                            try:
                                comparison_page_path = build_prediction_site(
                                    result,
                                    selected_prediction_date,
                                    str(model_id),
                                    PATHS.root / "docs",
                                    comparison=comparison,
                                    comparison_summary=comparison_summary,
                                )
                            except Exception as site_exc:
                                logger.exception(
                                    f"比較内容のGitHub Pages用HTML反映失敗: {site_exc}"
                                )
                                st.warning(
                                    "確定結果との比較は完了しましたが、比較内容を"
                                    "GitHub Pages用HTMLへ反映できませんでした: "
                                    f"{site_exc}"
                                )
                            st.session_state["latest_prediction_comparison"] = {
                                "race_date": selected_prediction_date,
                                "model_id": str(model_id),
                                "result": comparison,
                                "summary": comparison_summary,
                            }
                            if comparison_page_path is not None:
                                st.session_state["prediction_comparison_notice"] = (
                                    "確定結果との比較が完了し、比較内容を"
                                    "GitHub Pages用HTMLへ反映しました: "
                                    f"{comparison_page_path}"
                                )
                                st.rerun()
                        except Exception as exc:
                            logger.exception(f"日付一括予想結果比較失敗: {exc}")
                            st.exception(exc)

                    comparison_state = st.session_state.get(
                        "latest_prediction_comparison"
                    )
                    if (
                        comparison_state
                        and comparison_state.get("race_date")
                        == selected_prediction_date
                        and comparison_state.get("model_id") == str(model_id)
                    ):
                        comparison_summary = comparison_state["summary"]
                        metric1, metric2, metric3, metric4 = st.columns(4)
                        metric1.metric(
                            "比較完了レース",
                            f"{comparison_summary['compared_races']} / "
                            f"{comparison_summary['requested_races']}",
                        )
                        metric2.metric(
                            "予測上位3頭の的中",
                            f"{comparison_summary['top3_hit_count']}頭",
                        )
                        metric3.metric(
                            "上位3頭完全的中",
                            f"{comparison_summary['perfect_top3_races']}レース",
                        )
                        metric4.metric(
                            "未比較レース",
                            f"{comparison_summary['failed_races']}レース",
                        )
                        failures = comparison_summary.get("failures", [])
                        if failures:
                            st.warning(
                                "結果未確定または取得失敗: "
                                + " / ".join(
                                    f"{item['race_id']} ({item['message']})"
                                    for item in failures
                                )
                            )
                        comparison_display = comparison_state["result"].copy()
                        if comparison_display.empty:
                            st.info("比較できる確定結果がまだありません。")
                        else:
                            comparison_display["top3_probability"] = comparison_display[
                                "top3_probability"
                            ].map(lambda value: f"{value:.1%}")
                            comparison_display["expected_value_index"] = comparison_display[
                                "expected_value_index"
                            ].map(lambda value: f"{value:.2f}" if pd.notna(value) else "-")
                            for column in ("predicted_top3", "actual_top3"):
                                comparison_display[column] = comparison_display[column].map(
                                    {True: "○", False: ""}
                                )
                            comparison_display["top3_hit"] = comparison_display[
                                "top3_hit"
                            ].map({True: "的中", False: ""})
                            comparison_columns = [
                                "course_name", "race_number", "race_name",
                                "prediction_rank", "horse_number", "horse_name",
                                "top3_probability", "finish_position", "predicted_top3",
                                "actual_top3", "top3_hit", "odds", "popularity",
                                "expected_value_index", "jockey_name",
                            ]
                            st.subheader("開催日一括：予想結果と確定着順の比較")
                            st.dataframe(
                                comparison_display[
                                    [
                                        column for column in comparison_columns
                                        if column in comparison_display
                                    ]
                                ].rename(columns=_PREDICTION_COLUMN_LABELS),
                                use_container_width=True,
                                hide_index=True,
                            )

        else:
            st.info(
                "対象レースより前の履歴だけで特徴量を作り、"
                "予測結果と実際の着順を比較します。"
            )

            race_options = (
                historical[
                    [
                        "race_id",
                        "race_date",
                        "course_name",
                        "race_number",
                        "race_name",
                    ]
                ]
                .drop_duplicates("race_id")
                .dropna(subset=["race_date"])
                .sort_values(
                    [
                        "race_date",
                        "course_name",
                        "race_number",
                    ],
                    ascending=[False, True, True],
                )
            )

            labels = {
                str(row.race_id): (
                    f"{row.race_date} "
                    f"{row.course_name}"
                    f"{int(row.race_number) if pd.notna(row.race_number) else ''}R "
                    f"{row.race_name} ({row.race_id})"
                )
                for row in race_options.itertuples()
            }

            race_id = st.selectbox(
                "検証する過去レース",
                race_options["race_id"].astype(str).tolist(),
                format_func=lambda x: labels.get(x, x),
            )

            try:
                metrics_path = model_path / "metrics.json"
                model_metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                test_range = (
                    model_metrics
                    .get("date_ranges", {})
                    .get("test", [])
                )
                if len(test_range) == 2:
                    selected_date = pd.to_datetime(
                        race_options.loc[
                            race_options["race_id"].astype(str)
                            == str(race_id),
                            "race_date",
                        ].iloc[0]
                    )
                    test_start = pd.to_datetime(test_range[0])
                    test_end = pd.to_datetime(test_range[1])
                    if not (
                        test_start
                        <= selected_date
                        <= test_end
                    ):
                        st.warning(
                            "このレースは選択モデルのテスト期間外です。"
                            "モデル学習時に使用された可能性があるため、"
                            "参考値として確認してください。"
                        )
                    else:
                        st.success(
                            "このレースはモデルのテスト期間内です。"
                            "未見データに近い条件で確認できます。"
                        )
            except Exception:
                st.caption(
                    "モデルのテスト期間を確認できませんでした。"
                )

            if st.button(
                "過去レース予想を実行",
                type="primary",
                key="historical_prediction_button",
            ):
                logger = new_logger()
                try:
                    logger.info(
                        f"過去レース検証開始: "
                        f"race_id={race_id}, model={model_id}"
                    )
                    with st.spinner(
                        "対象レース時点の特徴量を生成し、過去予想を検証しています。",
                        show_time=True,
                    ):
                        result, output_path, summary = (
                            predict_historical_race(
                                historical,
                                race_id,
                                model_path,
                            )
                        )

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "予測上位3頭の的中数",
                        f"{summary['top3_hit_count']} / 3",
                    )
                    col2.metric(
                        "予測1位馬の実着順",
                        summary[
                            "prediction_1_actual_finish"
                        ]
                        or "-",
                    )
                    col3.metric(
                        "実勝馬の予測順位",
                        summary[
                            "winner_prediction_rank"
                        ]
                        or "-",
                    )
                    col4.metric(
                        "使用した過去履歴",
                        f"{summary['history_rows_used']:,}行",
                    )

                    display = result.copy()
                    display["top3_probability"] = (
                        display["top3_probability"]
                        .map(lambda x: f"{x:.1%}")
                    )
                    display["expected_value_index"] = (
                        display["expected_value_index"]
                        .map(
                            lambda x: (
                                f"{x:.2f}"
                                if pd.notna(x)
                                else "-"
                            )
                        )
                    )
                    display["predicted_top3"] = (
                        display["predicted_top3"]
                        .map({True: "○", False: ""})
                    )
                    display["actual_top3"] = (
                        display["actual_top3"]
                        .map({True: "○", False: ""})
                    )
                    display["top3_hit"] = (
                        display["top3_hit"]
                        .map({True: "的中", False: ""})
                    )

                    preferred_columns = [
                        "prediction_rank",
                        "horse_number",
                        "horse_name",
                        "top3_probability",
                        "odds",
                        "popularity",
                        "finish_position",
                        "predicted_top3",
                        "actual_top3",
                        "top3_hit",
                        "expected_value_index",
                        "jockey_name",
                    ]
                    display_columns = [
                        column
                        for column in preferred_columns
                        if column in display.columns
                    ]

                    st.success(
                        f"検証結果を保存しました: {output_path}"
                    )
                    st.dataframe(
                        display[display_columns].rename(
                            columns=_PREDICTION_COLUMN_LABELS
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_bytes = result.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        "比較結果CSVをダウンロード",
                        csv_bytes,
                        file_name=(
                            f"historical_prediction_"
                            f"{race_id}.csv"
                        ),
                        mime="text/csv",
                    )
                    logger.info("過去レース検証完了")
                except Exception as exc:
                    logger.exception(
                        f"過去レース検証失敗: {exc}"
                    )
                    st.exception(exc)

elif page == "ログ・保存結果":
    st.header("ログ・保存結果")
    log_files = sorted(PATHS.logs.glob("*.log"), reverse=True)
    if log_files:
        selected_log = st.selectbox("ログファイル", log_files, format_func=lambda p: p.name)
        text = selected_log.read_text(encoding="utf-8", errors="replace")
        st.code("\n".join(text.splitlines()[-500:]), language="text")
    else:
        st.info("ログファイルはまだありません。")
    st.subheader("データ収集履歴")
    runs = collection_runs()
    st.dataframe(runs, use_container_width=True, hide_index=True)
    st.subheader("学習モデル履歴")
    st.dataframe(model_runs(), use_container_width=True, hide_index=True)

elif page == "メンテナンス":
    st.header("メンテナンス")
    st.warning(
        "削除したデータは元に戻せません。"
        "削除対象と削除方法を確認してから実行してください。"
    )

    maintenance_tab = st.radio(
        "削除対象",
        [
            "収集データ",
            "ダミーデータ",
            "特徴量データ",
            "学習結果",
            "取得HTML",
            "ログファイル",
        ],
        horizontal=True,
    )

    if maintenance_tab == "収集データ":
        st.info(
            "収集データの削除では、スクレイピング結果と、"
            "保存済みHTMLから作成したDBデータの両方を削除します。"
            "取得済みHTMLは削除しません。"
        )
    elif maintenance_tab == "ダミーデータ":
        st.info(
            "ダミーデータの削除では、選択したダミー生成実行に"
            "関連する学習用・予想用データだけを削除します。"
            "HTMLから作成した実データと取得済みHTMLは残ります。"
            "「まとめて削除」を選んだ場合も、削除対象は"
            "ダミーデータだけです。"
        )
    elif maintenance_tab == "取得HTML":
        st.info(
            "取得HTMLの削除では、保存済みHTMLのみ削除します。"
            "DB上の収集データは削除しません。"
        )
    elif maintenance_tab == "特徴量データ":
        st.info(
            "特徴量データのみ削除します。元レースデータ、取得HTML、"
            "学習済みモデルは変更しません。削除後に学習する場合は、"
            "前準備から特徴量を再生成してください。"
        )

    delete_mode = st.radio(
        "削除方法",
        ["個別に削除", "まとめて削除"],
        horizontal=True,
        key=f"delete_mode_{maintenance_tab}",
    )

    if maintenance_tab in {"収集データ", "ダミーデータ"}:
        runs = collection_runs()

        if runs.empty:
            st.info("削除できるデータ収集履歴がありません。")
        else:
            source_names = (
                {"netkeiba", "cached_html"}
                if maintenance_tab == "収集データ"
                else {"generator"}
            )
            target_runs = runs[
                runs["source"].astype(str).isin(source_names)
            ].copy()

            if target_runs.empty:
                st.info(f"{maintenance_tab}の履歴がありません。")
            else:
                target_runs = target_runs.sort_values(
                    "started_at",
                    ascending=False,
                )

                run_ids = target_runs["run_id"].astype(str).tolist()
                labels = {}
                for row in target_runs.itertuples():
                    started_at = getattr(row, "started_at", "")
                    dataset_type = getattr(row, "dataset_type", "")
                    row_count = getattr(row, "row_count", 0)
                    labels[str(row.run_id)] = (
                        f"{started_at} / {dataset_type} / "
                        f"{row_count}行 / {row.run_id}"
                    )

                if delete_mode == "個別に削除":
                    selected_run_id = st.selectbox(
                        "削除する実行",
                        run_ids,
                        format_func=lambda value: labels.get(value, value),
                    )
                    selected_run = target_runs[
                        target_runs["run_id"].astype(str)
                        == selected_run_id
                    ].iloc[0]

                    st.subheader("削除対象の確認")
                    display_columns = [
                        column
                        for column in [
                            "run_id",
                            "started_at",
                            "completed_at",
                            "status",
                            "dataset_type",
                            "source",
                            "start_date",
                            "end_date",
                            "race_count",
                            "row_count",
                            "output_path",
                        ]
                        if column in selected_run.index
                    ]
                    st.dataframe(
                        selected_run[display_columns].to_frame().T,
                        use_container_width=True,
                        hide_index=True,
                    )

                    output_paths = _split_output_paths(
                        selected_run.get("output_path")
                    )
                    race_ids = _read_race_ids_from_files(output_paths)
                    st.write(
                        f"確認できた関連レースID: **{len(race_ids):,}件**"
                    )
                    st.write(
                        "HTML保存先:",
                        "data/raw_html/historical/<年>/ または "
                        "data/raw_html/upcoming/<年>/"
                        "（実行履歴とは独立して保持）",
                    )

                    confirm = st.checkbox(
                        "選択した実行に関連するデータを削除する",
                        key=f"confirm_collection_single_{maintenance_tab}",
                    )

                    if st.button(
                        f"{maintenance_tab}を削除",
                        type="primary",
                        disabled=not confirm,
                    ):
                        result = _delete_collection_run(selected_run)
                        _clear_ui_state()

                        if result["failed_paths"]:
                            st.error("一部ファイルを削除できませんでした。")
                            st.code(
                                "\n".join(result["failed_paths"]),
                                language="text",
                            )
                        else:
                            st.success(
                                f"{selected_run_id} を削除しました。"
                            )

                        with st.expander("削除結果", expanded=True):
                            st.json(result)

                else:
                    st.error(
                        f"{maintenance_tab}をすべて削除します。"
                        f"対象実行数: {len(target_runs):,}件"
                    )
                    st.dataframe(
                        target_runs,
                        use_container_width=True,
                        hide_index=True,
                    )

                    confirm = True
                    if maintenance_tab != "収集データ":
                        confirm = st.checkbox(
                            f"{maintenance_tab}をすべて削除する",
                            key=f"confirm_collection_bulk_{maintenance_tab}",
                        )
                    confirmation_word = st.text_input(
                        "確認のため「全削除」と入力",
                        key=f"typed_collection_bulk_{maintenance_tab}",
                    )

                    if st.button(
                        f"{maintenance_tab}をまとめて削除",
                        type="primary",
                        disabled=(
                            not confirm
                            or confirmation_word.strip() != "全削除"
                        ),
                    ):
                        results = []
                        failed_items = []

                        for _, selected_run in target_runs.iterrows():
                            try:
                                results.append(
                                    _delete_collection_run(selected_run)
                                )
                            except Exception as exc:
                                failed_items.append(
                                    f"{selected_run['run_id']}: {exc}"
                                )

                        _clear_ui_state()

                        if failed_items:
                            st.error(
                                "一部の実行データを削除できませんでした。"
                            )
                            st.code(
                                "\n".join(failed_items),
                                language="text",
                            )
                        else:
                            st.success(
                                f"{len(results):,}件の実行データを"
                                "まとめて削除しました。"
                            )

                        with st.expander("削除結果", expanded=True):
                            st.json(results)

    elif maintenance_tab == "特徴量データ":
        feature_targets = {
            ("baseline", "1.1.0"),
            ("recent_speed", "1.0.0"),
            ("race_entry", "1.1.0"),
            ("speed_index", "1.0.0"),
        }
        stored_feature_runs = feature_runs()
        if not stored_feature_runs.empty:
            feature_targets.update(
                (
                    str(row.feature_set_name),
                    str(row.feature_set_version),
                )
                for row in stored_feature_runs[
                    ["feature_set_name", "feature_set_version"]
                ].dropna().drop_duplicates().itertuples(index=False)
            )

        feature_labels = {
            "baseline": "基礎近走成績",
            "recent_speed": "近走スピード成績",
            "race_entry": "出馬表基本条件・馬場状態",
            "speed_index": "1走単位スピード指数",
        }
        target_rows = []
        for name, version in sorted(feature_targets):
            if name == "speed_index":
                summary = performance_feature_summary(name, version)
                row_count = int(summary["row_count"])
                storage_type = "performance"
            else:
                summary = feature_store_summary(name, version)
                row_count = int(summary["row_count"])
                storage_type = "race"
            target_rows.append({
                "表示名": feature_labels.get(name, name),
                "識別子": f"{name}:{version}",
                "保存行数": row_count,
                "name": name,
                "version": version,
                "storage_type": storage_type,
            })
        feature_target_frame = pd.DataFrame(target_rows)
        st.dataframe(
            feature_target_frame[["表示名", "識別子", "保存行数"]],
            use_container_width=True,
            hide_index=True,
        )

        feature_jobs_active = any(
            job.get("status") in {"running", "cancelling"}
            for jobs in (
                list_feature_generation_jobs(),
                list_speed_index_jobs(),
                list_recent_speed_jobs(),
                list_race_entry_jobs(),
            )
            for job in jobs
        )
        if feature_jobs_active:
            st.warning(
                "特徴量生成が実行中のため、完了または中止するまで削除できません。"
            )

        if delete_mode == "個別に削除":
            identifiers = feature_target_frame["識別子"].tolist()
            selected_identifier = st.selectbox(
                "削除する特徴量",
                identifiers,
                format_func=lambda identifier: (
                    lambda row: (
                        f"{row['表示名']}（{identifier}） / "
                        f"{int(row['保存行数']):,}行"
                    )
                )(
                    feature_target_frame[
                        feature_target_frame["識別子"].eq(identifier)
                    ].iloc[0]
                ),
            )
            selected_feature = feature_target_frame[
                feature_target_frame["識別子"].eq(selected_identifier)
            ].iloc[0]
            confirm = st.checkbox(
                "選択した特徴量データを削除する",
                key="confirm_feature_single",
            )
            if st.button(
                "特徴量データを削除",
                type="primary",
                disabled=not confirm or feature_jobs_active,
            ):
                if selected_feature["storage_type"] == "performance":
                    deleted = clear_performance_features(
                        selected_feature["name"], selected_feature["version"]
                    )
                else:
                    deleted = clear_features(
                        selected_feature["name"], selected_feature["version"]
                    )
                st.success(
                    f"{selected_identifier} の特徴量を{deleted:,}行削除しました。"
                )
                st.rerun()
        else:
            stored_rows = int(feature_target_frame["保存行数"].sum())
            st.error(
                "すべての特徴量データを削除します。"
                f"対象: {len(feature_target_frame):,}セット / {stored_rows:,}行"
            )
            confirm = st.checkbox(
                "特徴量データをすべて削除する",
                key="confirm_feature_bulk",
            )
            confirmation_word = st.text_input(
                "確認のため「全削除」と入力",
                key="typed_feature_bulk",
            )
            if st.button(
                "特徴量データをまとめて削除",
                type="primary",
                disabled=(
                    not confirm
                    or confirmation_word.strip() != "全削除"
                    or feature_jobs_active
                ),
            ):
                deleted_total = 0
                for row in feature_target_frame.itertuples(index=False):
                    if row.storage_type == "performance":
                        deleted_total += clear_performance_features(
                            row.name, row.version
                        )
                    else:
                        deleted_total += clear_features(row.name, row.version)
                st.success(
                    f"特徴量データを合計{deleted_total:,}行削除しました。"
                )
                st.rerun()

    elif maintenance_tab == "学習結果":
        runs = model_runs()

        if runs.empty:
            st.info("削除できる学習結果がありません。")
        else:
            sort_column = (
                "created_at"
                if "created_at" in runs.columns
                else runs.columns[0]
            )
            runs = runs.sort_values(
                sort_column,
                ascending=False,
            )

            if delete_mode == "個別に削除":
                model_ids = runs["model_run_id"].astype(str).tolist()
                labels = {
                    str(row.model_run_id): (
                        f"{getattr(row, 'created_at', '')} / "
                        f"{row.model_run_id}"
                    )
                    for row in runs.itertuples()
                }

                selected_model_id = st.selectbox(
                    "削除するモデル",
                    model_ids,
                    format_func=lambda value: labels.get(value, value),
                )
                selected_model = runs[
                    runs["model_run_id"].astype(str)
                    == selected_model_id
                ].iloc[0]

                st.dataframe(
                    selected_model.to_frame().T,
                    use_container_width=True,
                    hide_index=True,
                )

                confirm = st.checkbox(
                    "選択した学習結果を削除する",
                    key="confirm_model_single",
                )

                if st.button(
                    "学習結果を削除",
                    type="primary",
                    disabled=not confirm,
                ):
                    result = _delete_model_run(selected_model)
                    _clear_ui_state()

                    if result["failed_paths"]:
                        st.error("一部ファイルを削除できませんでした。")
                        st.code(
                            "\n".join(result["failed_paths"]),
                            language="text",
                        )
                    else:
                        st.success(
                            f"{selected_model_id} を削除しました。"
                        )

                    with st.expander("削除結果", expanded=True):
                        st.json(result)

            else:
                st.error(
                    f"学習結果をすべて削除します。"
                    f"対象モデル数: {len(runs):,}件"
                )
                st.dataframe(
                    runs,
                    use_container_width=True,
                    hide_index=True,
                )

                confirm = st.checkbox(
                    "学習結果をすべて削除する",
                    key="confirm_model_bulk",
                )
                confirmation_word = st.text_input(
                    "確認のため「全削除」と入力",
                    key="typed_model_bulk",
                )

                if st.button(
                    "学習結果をまとめて削除",
                    type="primary",
                    disabled=(
                        not confirm
                        or confirmation_word.strip() != "全削除"
                    ),
                ):
                    results = []
                    failed_items = []

                    for _, selected_model in runs.iterrows():
                        try:
                            results.append(
                                _delete_model_run(selected_model)
                            )
                        except Exception as exc:
                            failed_items.append(
                                f"{selected_model['model_run_id']}: {exc}"
                            )

                    _clear_ui_state()

                    if failed_items:
                        st.error(
                            "一部の学習結果を削除できませんでした。"
                        )
                        st.code(
                            "\n".join(failed_items),
                            language="text",
                        )
                    else:
                        st.success(
                            f"{len(results):,}件の学習結果を"
                            "まとめて削除しました。"
                        )

                    with st.expander("削除結果", expanded=True):
                        st.json(results)

    elif maintenance_tab == "取得HTML":
        raw_html_root = Path(PATHS.raw_html)
        run_directories = (
            sorted(
                [
                    year_path
                    for category_path in (
                        PATHS.historical_html,
                        PATHS.upcoming_html,
                    )
                    if category_path.exists()
                    for year_path in category_path.iterdir()
                    if year_path.is_dir()
                ],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if raw_html_root.exists()
            else []
        )

        if not run_directories:
            st.info("削除できる取得HTMLがありません。")
        elif delete_mode == "個別に削除":
            selected_directories = st.multiselect(
                "削除する実行フォルダ",
                run_directories,
                format_func=lambda path: str(
                    path.relative_to(raw_html_root)
                ),
            )

            total_files = sum(
                1
                for directory in selected_directories
                for path in directory.rglob("*")
                if path.is_file()
            )
            st.write(
                f"選択フォルダ: **{len(selected_directories)}件** / "
                f"ファイル: **{total_files:,}件**"
            )

            confirm = st.checkbox(
                "選択した取得HTMLを削除する",
                key="confirm_html_single",
            )

            if st.button(
                "取得HTMLを削除",
                type="primary",
                disabled=not confirm or not selected_directories,
            ):
                deleted, failed = _remove_paths(
                    list(selected_directories)
                )
                if failed:
                    st.error("一部フォルダを削除できませんでした。")
                    st.code("\n".join(failed), language="text")
                else:
                    st.success(
                        f"{len(deleted)}フォルダを削除しました。"
                    )
        else:
            total_files = sum(
                1
                for directory in run_directories
                for path in directory.rglob("*")
                if path.is_file()
            )
            st.error(
                f"取得HTMLをすべて削除します。"
                f"対象フォルダ: {len(run_directories):,}件 / "
                f"ファイル: {total_files:,}件"
            )

            confirm = st.checkbox(
                "取得HTMLをすべて削除する",
                key="confirm_html_bulk",
            )
            confirmation_word = st.text_input(
                "確認のため「全削除」と入力",
                key="typed_html_bulk",
            )

            if st.button(
                "取得HTMLをまとめて削除",
                type="primary",
                disabled=(
                    not confirm
                    or confirmation_word.strip() != "全削除"
                ),
            ):
                deleted, failed = _remove_paths(run_directories)
                if failed:
                    st.error("一部フォルダを削除できませんでした。")
                    st.code("\n".join(failed), language="text")
                else:
                    st.success(
                        f"{len(deleted):,}フォルダを"
                        "まとめて削除しました。"
                    )

    elif maintenance_tab == "ログファイル":
        log_files = sorted(
            Path(PATHS.logs).glob("*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not log_files:
            st.info("削除できるログファイルがありません。")
        elif delete_mode == "個別に削除":
            selected_logs = st.multiselect(
                "削除するログ",
                log_files,
                format_func=lambda path: (
                    f"{path.name} / "
                    f"{path.stat().st_size / 1024:.1f} KB"
                ),
            )

            confirm = st.checkbox(
                "選択したログファイルを削除する",
                key="confirm_log_single",
            )

            if st.button(
                "ログファイルを削除",
                type="primary",
                disabled=not confirm or not selected_logs,
            ):
                deleted, failed = _remove_paths(list(selected_logs))
                _clear_ui_state()

                if failed:
                    st.error("一部ログを削除できませんでした。")
                    st.code("\n".join(failed), language="text")
                else:
                    st.success(
                        f"{len(deleted)}ファイルを削除しました。"
                    )
        else:
            total_size = sum(path.stat().st_size for path in log_files)
            st.error(
                f"ログファイルをすべて削除します。"
                f"対象: {len(log_files):,}件 / "
                f"{total_size / 1024 / 1024:.2f} MB"
            )

            confirm = st.checkbox(
                "ログファイルをすべて削除する",
                key="confirm_log_bulk",
            )
            confirmation_word = st.text_input(
                "確認のため「全削除」と入力",
                key="typed_log_bulk",
            )

            if st.button(
                "ログファイルをまとめて削除",
                type="primary",
                disabled=(
                    not confirm
                    or confirmation_word.strip() != "全削除"
                ),
            ):
                deleted, failed = _remove_paths(log_files)
                _clear_ui_state()

                if failed:
                    st.error("一部ログを削除できませんでした。")
                    st.code("\n".join(failed), language="text")
                else:
                    st.success(
                        f"{len(deleted):,}ファイルを"
                        "まとめて削除しました。"
                    )
