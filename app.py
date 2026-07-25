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

from src.config import PATHS
from src.demo_data import generate_demo_records
from src.html_collection_jobs import cancel_job, list_jobs, start_job
from src.logging_utils import AppLogger
from src.modeling import TrainConfig, predict_historical_race, predict_race, train_model
from src.scrapers_netkeiba import NetkeibaScraper
from src.storage import (
    collection_runs,
    dashboard_summary,
    load_records,
    model_runs,
    save_collection_run,
    save_race_records,
)


st.set_page_config(page_title="競馬予想AI", page_icon="🏇", layout="wide")
st.title("🏇 競馬予想AIシステム")
st.caption("私的利用向けMVP：データ収集・保存・学習・過学習確認・週末予想を個別実行できます。")

if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = []

if "scraping_urls" not in st.session_state:
    st.session_state.scraping_urls = []


_URL_PATTERN = re.compile(r"https?://[^\s<>'\"\]\)]+", re.IGNORECASE)


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


with st.sidebar:
    page = st.radio(
        "メニュー",
        ["ダッシュボード", "HTML収集（学習用）", "HTML収集（予想用）", "データベース作成", "モデル学習", "学習評価", "週末予想", "ログ・保存結果", "メンテナンス"],
    )
    st.divider()
    st.caption(f"DB: {PATHS.database}")
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
        tomorrow = date.today() + timedelta(days=1)
        prediction_date_limit = date.today() + timedelta(
            days=7
        )
        target_date = st.date_input(
            "取得日",
            value=tomorrow,
            min_value=tomorrow,
            max_value=prediction_date_limit,
        )
        start_date = target_date
        end_date = target_date
        st.caption(
            f"{target_date}の予想用出馬表HTMLを取得します。"
            f"選択可能期間: {tomorrow} ～ "
            f"{prediction_date_limit}"
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
    mode = st.radio("作成方法", ["デモデータを生成", "取得済みHTMLから作成"], horizontal=True)

    if mode == "デモデータを生成":
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
    else:
        dataset_type_label = st.radio(
            "対象", ["学習用の過去結果", "予想用の出馬表"], horizontal=True
        )
        dataset_type = (
            "historical"
            if dataset_type_label == "学習用の過去結果"
            else "upcoming"
        )
        if dataset_type == "historical":
            selected_year = st.selectbox(
                "対象年",
                list(range(date.today().year, 1985, -1)),
                key="database_year",
            )
            start_date = date(int(selected_year), 1, 1)
            end_date = min(date(int(selected_year), 12, 31), date.today())
        else:
            target_date = st.date_input(
                "対象日", value=date.today(), key="database_date"
            )
            start_date = target_date
            end_date = target_date
        st.caption("ネットワークアクセスは行わず、取得済みHTMLだけを解析します。")
        if st.button("取得済みHTMLからデータベースを作成", type="primary"):
            run_id = f"database_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            started = datetime.now()
            logger = new_logger()
            scraper = NetkeibaScraper(logger=logger)
            progress_bar = st.progress(0.0, text="解析中")
            try:
                frame = scraper.parse_cached_date_range(
                    start_date=start_date,
                    end_date=end_date,
                    dataset_type=dataset_type,
                    progress_callback=lambda value: progress_bar.progress(
                        min(max(float(value), 0.0), 1.0),
                        text=f"解析中 {float(value):.0%}",
                    ),
                )
                if frame.empty:
                    raise ValueError(
                        "対象期間の解析可能なHTMLがありません。先にHTML収集を実行してください。"
                    )
                save_race_records(frame, run_id, dataset_type)
                save_collection_run(
                    {
                        "run_id": run_id,
                        "started_at": started,
                        "completed_at": datetime.now(),
                        "status": "completed",
                        "dataset_type": dataset_type,
                        "source": "cached_html",
                        "start_date": start_date,
                        "end_date": end_date,
                        "race_count": frame["race_id"].nunique(),
                        "row_count": len(frame),
                        "output_path": "",
                        "message": "取得済みHTMLからデータベース作成完了",
                    }
                )
                progress_bar.progress(1.0, text="完了")
                st.success(
                    f"{frame['race_id'].nunique()}レース、{len(frame)}行を保存しました。"
                )
                st.caption(f"保存先: {PATHS.database}")
            except Exception as exc:
                logger.exception(f"データベース作成失敗: {exc}")
                progress_bar.empty()
                save_collection_run(
                    {
                        "run_id": run_id,
                        "started_at": started,
                        "completed_at": datetime.now(),
                        "status": "failed",
                        "dataset_type": dataset_type,
                        "source": "cached_html",
                        "start_date": start_date,
                        "end_date": end_date,
                        "race_count": 0,
                        "row_count": 0,
                        "output_path": "",
                        "message": str(exc),
                    }
                )
                st.exception(exc)

elif page == "モデル学習":
    st.header("ディープラーニングモデル学習")
    historical = load_records("historical")
    st.write(f"利用可能な過去データ: **{len(historical):,}行 / {historical['race_id'].nunique() if not historical.empty else 0:,}レース**")
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("最大エポック数", min_value=10, max_value=500, value=80, step=10)
    batch_size = c2.selectbox("バッチサイズ", [32, 64, 128, 256], index=2)
    learning_rate = c3.selectbox("学習率", [0.0001, 0.0003, 0.001, 0.003], index=2)
    c4, c5, c6 = st.columns(3)
    hidden_dim = c4.selectbox("隠れ層サイズ", [32, 64, 128, 256], index=1)
    dropout = c5.slider("Dropout", 0.0, 0.7, 0.25, 0.05)
    patience = c6.number_input("Early Stopping待機", 3, 50, 12)
    st.caption("開催日順に70%／15%／15%へ分割し、未来データが学習側へ混ざらないようにします。")
    if st.button("モデル学習を開始", type="primary", disabled=historical.empty):
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

elif page == "学習評価":
    st.header("学習結果・過学習チェック")
    runs = model_runs()
    if runs.empty:
        st.warning("学習済みモデルがありません。")
    else:
        selected_id = st.selectbox("モデル", runs["model_run_id"].tolist())
        selected = runs[runs["model_run_id"] == selected_id].iloc[0]
        model_dir = Path(selected["model_path"])
        history = pd.read_csv(model_dir / "training_history.csv")
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("検証AUC", f"{metrics['validation_auc']:.3f}")
        c2.metric("テストAUC", f"{metrics['test_auc']:.3f}")
        c3.metric("テストLog Loss", f"{metrics['test_log_loss']:.3f}")
        c4.metric("最良エポック", str(metrics["best_epoch"]))

        loss_long = history.melt(
            id_vars="epoch", value_vars=["train_loss", "validation_loss"],
            var_name="series", value_name="loss"
        )
        auc_long = history.melt(
            id_vars="epoch", value_vars=["train_auc", "validation_auc"],
            var_name="series", value_name="auc"
        )
        st.plotly_chart(px.line(loss_long, x="epoch", y="loss", color="series", title="学習Lossと検証Loss"), use_container_width=True)
        st.plotly_chart(px.line(auc_long, x="epoch", y="auc", color="series", title="学習AUCと検証AUC"), use_container_width=True)

        best = history.iloc[int(metrics["best_epoch"]) - 1]
        auc_gap = float(best["train_auc"] - best["validation_auc"])
        last_val_loss = float(history.iloc[-1]["validation_loss"])
        best_val_loss = float(history["validation_loss"].min())
        if auc_gap >= 0.15 or last_val_loss > best_val_loss * 1.15:
            st.error(
                f"過学習の可能性があります。最良時点の学習AUC−検証AUC={auc_gap:.3f}。"
                "Dropout増加、特徴量削減、データ追加を検討してください。"
            )
        elif auc_gap >= 0.08:
            st.warning(f"軽度の過学習傾向があります。AUC差={auc_gap:.3f}。")
        else:
            st.success(f"大きな過学習は確認されません。AUC差={auc_gap:.3f}。")
        with st.expander("全評価指標"):
            st.json(metrics)

elif page == "週末予想":
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
                race_options = (
                    upcoming[
                        [
                            "race_id",
                            "race_date",
                            "course_name",
                            "race_number",
                            "race_name",
                        ]
                    ]
                    .drop_duplicates("race_id")
                    .sort_values(
                        [
                            "race_date",
                            "course_name",
                            "race_number",
                        ]
                    )
                )
                labels = {
                    row.race_id: (
                        f"{row.race_date} "
                        f"{row.course_name}"
                        f"{int(row.race_number) if pd.notna(row.race_number) else ''}R "
                        f"{row.race_name} ({row.race_id})"
                    )
                    for row in race_options.itertuples()
                }
                race_id = st.selectbox(
                    "予想対象レース",
                    race_options["race_id"].tolist(),
                    format_func=lambda x: labels[x],
                )

                if st.button(
                    "予想を実行",
                    type="primary",
                    key="upcoming_prediction_button",
                ):
                    logger = new_logger()
                    try:
                        logger.info(
                            f"予想開始: race_id={race_id}, "
                            f"model={model_id}"
                        )
                        result, output_path = predict_race(
                            historical,
                            upcoming,
                            race_id,
                            model_path,
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
                        st.success(
                            f"予想結果を保存しました: {output_path}"
                        )
                        st.dataframe(
                            display,
                            use_container_width=True,
                            hide_index=True,
                        )
                        csv_bytes = result.to_csv(
                            index=False
                        ).encode("utf-8-sig")
                        st.download_button(
                            "予想結果CSVをダウンロード",
                            csv_bytes,
                            file_name=f"prediction_{race_id}.csv",
                            mime="text/csv",
                        )
                        logger.info("予想完了")
                    except Exception as exc:
                        logger.exception(
                            f"予想失敗: {exc}"
                        )
                        st.exception(exc)

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
                        display[display_columns],
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
            "スクレイピングデータ",
            "ダミーデータ",
            "学習結果",
            "取得HTML",
            "ログファイル",
        ],
        horizontal=True,
    )

    if maintenance_tab == "スクレイピングデータ":
        st.info(
            "スクレイピングデータの削除では、DB上の収集データと"
            "関連する出力データのみ削除します。取得済みHTMLは削除しません。"
        )
    elif maintenance_tab == "取得HTML":
        st.info(
            "取得HTMLの削除では、保存済みHTMLのみ削除します。"
            "DB上のスクレイピングデータは削除しません。"
        )

    delete_mode = st.radio(
        "削除方法",
        ["個別に削除", "まとめて削除"],
        horizontal=True,
        key=f"delete_mode_{maintenance_tab}",
    )

    if maintenance_tab in {"スクレイピングデータ", "ダミーデータ"}:
        runs = collection_runs()

        if runs.empty:
            st.info("削除できるデータ収集履歴がありません。")
        else:
            source_name = (
                "netkeiba"
                if maintenance_tab == "スクレイピングデータ"
                else "generator"
            )
            target_runs = runs[
                runs["source"].astype(str) == source_name
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
