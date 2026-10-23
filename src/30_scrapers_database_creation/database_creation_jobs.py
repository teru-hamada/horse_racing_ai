# 保存済みHTMLを解析してレース・オッズを登録するバックグラウンドジョブ。
# ジョブの状態、対象期間、進捗、ログを保持し、処理結果を収集履歴にも記録する。
# HTMLの取得とDB作成を分けることで、保存済みデータからの再構築を可能にする。
#
from __future__ import annotations

import threading
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib import import_module
from typing import Any

import pandas as pd

from .scrapers_database_creation_netkeiba import (
    NetkeibaDatabaseCreator,
)


AppLogger = import_module(
    "src.00_common.logging_utils"
).AppLogger
# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS
_storage = import_module("src.00_common.storage")
save_collection_run = _storage.save_collection_run
save_race_records = _storage.save_race_records
save_race_odds = _storage.save_race_odds


# 保存HTMLの解析・DB作成一回分を表す。実行結果とエラーは別のフィールドに記録する。
@dataclass
class DatabaseCreationJob:
    # 開始時に発行する一意なジョブID。状態照会と取消要求のキーとして使う。
    job_id: str
    # 過去結果historicalと未開催出馬表upcomingを区別する保存・取得の種別。
    dataset_type: str
    # 処理対象期間の開始日。Noneを許容する設定では開始側を制限しない。
    start_date: date
    # 処理対象期間の終了日。Noneを許容する設定では終了側を制限しない。
    end_date: date
    # 処理状態。running／cancellingから、completed・cancelled・failedなどの終了状態へ移る。
    status: str = "running"
    # 画面に通知する完了率。0.0が開始、1.0が完了を表す。
    progress: float = 0.0
    # 実行開始日時。default_factoryでインスタンスごとに現在時刻を取得する。
    started_at: datetime = field(default_factory=datetime.now)
    # 終了日時。未完了はNoneで、成功・失敗・取消の終了時に設定する。
    completed_at: datetime | None = None
    # サービスが返す件数などの結果。default_factoryによりジョブ間で辞書を共有しない。
    result: dict[str, Any] = field(default_factory=dict)
    sync_repository: str | None = None
    sync_status: str = "disabled"
    sync_error: str = ""
    sync_commit: str = ""
    sync_batch: str = ""
    local_saved: bool = False
    # 失敗時に画面へ渡す例外メッセージ。正常時は空文字列。
    error: str = ""
    # 画面から参照するログ行の一覧。インスタンスごとに独立したリストを持つ。
    logs: list[str] = field(default_factory=list)


# ジョブ共有状態の読み書きを保護する再入可能ロック。スナップショット取得時も使用する。
_LOCK = threading.RLock()
# ジョブIDから状態オブジェクトを引くプロセス内の登録簿。永続化された履歴とは別物。
_JOBS: dict[str, DatabaseCreationJob] = {}
# ジョブIDごとの取消通知Event。要求側がsetし、処理側がis_setで確認する。
_CANCEL_EVENTS: dict[str, threading.Event] = {}


# DB作成の取消を通知する専用例外。解析やDB保存の一般エラーとは区別する。
class DatabaseCreationCancelled(Exception):
    """利用者からデータベース作成の中止が要求された。"""


# 可変のリスト・辞書をコピーして公開用のジョブ状態を返す。内部状態への直接変更を避ける。
def _snapshot(job: DatabaseCreationJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_type": job.dataset_type,
        "start_date": job.start_date,
        "end_date": job.end_date,
        "status": job.status,
        "progress": job.progress,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": dict(job.result),
        "error": job.error,
        "logs": list(job.logs),
        "sync_status": job.sync_status,
        "sync_error": job.sync_error,
        "sync_commit": job.sync_commit,
        "local_saved": job.local_saved,
    }


# DB作成ジョブをコピーして開始日時の降順で返す。種別が指定されれば絞り込む。
def list_database_jobs(
    dataset_type: str | None = None,
) -> list[dict[str, Any]]:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        jobs = [
            _snapshot(job)
            for job in _JOBS.values()
            if (
                dataset_type is None
                or job.dataset_type == dataset_type
            )
        ]
    return sorted(
        jobs,
        key=lambda item: item["started_at"],
        reverse=True,
    )


# DB作成の同時実行を防いでジョブを登録し、HTML解析・保存をワーカースレッドへ渡す。
def start_database_job(
    dataset_type: str,
    start_date: date,
    end_date: date,
    *, sync_repository: str | None = None,
) -> str:
    # 保存先と解析形式が決まらない種別は受け入れず、既知の二種類へ限定する。
    if dataset_type not in {"historical", "upcoming"}:
        raise ValueError(
            f"未対応のdataset_typeです: {dataset_type}"
        )
    if sync_repository and not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', sync_repository):
        raise ValueError('統合先は owner/name 形式で指定してください。')
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        # 実行中だけでなく取消処理中も排他対象にし、同じDBへの生成・更新の競合を避ける。
        if any(
            job.status in {"running", "cancelling"}
            for job in _JOBS.values()
        ):
            raise RuntimeError(
                "データベース作成がすでに実行中です。"
            )
        # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
        job_id = (
            f"database_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        _JOBS[job_id] = DatabaseCreationJob(
            job_id=job_id,
            dataset_type=dataset_type,
            start_date=start_date,
            end_date=end_date,
            sync_repository=sync_repository if dataset_type == 'historical' else None,
            sync_status='pending' if sync_repository and dataset_type == 'historical' else 'disabled',
        )
        _CANCEL_EVENTS[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_job,
        args=(job_id,),
        name=job_id,
        daemon=True,
    )
    thread.start()
    return job_id


# 実行中のDB作成へ取消要求を通知する。終了済みや未知のIDならFalseを返す。
def cancel_database_job(job_id: str) -> bool:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        job = _JOBS.get(job_id)
        cancel_event = _CANCEL_EVENTS.get(job_id)
        if (
            job is None
            or cancel_event is None
            or job.status not in {"running", "cancelling"}
            or job.local_saved
        ):
            return False
        cancel_event.set()
        # 要求は受け付けたが処理側の停止確認前なので、cancelledではなく取消処理中とする。
        job.status = "cancelling"
        job.logs.append(
            f"[{datetime.now():%H:%M:%S}] INFO    "
            "中止要求を受け付けました。"
        )
        return True


# ジョブの状態と結果を収集履歴の形式へ整え、実行記録として保存する。
def _save_run(
    job: DatabaseCreationJob,
    *,
    status: str,
    race_count: int,
    row_count: int,
    message: str,
) -> None:
    save_collection_run(
        {
            "run_id": job.job_id,
            "started_at": job.started_at,
            "completed_at": datetime.now(),
            "status": status,
            "dataset_type": job.dataset_type,
            "source": "cached_html",
            "start_date": job.start_date,
            "end_date": job.end_date,
            "race_count": race_count,
            "row_count": row_count,
            "output_path": "",
            "message": message,
        }
    )


# ジョブ条件を処理実装へ渡し、ログ・進捗を更新しながら成功・取消・失敗を記録する。
def _run_job(job_id: str) -> None:
    job = _JOBS[job_id]
    cancel_event = _CANCEL_EVENTS[job_id]

    # 処理側からのログ行をジョブへ転送し、画面で参照できるようにする。
    def log_callback(line: str) -> None:
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.logs.append(line)
            job.logs[:] = job.logs[-300:]

    # 収集・解析の進捗をジョブへ反映する。取消要求の確認にも使われる。
    def progress_callback(value: float) -> None:
        # 取消要求はコールバック到達時に検知し、例外で安全に収集・解析ループを抜ける。
        if cancel_event.is_set():
            raise DatabaseCreationCancelled()
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.progress = min(
                max(float(value), 0.0),
                0.95,
            )

    logger = AppLogger(PATHS.logs, callback=log_callback)
    scraper = NetkeibaDatabaseCreator(logger=logger)
    try:
        logger.info(
            f"データベース作成開始: "
            f"{job.start_date}～{job.end_date}, "
            f"dataset_type={job.dataset_type}"
        )
        frame = scraper.parse_cached_date_range(
            start_date=job.start_date,
            end_date=job.end_date,
            dataset_type=job.dataset_type,
            progress_callback=progress_callback,
        )
        # 取消要求はコールバック到達時に検知し、例外で安全に収集・解析ループを抜ける。
        if cancel_event.is_set():
            raise DatabaseCreationCancelled()
        if frame.empty:
            raise ValueError(
                "対象期間の解析可能なHTMLがありません。"
                "先にHTML収集を実行してください。"
            )

        logger.info(
            f"解析完了: {frame['race_id'].nunique():,}レース / "
            f"{len(frame):,}頭。データベースへ保存します。"
        )
        if job.sync_repository:
            try:
                sync = import_module('src.10_workflows.database_sync')
                job.sync_batch = sync.enqueue(frame, job.job_id, job.sync_repository, paths=PATHS, prepared=True)
            except Exception as exc:
                job.sync_error = str(exc)
                logger.warning(f'同期用の控えを保存できません。ローカル登録を続けます: {exc}')
        save_race_records(
            frame,
            job.job_id,
            job.dataset_type,
        )
        with _LOCK:
            job.local_saved = True
        odds_count = 0
        if job.dataset_type == "upcoming":
            try:
                logger.info(
                    "JRAオッズDB登録開始: "
                    f"対象期間={job.start_date}～{job.end_date}"
                )
                odds_parser = import_module(
                    "src.30_scrapers_database_creation.jra_odds_parser"
                )
                odds_frames = []
                dated_frame = frame.assign(
                    race_date=pd.to_datetime(frame["race_date"], errors="coerce")
                )
                for race_date_value, date_rows in dated_frame.groupby(
                    dated_frame["race_date"].dt.date
                ):
                    if pd.isna(race_date_value):
                        continue
                    parsed_odds = odds_parser.parse_odds_directory(
                        PATHS.upcoming_html
                        / str(race_date_value.year)
                        / "odds",
                        race_date_value,
                        date_rows["race_id"].astype(str).unique().tolist(),
                    )
                    if not parsed_odds.empty:
                        odds_frames.append(parsed_odds)
                if odds_frames:
                    odds = pd.concat(odds_frames, ignore_index=True)
                    save_race_odds(odds, job.job_id)
                    odds_count = len(odds)
                    by_type = ", ".join(
                        f"{bet_type}={count:,}"
                        for bet_type, count in odds["bet_type"]
                        .value_counts()
                        .sort_index()
                        .items()
                    )
                    logger.info(
                        "JRAオッズDB登録完了: "
                        f"レース数={odds['race_id'].nunique():,}, "
                        f"オッズ行数={odds_count:,}, 券種別=[{by_type}]"
                    )
                else:
                    logger.warning(
                        "JRAオッズDB登録なし: 対象レースに対応する"
                        "保存済みオッズHTMLがありません。"
                    )
            # ワーカースレッドの例外を状態とログへ残し、画面側が失敗理由を確認できるようにする。
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JRAオッズをDB登録できませんでした。"
                    f"レース情報の登録は完了しています: {exc}"
                )
        race_count = int(frame["race_id"].nunique())
        row_count = len(frame)
        result = {
            "race_count": race_count,
            "row_count": row_count,
            "odds_count": odds_count,
        }
        _save_run(
            job,
            status="completed",
            race_count=race_count,
            row_count=row_count,
            message="取得済みHTMLからデータベース作成完了",
        )
        if job.sync_repository:
            with _LOCK:
                job.result = result
                job.progress = 0.96
                job.sync_status = 'running'
            try:
                sync = import_module('src.10_workflows.database_sync')
                if not job.sync_batch:
                    job.sync_batch = sync.enqueue(frame, job.job_id, job.sync_repository, paths=PATHS)
                else:
                    sync.mark_local_saved(job.sync_batch, paths=PATHS)
                synced = sync.synchronize(job.sync_batch, paths=PATHS, log=logger.info)
                with _LOCK:
                    job.sync_status = 'completed'
                    job.sync_commit = synced['commit']
                    job.sync_error = ''
            except Exception as exc:
                logger.warning(f'ローカルDB登録は完了しました。非公開DBとの統合は未完了です: {exc}')
                with _LOCK:
                    job.sync_status = 'failed'
                    job.sync_error = str(exc)
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.result = result
            job.progress = 1.0
            # 成果物が完成したので成功を確定する。進捗と完了日時も同じ終了処理で更新する。
            job.status = "completed"
            job.completed_at = datetime.now()
        logger.info(
            f"データベース作成完了: {race_count:,}レース / "
            f"{row_count:,}頭"
        )
    # 利用者の取消は一般エラーと分けて扱い、取消状態と終了日時を記録する。
    except DatabaseCreationCancelled:
        logger.info(
            "データベース作成を中止しました。"
            "保存前の解析結果は登録していません。"
        )
        _save_run(
            job,
            status="cancelled",
            race_count=0,
            row_count=0,
            message="データベース作成を中止",
        )
        with _LOCK:
            # 取消例外を受けた終了。完了済み成果物を失敗扱いして作り直すことはしない。
            job.status = "cancelled"
            job.completed_at = datetime.now()
    # ワーカースレッドの例外を状態とログへ残し、画面側が失敗理由を確認できるようにする。
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"データベース作成失敗: {exc}")
        _save_run(
            job,
            status="failed",
            race_count=0,
            row_count=0,
            message=str(exc),
        )
        with _LOCK:
            job.error = str(exc)
            # 一般の例外で終了したことを記録する。利用者による取消とは区別する。
            job.status = "failed"
            job.completed_at = datetime.now()


def retry_database_sync(batch_id: str) -> str:
    """Retry a durable batch without parsing or registering local HTML again."""
    sync = import_module('src.10_workflows.database_sync')
    _, metadata = sync.read_batch(batch_id, PATHS)
    with _LOCK:
        if any(job.status in {'running', 'cancelling'} for job in _JOBS.values()):
            raise RuntimeError('データベース作成・統合の処理が実行中です。')
        job_id = 'sync_' + uuid.uuid4().hex[:12]
        job = DatabaseCreationJob(job_id, 'historical', date.fromisoformat(min(metadata['dates'])),
                                  date.fromisoformat(max(metadata['dates'])),
                                  sync_repository=metadata['repository'], sync_status='running',
                                  sync_batch=batch_id, local_saved=True,
                                  result={'race_count': metadata['races'], 'row_count': metadata['rows']})
        _JOBS[job_id] = job
        _CANCEL_EVENTS[job_id] = threading.Event()

    def worker():
        logger = AppLogger(PATHS.logs)
        try:
            result = sync.synchronize(batch_id, paths=PATHS, log=logger.info)
            with _LOCK:
                job.sync_status, job.sync_commit = 'completed', result['commit']
        except Exception as exc:
            with _LOCK:
                job.sync_status, job.sync_error = 'failed', str(exc)
        finally:
            with _LOCK:
                job.status, job.progress, job.completed_at = 'completed', 1.0, datetime.now()
    threading.Thread(target=worker, name=job_id, daemon=True).start()
    return job_id
