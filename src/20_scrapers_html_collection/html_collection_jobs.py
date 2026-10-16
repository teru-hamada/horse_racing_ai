# HTML収集をバックグラウンドスレッドで実行し、進捗・ログ・取消要求を管理する。
# 画面はジョブのスナップショットを参照し、共有状態の読み書きにはロックを使う。
# 取消は要求フラグを処理側が確認する協調的な方式で、スレッドを強制終了するものではない。
#
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib import import_module
from typing import Any

from .scrapers_html_collection_netkeiba import (
    PAGE_REQUEST_INTERVAL_SECONDS,
    NetkeibaHtmlCollector,
)


AppLogger = import_module(
    "src.00_common.logging_utils"
).AppLogger
# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = import_module("src.00_common.config").PATHS


# 一回のHTML収集について、要求条件と実行中に更新される状態を保持する。
@dataclass
class HtmlCollectionJob:
    # 開始時に発行する一意なジョブID。状態照会と取消要求のキーとして使う。
    job_id: str
    # 過去結果historicalと未開催出馬表upcomingを区別する保存・取得の種別。
    dataset_type: str
    # 処理対象期間の開始日。Noneを許容する設定では開始側を制限しない。
    start_date: date
    # 処理対象期間の終了日。Noneを許容する設定では終了側を制限しない。
    end_date: date
    # Trueなら取得キャッシュを再利用せず、新しいHTMLの取得を要求する。
    force: bool
    # 処理状態。running／cancellingから、completed・cancelled・failedなどの終了状態へ移る。
    status: str = "running"
    # 画面に通知する完了率。0.0が開始、1.0が完了を表す。
    progress: float = 0.0
    # 実行開始日時。default_factoryでインスタンスごとに現在時刻を取得する。
    started_at: datetime = field(default_factory=datetime.now)
    # 終了日時。未完了はNoneで、成功・失敗・取消の終了時に設定する。
    completed_at: datetime | None = None
    # サービスが返す件数などの結果。default_factoryによりジョブ間で辞書を共有しない。
    result: dict[str, int] = field(default_factory=dict)
    # 失敗時に画面へ渡す例外メッセージ。正常時は空文字列。
    error: str = ""
    # 画面から参照するログ行の一覧。インスタンスごとに独立したリストを持つ。
    logs: list[str] = field(default_factory=list)


# ジョブ共有状態の読み書きを保護する再入可能ロック。スナップショット取得時も使用する。
_LOCK = threading.RLock()
# ジョブIDから状態オブジェクトを引くプロセス内の登録簿。永続化された履歴とは別物。
_JOBS: dict[str, HtmlCollectionJob] = {}
# ジョブIDごとの取消通知Event。要求側がsetし、処理側がis_setで確認する。
_CANCEL_EVENTS: dict[str, threading.Event] = {}


# 利用者の取消要求を通常の取得失敗と区別し、ジョブをcancelledへ移すための例外。
class CollectionCancelled(Exception):
    """利用者からHTML収集の中止が要求された。"""


# 可変のリスト・辞書をコピーして公開用のジョブ状態を返す。内部状態への直接変更を避ける。
def _snapshot(job: HtmlCollectionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "dataset_type": job.dataset_type,
        "start_date": job.start_date,
        "end_date": job.end_date,
        "force": job.force,
        "status": job.status,
        "progress": job.progress,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": dict(job.result),
        "error": job.error,
        "logs": list(job.logs),
    }


# 任意のデータ種別で絞ったジョブのコピーを、開始日時が新しい順に返す。
def list_jobs(dataset_type: str | None = None) -> list[dict[str, Any]]:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        jobs = [
            _snapshot(job)
            for job in _JOBS.values()
            if dataset_type is None or job.dataset_type == dataset_type
        ]
    return sorted(jobs, key=lambda item: item["started_at"], reverse=True)


# 種別と同時実行を検査し、状態と取消Eventを登録してから収集スレッドを開始する。
def start_job(
    dataset_type: str,
    start_date: date,
    end_date: date,
    force: bool,
) -> str:
    # force: キャッシュを使わず新規取得するかを指定する。
    # 保存先と解析形式が決まらない種別は受け入れず、既知の二種類へ限定する。
    if dataset_type not in {"historical", "upcoming"}:
        raise ValueError(f"未対応のdataset_typeです: {dataset_type}")
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        for existing in _JOBS.values():
            # 同じ種別を重複収集しないよう、既に走っているジョブがあれば開始を拒否する。
            if existing.dataset_type == dataset_type and existing.status == "running":
                raise RuntimeError("同じ種類のHTML収集がすでに実行中です。")
        # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
        job_id = (
            f"html_{dataset_type}_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        job = HtmlCollectionJob(
            job_id=job_id,
            dataset_type=dataset_type,
            start_date=start_date,
            end_date=end_date,
            force=force,
        )
        _JOBS[job_id] = job
        _CANCEL_EVENTS[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_job,
        args=(job_id,),
        name=job_id,
        daemon=True,
    )
    thread.start()
    return job_id


# 有効なジョブの取消Eventを立て、停止完了まではcancellingとして応答する。
def cancel_job(job_id: str) -> bool:
    """実行中ジョブへ中止を要求する。すでに終了している場合はFalse。"""
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        job = _JOBS.get(job_id)
        cancel_event = _CANCEL_EVENTS.get(job_id)
        if (
            job is None
            or cancel_event is None
            or job.status not in {"running", "cancelling"}
        ):
            return False
        cancel_event.set()
        # 要求は受け付けたが処理側の停止確認前なので、cancelledではなく取消処理中とする。
        job.status = "cancelling"
        job.logs.append(
            f"[{datetime.now():%H:%M:%S}] INFO    中止要求を受け付けました。"
        )
        return True


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
            raise CollectionCancelled()
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.progress = min(max(float(value), 0.0), 1.0)

    logger = AppLogger(PATHS.logs, callback=log_callback)
    scraper = NetkeibaHtmlCollector(
        logger=logger,
        interval_seconds=PAGE_REQUEST_INTERVAL_SECONDS,
    )
    try:
        logger.info(
            f"HTML収集開始: {job.start_date}～{job.end_date}, "
            f"dataset_type={job.dataset_type}, force={job.force}"
        )
        result = scraper.collect_html_date_range(
            start_date=job.start_date,
            end_date=job.end_date,
            dataset_type=job.dataset_type,
            force=job.force,
            progress_callback=progress_callback,
        )
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.result = result
            job.progress = 1.0
            # 成果物が完成したので成功を確定する。進捗と完了日時も同じ終了処理で更新する。
            job.status = "completed"
            job.completed_at = datetime.now()
        logger.info(f"HTML収集完了: {result}")
    # 利用者の取消は一般エラーと分けて扱い、取消状態と終了日時を記録する。
    except CollectionCancelled:
        logger.info("HTML収集を中止しました。取得済みHTMLは保持します。")
        with _LOCK:
            # 取消例外を受けた終了。完了済み成果物を失敗扱いして作り直すことはしない。
            job.status = "cancelled"
            job.completed_at = datetime.now()
    # ワーカースレッドの例外を状態とログへ残し、画面側が失敗理由を確認できるようにする。
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"HTML収集失敗: {exc}")
        with _LOCK:
            job.error = str(exc)
            # 一般の例外で終了したことを記録する。利用者による取消とは区別する。
            job.status = "failed"
            job.completed_at = datetime.now()
