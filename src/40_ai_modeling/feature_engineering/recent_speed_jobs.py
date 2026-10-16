# 近走速度指数の生成をバックグラウンドジョブとして実行する。
# ジョブIDごとに進捗・ログ・結果・例外を管理し、画面から状態照会や取消要求を受け付ける。
# ジョブ情報はプロセス内のメモリで管理するため、アプリを再起動すると引き継がれない。
#
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .recent_speed_service import (
    RecentSpeedGenerationCancelled,
    RecentSpeedRunConfig,
    generate_recent_speed_features,
)


# 近走速度指数を集約するジョブ。依存指数の確認と計算の詳細はサービス側が担う。
@dataclass
class RecentSpeedJob:
    # 開始時に発行する一意なジョブID。状態照会と取消要求のキーとして使う。
    job_id: str
    # 今回の実行条件。ジョブ側が保持し、生成サービスへそのまま渡す。
    config: RecentSpeedRunConfig
    # 処理状態。running／cancellingから、completed・cancelled・failedなどの終了状態へ移る。
    status: str = "running"
    # 画面に通知する完了率。0.0が開始、1.0が完了を表す。
    progress: float = 0.0
    # 実行開始日時。default_factoryでインスタンスごとに現在時刻を取得する。
    started_at: datetime = field(default_factory=datetime.now)
    # 終了日時。未完了はNoneで、成功・失敗・取消の終了時に設定する。
    completed_at: datetime | None = None
    # サービスが返す件数などの結果。default_factoryによりジョブ間で辞書を共有しない。
    result: dict[str, object] = field(default_factory=dict)
    # 失敗時に画面へ渡す例外メッセージ。正常時は空文字列。
    error: str = ""
    # 画面から参照するログ行の一覧。インスタンスごとに独立したリストを持つ。
    logs: list[str] = field(default_factory=list)


# ジョブ共有状態の読み書きを保護する再入可能ロック。スナップショット取得時も使用する。
_LOCK = threading.RLock()
# ジョブIDから状態オブジェクトを引くプロセス内の登録簿。永続化された履歴とは別物。
_JOBS: dict[str, RecentSpeedJob] = {}
# ジョブIDごとの取消通知Event。要求側がsetし、処理側がis_setで確認する。
_CANCEL_EVENTS: dict[str, threading.Event] = {}


# 近走速度ジョブの一覧を開始日時の降順で返す。ログ・結果もコピーする。
def list_recent_speed_jobs() -> list[dict[str, Any]]:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        return sorted([
            {
                "job_id": job.job_id, "config": job.config, "status": job.status,
                "progress": job.progress, "started_at": job.started_at,
                "completed_at": job.completed_at, "result": dict(job.result),
                "error": job.error, "logs": list(job.logs),
            }
            for job in _JOBS.values()
        ], key=lambda item: item["started_at"], reverse=True)


# 近走速度の集約処理をバックグラウンドで開始し、ジョブIDを返す。
def start_recent_speed_job(config: RecentSpeedRunConfig) -> str:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        # 実行中だけでなく取消処理中も排他対象にし、同じDBへの生成・更新の競合を避ける。
        if any(job.status in {"running", "cancelling"} for job in _JOBS.values()):
            raise RuntimeError("Recent-speed generation is already running")
        # 非同期処理を特定するキー。状態照会と取消に同じ値を使用する。
        job_id = f"recent_speed_job_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        _JOBS[job_id] = RecentSpeedJob(job_id, config)
        _CANCEL_EVENTS[job_id] = threading.Event()
    threading.Thread(target=_run, args=(job_id,), name=job_id, daemon=True).start()
    return job_id


# 近走速度のワーカーへ取消Eventで要求を伝える。
def cancel_recent_speed_job(job_id: str) -> bool:
    # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
    with _LOCK:
        job = _JOBS.get(job_id)
        event = _CANCEL_EVENTS.get(job_id)
        # 未知のジョブや終了済みジョブは取消できない。状態を変更せずFalseで呼出元へ知らせる。
        if job is None or event is None or job.status not in {"running", "cancelling"}:
            return False
        event.set()
        # 要求は受け付けたが処理側の停止確認前なので、cancelledではなく取消処理中とする。
        job.status = "cancelling"
        job.logs.append("中止を要求しました。完成済みデータは維持されます。")
        return True


# 近走速度生成サービスを呼び出し、戻り値と例外をジョブの完了・失敗・取消状態へ反映する。
def _run(job_id: str) -> None:
    job = _JOBS[job_id]
    event = _CANCEL_EVENTS[job_id]

    # バックグラウンド処理からのログ行を受け取り、共有ジョブのログ一覧へ追加する。
    def log(message: str) -> None:
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.logs.append(f"[{datetime.now():%H:%M:%S}] {message}")

    # サービスから通知された進捗値をジョブへ反映し、表示範囲を0～1へ収める。
    def progress(value: float) -> None:
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.progress = min(max(float(value), 0.0), 1.0)

    try:
        result = generate_recent_speed_features(
            job.config, log=log, progress=progress, is_cancelled=event.is_set
        )
        # 共有ジョブをロックして参照・更新し、UI側の照会や取消要求との同時アクセスから保護する。
        with _LOCK:
            job.result = result
            job.progress = 1.0
            # 成果物が完成したので成功を確定する。進捗と完了日時も同じ終了処理で更新する。
            job.status = "completed"
            job.completed_at = datetime.now()
    # 利用者の取消は一般エラーと分けて扱い、取消状態と終了日時を記録する。
    except RecentSpeedGenerationCancelled:
        with _LOCK:
            # 取消例外を受けた終了。完了済み成果物を失敗扱いして作り直すことはしない。
            job.status = "cancelled"
            job.completed_at = datetime.now()
    # ワーカースレッドの例外を状態とログへ残し、画面側が失敗理由を確認できるようにする。
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            job.error = str(exc)
            # 一般の例外で終了したことを記録する。利用者による取消とは区別する。
            job.status = "failed"
            job.completed_at = datetime.now()
