# 出馬表条件について、元データの読み込みから生成・保存・実行履歴記録までをまとめる。
# 計算前後の元データ状態を比較し、計算中に更新された場合は保存前に失敗させる。
# 取消要求は生成中と保存前に確認し、正常・失敗・取消を生成履歴へ区別して残す。
#
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

from .context import FeatureContext
from .freshness import source_data_state, source_state_token
from .generators import RaceEntryGenerator, prepare_race_entries
from .storage import replace_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


# 出馬表特徴量を保存するときのセット名と版を指定する設定。
@dataclass(frozen=True)
class RaceEntryRunConfig:
    # 予想入力用の特徴量セット名。版と組み合わせて保存データを識別する。
    feature_set_name: str = "race_entry"
    # 特徴量セットの版。計算定義が異なるデータを区別して保存する。
    feature_set_version: str = "1.1.0"


# 出馬表特徴量の生成中止を、一般の計算エラーと区別して通知する。
class RaceEntryGenerationCancelled(Exception):
    """Raised when generation is cancelled before persistence."""


# running履歴を作り、元データ読込・計算・状態再検査・版単位の保存を行う。例外時もfailedまたはcancelledを記録する。
def generate_race_entry_features(
    config: RaceEntryRunConfig = RaceEntryRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    # progress: 任意の進捗コールバック。値は通常0～1の完了率。
    # is_cancelled: 呼出時に取消要求の有無を返す関数。計算や保存前に確認する。
    # 今回の生成・登録を識別する実行ID。保存行と履歴の対応付けに用いる。
    run_id = f"race_entry_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    # この生成runで共通に付与する生成日時。行ごとにばらつかせない。
    generated_at = datetime.now()

    # 取消フラグを例外へ変換し、処理中や保存直前の確認箇所から安全に抜ける。
    def check_cancelled() -> None:
        # 通知関数があり取消要求もある場合だけ中止する。確認時点より後の処理は実行しない。
        if is_cancelled and is_cancelled():
            raise RaceEntryGenerationCancelled("Race-entry generation was cancelled")

    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": asdict(config),
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        check_cancelled()
        # 計算開始前の元DB状態。保存前の状態と比較して途中更新を検出する。
        initial_state = source_data_state()
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Loading historical race-entry data")
        raw = _source_storage.load_records("historical")
        entries = prepare_race_entries(raw)
        targets = entries[["race_id", "horse_id", "race_date"]].copy()
        generated = RaceEntryGenerator(
            progress=progress, check_cancelled=check_cancelled
        ).transform(targets, entries, FeatureContext(
            config.feature_set_name, config.feature_set_version, {}
        ))
        generated["race_date"] = targets["race_date"].dt.date
        generated["feature_set_name"] = config.feature_set_name
        generated["feature_set_version"] = config.feature_set_version
        generated["history_cutoff"] = targets["race_date"]
        generated["feature_run_id"] = run_id
        generated["generated_at"] = generated_at
        # 計算後の元DB状態。開始時と異なれば計算中に元データが変わったことを示す。
        final_state = source_data_state()
        check_cancelled()
        # 計算中に元データが更新された場合、整合した一回の計算結果と認めず保存を中止する。
        if final_state != initial_state:
            raise RuntimeError("Historical data changed during race-entry generation")
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Saving race-entry features")
        replace_features(generated, feature_database)
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": source_state_token(final_state),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Race-entry feature generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    # 生成runの失敗・取消を履歴へ記録してから再送出し、呼出元にも失敗を伝える。
    except Exception as exc:
        status = "cancelled" if isinstance(exc, RaceEntryGenerationCancelled) else "failed"
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
