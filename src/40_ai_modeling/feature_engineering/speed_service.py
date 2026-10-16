# 1走ごとの速度指数について、元データの読み込みから生成・保存・実行履歴記録までをまとめる。
# 計算前後の元データ状態を比較し、計算中に更新された場合は保存前に失敗させる。
# 取消要求は生成中と保存前に確認し、正常・失敗・取消を生成履歴へ区別して残す。
#
from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd

from .context import FeatureContext
from .freshness import source_data_state, source_state_token
from .generators import SpeedIndexGenerator, prepare_speed_performances
from .storage import replace_performance_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


# 標準時計の参照期間・半減期、時間差と斤量の換算係数をまとめる設定。
@dataclass(frozen=True)
class SpeedIndexRunConfig:
    # 1走ごとの指数を保存するときの特徴量名。
    feature_name: str = "speed_index"
    # 1走ごとの指数の計算版。名前と組で保存・参照に使用する。
    feature_version: str = "1.0.0"
    # 時間減衰の半減期（日）。この日数だけ古い観測の重みは半分になる。
    half_life_days: float = 730.0
    # 時間減衰集計に含める過去データの最大経過日数。
    max_lookback_days: int = 3650
    # 1600m換算で標準タイムとの差1秒を何ポイントへ換算するか。
    seconds_scale_at_1600m: float = 10.0
    # 基準斤量55kgとの差1kgを何ポイントへ換算するか。
    weight_points_per_kg: float = 1.5


# 速度指数の取消用例外。ジョブ管理側はfailedではなくcancelledとして記録する。
class SpeedIndexGenerationCancelled(Exception):
    """Raised when speed-index generation is cancelled before persistence."""


# 元データの対象列を安定した順でハッシュ化し、生成runへ記録する。
def _fingerprint(frame: pd.DataFrame) -> str:
    columns = list(prepare_speed_performances(frame).columns)
    ordered = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    ordered = ordered.sort_values(["race_date", "race_id", "horse_id"], kind="stable")
    # ファイルや入力データのハッシュを順次更新するオブジェクト。全体を一度に保持しない。
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    digest.update(str(len(ordered)).encode("ascii"))
    return digest.hexdigest()


# running履歴を作り、元データ読込・計算・状態再検査・版単位の保存を行う。例外時もfailedまたはcancelledを記録する。
def generate_speed_index_features(
    config: SpeedIndexRunConfig = SpeedIndexRunConfig(),
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
    run_id = f"speed_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    # この生成runで共通に付与する生成日時。行ごとにばらつかせない。
    generated_at = datetime.now()
    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_name,
        "feature_set_version": config.feature_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": asdict(config),
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        # 取消フラグを例外へ変換し、処理中や保存直前の確認箇所から安全に抜ける。
        def check_cancelled() -> None:
            # 通知関数があり取消要求もある場合だけ中止する。確認時点より後の処理は実行しない。
            if is_cancelled and is_cancelled():
                raise SpeedIndexGenerationCancelled(
                    "Speed-index generation was cancelled"
                )

        check_cancelled()
        # 計算開始前の元DB状態。保存前の状態と比較して途中更新を検出する。
        initial_state = source_data_state()
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Loading historical performances for speed-index generation")
        raw = _source_storage.load_records("historical")
        history = prepare_speed_performances(raw)
        targets = history.copy()
        generator = SpeedIndexGenerator(
            progress=progress,
            check_cancelled=check_cancelled,
        )
        context = FeatureContext(
            config.feature_name,
            config.feature_version,
            {"speed_index": {
                "half_life_days": config.half_life_days,
                "max_lookback_days": config.max_lookback_days,
                "seconds_scale_at_1600m": config.seconds_scale_at_1600m,
                "weight_points_per_kg": config.weight_points_per_kg,
            }},
        )
        generated = generator.transform(targets, history, context)
        generated["race_date"] = targets["race_date"].dt.date
        generated["performance_feature_name"] = config.feature_name
        generated["performance_feature_version"] = config.feature_version
        generated["history_cutoff"] = targets["race_date"]
        generated["feature_run_id"] = run_id
        generated["generated_at"] = generated_at
        # 計算後の元DB状態。開始時と異なれば計算中に元データが変わったことを示す。
        final_state = source_data_state()
        check_cancelled()
        # 計算中に元データが更新された場合、整合した一回の計算結果と認めず保存を中止する。
        if final_state != initial_state:
            raise RuntimeError("Historical data changed during speed-index generation")
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Saving per-performance speed indexes")
        replace_performance_features(generated, feature_database)
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": _fingerprint(raw),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Speed-index generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    # 生成runの失敗・取消を履歴へ記録してから再送出し、呼出元にも失敗を伝える。
    except Exception as exc:
        status = (
            "cancelled"
            if isinstance(exc, SpeedIndexGenerationCancelled)
            else "failed"
        )
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
