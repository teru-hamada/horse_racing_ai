# 近走成績について、元データの読み込みから生成・保存・実行履歴記録までをまとめる。
# 計算前後の元データ状態を比較し、計算中に更新された場合は保存前に失敗させる。
# 取消要求は生成中と保存前に確認し、正常・失敗・取消を生成履歴へ区別して残す。
#
from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd

from .generators import RecentFormGenerator, prepare_historical_performances
from .freshness import source_data_state, source_state_token
from .pipeline import FeaturePipeline
from .registry import FeatureRegistry, FeatureSetDefinition
from .storage import replace_features, save_feature_run


_source_storage = import_module("src.00_common.storage")


# 近走成績の出力期間、セットの版、時間減衰の計算条件をまとめる設定。
@dataclass(frozen=True)
class RecentFormRunConfig:
    # 予想入力用の特徴量セット名。版と組み合わせて保存データを識別する。
    feature_set_name: str = "baseline"
    # 特徴量セットの版。計算定義が異なるデータを区別して保存する。
    feature_set_version: str = "1.1.0"
    # 時間減衰の半減期（日）。この日数だけ古い観測の重みは半分になる。
    half_life_days: float = 180.0
    # 時間減衰集計に含める過去データの最大経過日数。
    max_lookback_days: int = 1095
    # 処理対象期間の開始日。Noneを許容する設定では開始側を制限しない。
    start_date: date | None = None
    # 処理対象期間の終了日。Noneを許容する設定では終了側を制限しない。
    end_date: date | None = None


# 近走成績生成の協調的な取消を通知する例外。保存前の確認箇所から処理を抜ける。
class FeatureGenerationCancelled(Exception):
    """Raised when a user requests cancellation before feature persistence."""


# 使用する履歴列をキー順へそろえてハッシュ化し、生成元データの指紋を作る。
def _source_fingerprint(history: pd.DataFrame) -> str:
    # history: 履歴の入力表。集計対象の日付制限はこの処理の境界条件に従う。
    columns = [
        "race_id", "horse_id", "race_date", "finish_position", "last_3f_time"
    ]
    ordered = history.loc[:, columns].sort_values(
        ["race_date", "race_id", "horse_id"], kind="stable"
    )
    hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    # ファイルや入力データのハッシュを順次更新するオブジェクト。全体を一度に保持しない。
    digest = hashlib.sha256()
    digest.update(hashes.tobytes())
    digest.update(str(len(ordered)).encode("ascii"))
    return digest.hexdigest()


# running履歴を作り、元データ読込・計算・状態再検査・版単位の保存を行う。例外時もfailedまたはcancelledを記録する。
def generate_recent_form_features(
    config: RecentFormRunConfig = RecentFormRunConfig(),
    *,
    feature_database: Path | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Generate and atomically persist recent-form features from source history."""

    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    # progress: 任意の進捗コールバック。値は通常0～1の完了率。
    # is_cancelled: 呼出時に取消要求の有無を返す関数。計算や保存前に確認する。
    # 今回の生成・登録を識別する実行ID。保存行と履歴の対応付けに用いる。
    run_id = f"features_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    # この生成runで共通に付与する生成日時。行ごとにばらつかせない。
    generated_at = datetime.now()
    # 取消フラグを例外へ変換し、処理中や保存直前の確認箇所から安全に抜ける。
    def check_cancelled() -> None:
        # 通知関数があり取消要求もある場合だけ中止する。確認時点より後の処理は実行しない。
        if is_cancelled and is_cancelled():
            raise FeatureGenerationCancelled("Feature generation was cancelled")

    generator = RecentFormGenerator(
        progress=(
            (lambda value: progress(0.15 + 0.75 * value))
            if progress else None
        ),
        check_cancelled=check_cancelled,
    )
    run_config = {
        **asdict(config),
        "generators": {generator.name: generator.version},
    }
    running_metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "row_count": 0,
        "config": run_config,
        "status": "running",
    }
    save_feature_run(running_metadata, feature_database)

    try:
        # 進捗の通知先が渡された場合だけ通知する。CLIなど通知不要の利用でも同じ処理を使える。
        if progress:
            progress(0.02)
        check_cancelled()
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Loading and normalizing historical race results")
        # 近走特徴量の計算を始める時点の元データ状態。
        initial_source_state = source_data_state()
        raw_history = _source_storage.load_records("historical")
        history = prepare_historical_performances(raw_history)
        # 進捗の通知先が渡された場合だけ通知する。CLIなど通知不要の利用でも同じ処理を使える。
        if progress:
            progress(0.15)
        check_cancelled()
        targets = history[["race_id", "horse_id", "race_date"]].copy()
        if config.start_date is not None:
            targets = targets[targets["race_date"].dt.date >= config.start_date]
        if config.end_date is not None:
            targets = targets[targets["race_date"].dt.date <= config.end_date]
        targets = targets.reset_index(drop=True)
        # 対象期間に行がなければ、既存の版を空の生成結果で置換しない。
        if targets.empty:
            raise ValueError("No historical races exist in the requested target period")

        definition = FeatureSetDefinition(
            name=config.feature_set_name,
            version=config.feature_set_version,
            generators=(generator.name,),
            parameters={
                "recent_form": {
                    "half_life_days": config.half_life_days,
                    "max_lookback_days": config.max_lookback_days,
                }
            },
        )
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log(f"Generating recent-form features for {len(targets):,} runners")
        pipeline = FeaturePipeline(FeatureRegistry([generator]))
        features = pipeline.transform(targets, history, definition)
        features["race_date"] = targets["race_date"].dt.date
        features["feature_set_name"] = config.feature_set_name
        features["feature_set_version"] = config.feature_set_version
        features["history_cutoff"] = targets["race_date"]
        features["feature_run_id"] = run_id
        features["generated_at"] = generated_at

        # 特徴量を保存する直前の元データ状態。開始時と一致する必要がある。
        final_source_state = source_data_state()
        # 生成開始時と保存直前の元DB状態が違えば、混在した結果を確定せず再実行を要求する。
        if final_source_state != initial_source_state:
            raise RuntimeError(
                "Historical source data changed during feature generation; run it again"
            )

        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Replacing the stored feature-set version atomically")
        # 進捗の通知先が渡された場合だけ通知する。CLIなど通知不要の利用でも同じ処理を使える。
        if progress:
            progress(0.92)
        check_cancelled()
        replace_features(features, feature_database)
        # 利用した元データ内容を識別する指紋。軽量な状態トークンとは用途を分ける。
        fingerprint = _source_fingerprint(history)
        completed = {
            **running_metadata,
            "source_data_fingerprint": fingerprint,
            "source_state_token": source_state_token(final_source_state),
            "source_state": final_source_state,
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "row_count": len(features),
            "status": "completed",
            "message": "Recent-form feature generation completed",
        }
        save_feature_run(completed, feature_database)
        # 進捗の通知先が渡された場合だけ通知する。CLIなど通知不要の利用でも同じ処理を使える。
        if progress:
            progress(1.0)
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log(f"Saved {len(features):,} feature rows")
        return completed
    # 生成runの失敗・取消を履歴へ記録してから再送出し、呼出元にも失敗を伝える。
    except Exception as exc:
        status = "cancelled" if isinstance(exc, FeatureGenerationCancelled) else "failed"
        save_feature_run(
            {
                **running_metadata,
                "status": status,
                "message": str(exc),
            },
            feature_database,
        )
        raise
