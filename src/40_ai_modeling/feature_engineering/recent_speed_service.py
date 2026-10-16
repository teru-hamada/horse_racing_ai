# 近走速度指数について、元データの読み込みから生成・保存・実行履歴記録までをまとめる。
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

from .context import FeatureContext
from .freshness import performance_feature_freshness, source_data_state, source_state_token
from .generators import RecentSpeedGenerator, prepare_historical_performances
from .storage import (
    load_performance_features,
    performance_feature_summary,
    replace_features,
    save_feature_run,
)


_source_storage = import_module("src.00_common.storage")


# 近走速度の集約条件と、入力となる速度指数の名前・版を指定する設定。
@dataclass(frozen=True)
class RecentSpeedRunConfig:
    # 予想入力用の特徴量セット名。版と組み合わせて保存データを識別する。
    feature_set_name: str = "recent_speed"
    # 特徴量セットの版。計算定義が異なるデータを区別して保存する。
    feature_set_version: str = "1.0.0"
    # 集約入力として参照する1走単位の特徴量名。
    dependency_name: str = "speed_index"
    # 依存する1走単位の特徴量の版。別の計算定義の指数を混在させない。
    dependency_version: str = "1.0.0"
    # 時間減衰の半減期（日）。この日数だけ古い観測の重みは半分になる。
    half_life_days: float = 180.0
    # 時間減衰集計に含める過去データの最大経過日数。
    max_lookback_days: int = 1095


# 近走速度の取消要求を上位へ伝える専用例外。
class RecentSpeedGenerationCancelled(Exception):
    """Raised when generation is cancelled before persistence."""


# running履歴を作り、元データ読込・計算・状態再検査・版単位の保存を行う。例外時もfailedまたはcancelledを記録する。
def generate_recent_speed_features(
    config: RecentSpeedRunConfig = RecentSpeedRunConfig(),
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
    run_id = f"recent_speed_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    # この生成runで共通に付与する生成日時。行ごとにばらつかせない。
    generated_at = datetime.now()

    # 取消フラグを例外へ変換し、処理中や保存直前の確認箇所から安全に抜ける。
    def check_cancelled() -> None:
        # 通知関数があり取消要求もある場合だけ中止する。確認時点より後の処理は実行しない。
        if is_cancelled and is_cancelled():
            raise RecentSpeedGenerationCancelled("Recent-speed generation was cancelled")

    # 元になる1走ごとの指数が古い場合は、その上に新しい集計を作らず再生成を要求する。
    dependency_freshness = performance_feature_freshness(
        config.dependency_name,
        config.dependency_version,
        feature_database=feature_database,
    )
    # 依存指数の件数と生成run・指紋などをまとめた情報。
    dependency_summary = performance_feature_summary(
        config.dependency_name, config.dependency_version, feature_database
    )
    # 近走速度は保存済み指数に依存するため、欠落・不明・古い状態の指数では集約を始めない。
    if dependency_freshness["status"] != "fresh":
        raise RuntimeError("The dependent speed index is missing or stale; generate it first")
    # 集約に使った速度指数の生成run。終了前の再確認と次回の鮮度判定に使う。
    dependency_run_id = dependency_summary["latest_run_id"]
    run_config = {
        **asdict(config),
        "dependency_run_id": dependency_run_id,
        "dependency_fingerprint": dependency_summary["source_data_fingerprint"],
    }
    metadata = {
        "feature_run_id": run_id,
        "feature_set_name": config.feature_set_name,
        "feature_set_version": config.feature_set_version,
        "generated_at": generated_at,
        "row_count": 0,
        "config": run_config,
        "status": "running",
    }
    save_feature_run(metadata, feature_database)
    try:
        check_cancelled()
        # 計算開始前の元DB状態。保存前の状態と比較して途中更新を検出する。
        initial_state = source_data_state()
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Loading targets and one-run speed indexes")
        raw = _source_storage.load_records("historical")
        prepared = prepare_historical_performances(raw)
        targets = prepared[["race_id", "horse_id", "race_date"]].copy()
        speed_history = load_performance_features(
            config.dependency_name, config.dependency_version, feature_database
        )
        generated = RecentSpeedGenerator(
            progress=progress,
            check_cancelled=check_cancelled,
        ).transform(
            targets,
            speed_history,
            FeatureContext(
                config.feature_set_name,
                config.feature_set_version,
                {"recent_speed": {
                    "half_life_days": config.half_life_days,
                    "max_lookback_days": config.max_lookback_days,
                }},
            ),
        )
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
            raise RuntimeError("Historical data changed during recent-speed generation")
        # 元DBが不変でも、計算中に依存指数が作り直された可能性があるためrun IDも再確認する。
        current_dependency = performance_feature_summary(
            config.dependency_name, config.dependency_version, feature_database
        )
        # 元DBが同じでも依存指数が計算中に作り直されたら、今回の集計は保存せずやり直す。
        if current_dependency["latest_run_id"] != dependency_run_id:
            raise RuntimeError("The speed-index dependency changed during generation")
        # ログの受取先がある場合だけ診断を通知し、計算本体をUI実装へ依存させない。
        if log:
            log("Saving recent-speed aggregates")
        replace_features(generated, feature_database)
        # ファイルや入力データのハッシュを順次更新するオブジェクト。全体を一度に保持しない。
        digest = hashlib.sha256()
        digest.update(str(dependency_run_id).encode("utf-8"))
        digest.update(source_state_token(final_state).encode("ascii"))
        completed = {
            **metadata,
            "status": "completed",
            "row_count": len(generated),
            "start_date": targets["race_date"].min().date(),
            "end_date": targets["race_date"].max().date(),
            "source_data_fingerprint": digest.hexdigest(),
            "source_state": final_state,
            "source_state_token": source_state_token(final_state),
            "message": "Recent-speed feature generation completed",
        }
        save_feature_run(completed, feature_database)
        return completed
    # 生成runの失敗・取消を履歴へ記録してから再送出し、呼出元にも失敗を伝える。
    except Exception as exc:
        status = "cancelled" if isinstance(exc, RecentSpeedGenerationCancelled) else "failed"
        save_feature_run({**metadata, "status": status, "message": str(exc)}, feature_database)
        raise
