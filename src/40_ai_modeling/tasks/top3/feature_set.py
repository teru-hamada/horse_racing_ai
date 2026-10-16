# Top3学習と同じ特徴量群を、予想時点の出馬表と過去履歴から再構築する。
# 保存済みの列一覧に合わせて欠損フラグと列順をそろえ、学習と推論の入力定義を一致させる。
#
from __future__ import annotations

import pandas as pd
from pathlib import Path

from ...feature_engineering.context import FeatureContext
from ...feature_engineering.generators import (
    RaceEntryGenerator,
    RecentFormGenerator,
    RecentSpeedGenerator,
)
from ...feature_engineering.storage import load_performance_features


# 学習と同じ近走成績・近走速度・出馬表条件を生成し、対象馬のキーで結合する。
def build_top3_prediction_features(
    historical_records: pd.DataFrame,
    target_records: pd.DataFrame,
    *, feature_database: Path | None = None,
) -> pd.DataFrame:
    """Generate the same point-in-time feature families used by Top3 training."""

    # historical_records: 過去の出走結果の表。予想時点の履歴範囲は生成器側でも制限する。
    # feature_database: 特徴量DBの指定先。省略可能な場合は既定のfeatures.duckdbを使う。
    required = {"race_id", "horse_id", "race_date"}
    missing = required.difference(target_records.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"Prediction targets are missing columns: {sorted(missing)}")
    targets = target_records.copy()
    targets["race_date"] = pd.to_datetime(targets["race_date"], errors="coerce")
    if targets["race_date"].isna().any():
        raise ValueError("Prediction targets contain invalid race_date values")
    keys = targets[["race_id", "horse_id", "race_date"]]
    # 今回の予想対象に現れる馬IDの集合。履歴の索引作成を必要な馬に限定する。
    target_horses = set(targets["horse_id"].dropna().tolist())

    # 学習で使う3つの特徴量群を同じ名前・版・パラメーターで生成する。
    # 近走成績と出馬表条件は履歴から計算し、近走速度は保存済みの1走指数を集約する。
    baseline = RecentFormGenerator().transform(
        keys,
        historical_records,
        FeatureContext("baseline", "1.1.0", {
            "recent_form": {"half_life_days": 180.0, "max_lookback_days": 1095}
        }),
    )
    speed_history = (load_performance_features("speed_index", "1.0.0", feature_database)
                     if feature_database is not None else load_performance_features("speed_index", "1.0.0"))
    speed_history = speed_history[speed_history["horse_id"].isin(target_horses)]
    recent_speed = RecentSpeedGenerator().transform(
        keys,
        speed_history,
        FeatureContext("recent_speed", "1.0.0", {
            "recent_speed": {"half_life_days": 180.0, "max_lookback_days": 1095}
        }),
    )
    relevant_entries = historical_records[
        historical_records["horse_id"].isin(target_horses)
    ]
    entry_history = pd.concat(
        [relevant_entries, target_records], ignore_index=True, sort=False
    )
    race_entry = RaceEntryGenerator().transform(
        keys,
        entry_history,
        FeatureContext("race_entry", "1.1.0", {}),
    )

    # 当日の出馬表の列を保ったままキーで一対一結合する。生成結果の行順だけには依存しない。
    result = targets.copy()
    for generated in (baseline, recent_speed, race_entry):
        result = result.merge(
            generated, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
    return result


# 保存列の順序を正とし、欠損フラグを再構築する。不足する実特徴量は黙って補わない。
def align_saved_top3_features(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Recreate training-time missing indicators and enforce the saved schema."""

    # 保存済みスキーマに合わせて欠損フラグと列を整えた予想入力。
    aligned = frame.copy()
    for column in features:
        if column.endswith("__missing"):
            source = column.removesuffix("__missing")
            if source not in aligned:
                raise ValueError(f"Missing-indicator source is unavailable: {source}")
            aligned[column] = aligned[source].isna().astype("int8")
    missing = set(features).difference(aligned.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"Prediction feature generation is missing: {sorted(missing)}")
    return aligned
