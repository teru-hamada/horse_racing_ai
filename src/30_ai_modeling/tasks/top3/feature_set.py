from __future__ import annotations

import pandas as pd

from ...feature_engineering.context import FeatureContext
from ...feature_engineering.generators import (
    RaceEntryGenerator,
    RecentFormGenerator,
    RecentSpeedGenerator,
)
from ...feature_engineering.storage import load_performance_features


def build_top3_prediction_features(
    historical_records: pd.DataFrame,
    target_records: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the same point-in-time feature families used by Top3 training."""

    required = {"race_id", "horse_id", "race_date"}
    missing = required.difference(target_records.columns)
    if missing:
        raise ValueError(f"Prediction targets are missing columns: {sorted(missing)}")
    targets = target_records.copy()
    targets["race_date"] = pd.to_datetime(targets["race_date"], errors="coerce")
    if targets["race_date"].isna().any():
        raise ValueError("Prediction targets contain invalid race_date values")
    keys = targets[["race_id", "horse_id", "race_date"]]

    baseline = RecentFormGenerator().transform(
        keys,
        historical_records,
        FeatureContext("baseline", "1.1.0", {
            "recent_form": {"half_life_days": 180.0, "max_lookback_days": 1095}
        }),
    )
    speed_history = load_performance_features("speed_index", "1.0.0")
    recent_speed = RecentSpeedGenerator().transform(
        keys,
        speed_history,
        FeatureContext("recent_speed", "1.0.0", {
            "recent_speed": {"half_life_days": 180.0, "max_lookback_days": 1095}
        }),
    )
    entry_history = pd.concat(
        [historical_records, target_records], ignore_index=True, sort=False
    )
    race_entry = RaceEntryGenerator().transform(
        keys,
        entry_history,
        FeatureContext("race_entry", "1.0.0", {}),
    )

    result = targets.copy()
    for generated in (baseline, recent_speed, race_entry):
        result = result.merge(
            generated, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
    return result


def align_saved_top3_features(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Recreate training-time missing indicators and enforce the saved schema."""

    aligned = frame.copy()
    for column in features:
        if column.endswith("__missing"):
            source = column.removesuffix("__missing")
            if source not in aligned:
                raise ValueError(f"Missing-indicator source is unavailable: {source}")
            aligned[column] = aligned[source].isna().astype("int8")
    missing = set(features).difference(aligned.columns)
    if missing:
        raise ValueError(f"Prediction feature generation is missing: {sorted(missing)}")
    return aligned
