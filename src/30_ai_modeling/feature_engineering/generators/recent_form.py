from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..common import decay_weights, effective_count, weighted_mean
from ..context import FeatureContext


REQUIRED_TARGET_COLUMNS = ("race_id", "horse_id", "race_date")
REQUIRED_HISTORY_COLUMNS = (
    "race_id",
    "horse_id",
    "race_date",
    "finish_position",
    "last_3f_time",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def prepare_historical_performances(history: pd.DataFrame) -> pd.DataFrame:
    """Create result-only, race-relative values used by recent-form aggregation."""

    _require_columns(history, REQUIRED_HISTORY_COLUMNS, "History")
    selected_columns = list(REQUIRED_HISTORY_COLUMNS)
    if "collected_at" in history.columns:
        selected_columns.append("collected_at")
    prepared = history.loc[:, selected_columns].copy()
    prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
    prepared["finish_position"] = pd.to_numeric(
        prepared["finish_position"], errors="coerce"
    )
    prepared["last_3f_time"] = pd.to_numeric(prepared["last_3f_time"], errors="coerce")
    if prepared[["race_id", "horse_id"]].isna().any().any():
        raise ValueError("History race_id and horse_id cannot be missing")
    duplicate_keys = prepared.duplicated(["race_id", "horse_id"], keep=False)
    if duplicate_keys.any():
        if "collected_at" in prepared.columns:
            prepared["collected_at"] = pd.to_datetime(
                prepared["collected_at"], errors="coerce"
            )
            prepared = prepared.sort_values(
                ["race_id", "horse_id", "collected_at"],
                kind="stable",
                na_position="first",
            ).drop_duplicates(["race_id", "horse_id"], keep="last")
        else:
            prepared = prepared.drop_duplicates()
            if prepared.duplicated(["race_id", "horse_id"]).any():
                raise ValueError(
                    "History contains conflicting duplicate race/horse rows "
                    "without collected_at"
                )
    prepared = prepared.drop(columns=["collected_at"], errors="ignore")

    prepared["field_size"] = prepared.groupby("race_id")["horse_id"].transform("size")
    valid_finish = prepared["finish_position"].gt(0)
    denominator = prepared["field_size"] - 1
    prepared["relative_finish"] = np.where(
        valid_finish & denominator.gt(0),
        (prepared["finish_position"] - 1) / denominator,
        np.nan,
    )
    prepared["is_win"] = prepared["finish_position"].eq(1).where(valid_finish)
    prepared["is_top3"] = prepared["finish_position"].le(3).where(valid_finish)

    valid_last3f_count = prepared.groupby("race_id")["last_3f_time"].transform("count")
    last3f_rank = prepared.groupby("race_id")["last_3f_time"].rank(
        method="average", ascending=True, na_option="keep"
    )
    prepared["last3f_rank_rate"] = np.where(
        valid_last3f_count.gt(1),
        (last3f_rank - 1) / (valid_last3f_count - 1),
        np.nan,
    )
    return prepared.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)


def _settings(parameters: Mapping[str, Any]) -> tuple[float, int]:
    configured = parameters.get("recent_form", {})
    if not isinstance(configured, Mapping):
        raise ValueError("recent_form parameters must be a mapping")
    half_life_days = float(configured.get("half_life_days", 180.0))
    max_lookback_days = int(configured.get("max_lookback_days", 1095))
    if max_lookback_days <= 0:
        raise ValueError("max_lookback_days must be greater than zero")
    return half_life_days, max_lookback_days


class RecentFormGenerator(FeatureGenerator):
    """Aggregate point-in-time form without odds, body weight, or speed indexes."""

    name = "recent_form"
    version = "1.0.0"
    output_columns = (
        "recent_starts_total",
        "recent_days_since_last",
        "recent_finish_last",
        "recent_finish_mean_3",
        "recent_finish_mean_5",
        "recent_relative_finish_last",
        "recent_relative_finish_mean_3",
        "recent_relative_finish_mean_5",
        "recent_relative_finish_decay_mean",
        "recent_win_rate_5",
        "recent_top3_rate_5",
        "recent_win_rate_decay",
        "recent_top3_rate_decay",
        "recent_last3f_rank_rate_last",
        "recent_last3f_rank_rate_mean_3",
        "recent_last3f_rank_rate_decay_mean",
        "recent_relative_finish_std_5",
        "recent_finish_available_count_5",
        "recent_effective_starts",
    )

    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        _require_columns(targets, REQUIRED_TARGET_COLUMNS, "Targets")
        half_life_days, max_lookback_days = _settings(context.parameters)
        prepared = prepare_historical_performances(history)
        histories_by_horse = {
            horse_id: group
            for horse_id, group in prepared.groupby("horse_id", sort=False)
        }
        target_dates = pd.to_datetime(targets["race_date"], errors="coerce")
        if target_dates.isna().any():
            raise ValueError("Target race_date cannot be missing or invalid")

        rows: list[dict[str, object]] = []
        for position, (_, target) in enumerate(targets.iterrows()):
            target_date = target_dates.iloc[position]
            all_horse_history = histories_by_horse.get(target["horse_id"])
            if all_horse_history is None:
                horse_history = prepared.iloc[0:0].copy()
            else:
                horse_history = all_horse_history[
                    all_horse_history["race_date"].lt(target_date)
                ].copy()
            horse_history = horse_history.sort_values(
                ["race_date", "race_id"], kind="stable"
            )
            rows.append(
                self._aggregate(
                    target,
                    target_date,
                    horse_history,
                    half_life_days,
                    max_lookback_days,
                )
            )
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)

    def _aggregate(
        self,
        target: pd.Series,
        target_date: pd.Timestamp,
        horse_history: pd.DataFrame,
        half_life_days: float,
        max_lookback_days: int,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "race_id": target["race_id"],
            "horse_id": target["horse_id"],
            "recent_starts_total": float(len(horse_history)),
        }
        if horse_history.empty:
            result.update({column: np.nan for column in self.output_columns[1:]})
            result["recent_finish_available_count_5"] = 0.0
            result["recent_effective_starts"] = 0.0
            return result

        elapsed_days = (target_date - horse_history["race_date"]).dt.total_seconds() / 86_400
        result["recent_days_since_last"] = float(elapsed_days.iloc[-1])
        last_three = horse_history.tail(3)
        last_five = horse_history.tail(5)
        decay_history = horse_history[elapsed_days.le(max_lookback_days)].copy()
        decay_elapsed = elapsed_days.loc[decay_history.index]
        weights = decay_weights(decay_elapsed, half_life_days)

        result.update({
            "recent_finish_last": horse_history["finish_position"].iloc[-1],
            "recent_finish_mean_3": last_three["finish_position"].mean(),
            "recent_finish_mean_5": last_five["finish_position"].mean(),
            "recent_relative_finish_last": horse_history["relative_finish"].iloc[-1],
            "recent_relative_finish_mean_3": last_three["relative_finish"].mean(),
            "recent_relative_finish_mean_5": last_five["relative_finish"].mean(),
            "recent_relative_finish_decay_mean": weighted_mean(
                decay_history["relative_finish"], weights
            ),
            "recent_win_rate_5": last_five["is_win"].mean(),
            "recent_top3_rate_5": last_five["is_top3"].mean(),
            "recent_win_rate_decay": weighted_mean(decay_history["is_win"], weights),
            "recent_top3_rate_decay": weighted_mean(decay_history["is_top3"], weights),
            "recent_last3f_rank_rate_last": horse_history["last3f_rank_rate"].iloc[-1],
            "recent_last3f_rank_rate_mean_3": last_three["last3f_rank_rate"].mean(),
            "recent_last3f_rank_rate_decay_mean": weighted_mean(
                decay_history["last3f_rank_rate"], weights
            ),
            "recent_relative_finish_std_5": last_five["relative_finish"].std(ddof=0),
            "recent_finish_available_count_5": float(last_five["finish_position"].count()),
            "recent_effective_starts": effective_count(weights),
        })
        return result
