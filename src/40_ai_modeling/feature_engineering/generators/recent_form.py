from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
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


def _nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if len(valid) else np.nan


def _nanstd(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid.std(ddof=0)) if len(valid) else np.nan


def _weighted_array_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = ~np.isnan(values) & ~np.isnan(weights)
    total_weight = float(weights[valid].sum())
    if not valid.any() or total_weight <= 0:
        return np.nan
    return float(np.dot(values[valid], weights[valid]) / total_weight)


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

    def __init__(
        self,
        progress: Callable[[float], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self._progress = progress
        self._check_cancelled = check_cancelled

    def transform(
        self,
        targets: pd.DataFrame,
        history: pd.DataFrame,
        context: FeatureContext,
    ) -> pd.DataFrame:
        _require_columns(targets, REQUIRED_TARGET_COLUMNS, "Targets")
        half_life_days, max_lookback_days = _settings(context.parameters)
        prepared = prepare_historical_performances(history)
        # Prediction targets contain only one race. Race-relative values must be
        # calculated with the full history first, but histories for horses that
        # are not in the target race do not need to be indexed afterward.
        target_horses = set(targets["horse_id"].dropna().tolist())
        prepared = prepared[prepared["horse_id"].isin(target_horses)]
        histories_by_horse = {}
        grouped = prepared.groupby("horse_id", sort=False)
        group_count = max(grouped.ngroups, 1)
        for group_position, (horse_id, group) in enumerate(grouped):
            if self._check_cancelled and group_position % 250 == 0:
                self._check_cancelled()
            if self._progress and group_position % 250 == 0:
                self._progress(0.25 * group_position / group_count)
            ordered = group.sort_values(["race_date", "race_id"], kind="stable")
            histories_by_horse[horse_id] = {
                "dates": ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                "finish": ordered["finish_position"].to_numpy(dtype=float),
                "relative": ordered["relative_finish"].to_numpy(dtype=float),
                "win": ordered["is_win"].to_numpy(dtype=float, na_value=np.nan),
                "top3": ordered["is_top3"].to_numpy(dtype=float, na_value=np.nan),
                "last3f": ordered["last3f_rank_rate"].to_numpy(dtype=float),
            }
        target_dates = pd.to_datetime(targets["race_date"], errors="coerce")
        if target_dates.isna().any():
            raise ValueError("Target race_date cannot be missing or invalid")

        rows: list[dict[str, object]] = []
        target_count = max(len(targets), 1)
        for position, target in enumerate(targets.itertuples(index=False)):
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            if self._progress and position % 1000 == 0:
                self._progress(0.25 + 0.75 * position / target_count)
            target_date = target_dates.iloc[position]
            arrays = histories_by_horse.get(target.horse_id)
            rows.append(
                self._aggregate(
                    target.race_id,
                    target.horse_id,
                    target_date,
                    arrays,
                    half_life_days,
                    max_lookback_days,
                )
            )
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)

    def _aggregate(
        self,
        race_id: object,
        horse_id: object,
        target_date: pd.Timestamp,
        arrays: dict[str, np.ndarray] | None,
        half_life_days: float,
        max_lookback_days: int,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "race_id": race_id,
            "horse_id": horse_id,
        }
        if arrays is None:
            result["recent_starts_total"] = 0.0
            result.update({column: np.nan for column in self.output_columns[1:]})
            result["recent_finish_available_count_5"] = 0.0
            result["recent_effective_starts"] = 0.0
            return result

        target_value = np.datetime64(target_date.to_datetime64(), "ns")
        starts = int(np.searchsorted(arrays["dates"], target_value, side="left"))
        result["recent_starts_total"] = float(starts)
        if starts == 0:
            result.update({column: np.nan for column in self.output_columns[1:]})
            result["recent_finish_available_count_5"] = 0.0
            result["recent_effective_starts"] = 0.0
            return result

        dates = arrays["dates"][:starts]
        finish = arrays["finish"][:starts]
        relative = arrays["relative"][:starts]
        win = arrays["win"][:starts]
        top3 = arrays["top3"][:starts]
        last3f = arrays["last3f"][:starts]
        elapsed_days = (target_value - dates) / np.timedelta64(1, "D")
        decay_mask = elapsed_days <= max_lookback_days
        weights = np.power(0.5, elapsed_days[decay_mask] / half_life_days)

        result.update({
            "recent_days_since_last": float(elapsed_days[-1]),
            "recent_finish_last": finish[-1],
            "recent_finish_mean_3": _nanmean(finish[-3:]),
            "recent_finish_mean_5": _nanmean(finish[-5:]),
            "recent_relative_finish_last": relative[-1],
            "recent_relative_finish_mean_3": _nanmean(relative[-3:]),
            "recent_relative_finish_mean_5": _nanmean(relative[-5:]),
            "recent_relative_finish_decay_mean": _weighted_array_mean(
                relative[decay_mask], weights
            ),
            "recent_win_rate_5": _nanmean(win[-5:]),
            "recent_top3_rate_5": _nanmean(top3[-5:]),
            "recent_win_rate_decay": _weighted_array_mean(win[decay_mask], weights),
            "recent_top3_rate_decay": _weighted_array_mean(top3[decay_mask], weights),
            "recent_last3f_rank_rate_last": last3f[-1],
            "recent_last3f_rank_rate_mean_3": _nanmean(last3f[-3:]),
            "recent_last3f_rank_rate_decay_mean": _weighted_array_mean(
                last3f[decay_mask], weights
            ),
            "recent_relative_finish_std_5": _nanstd(relative[-5:]),
            "recent_finish_available_count_5": float(np.count_nonzero(~np.isnan(finish[-5:]))),
            "recent_effective_starts": float(weights.sum()),
        })
        return result
