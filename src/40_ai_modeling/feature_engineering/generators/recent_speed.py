from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


class RecentSpeedGenerator(FeatureGenerator):
    """Aggregate only speed indexes recorded before each target race date."""

    name = "recent_speed"
    version = "1.0.0"
    output_columns = (
        "recent_speed_last",
        "recent_speed_mean_3",
        "recent_speed_mean_5",
        "recent_speed_max_5",
        "recent_speed_decay_mean",
        "recent_speed_std_5",
        "recent_speed_available_count_5",
        "recent_speed_effective_count",
    )

    def __init__(
        self,
        progress: Callable[[float], None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> None:
        self._progress = progress
        self._check_cancelled = check_cancelled

    def transform(
        self, targets: pd.DataFrame, history: pd.DataFrame, context: FeatureContext
    ) -> pd.DataFrame:
        target_required = {"race_id", "horse_id", "race_date"}
        history_required = {"horse_id", "race_date", "speed_index"}
        if missing := target_required.difference(targets.columns):
            raise ValueError(f"Recent-speed targets are missing: {sorted(missing)}")
        if missing := history_required.difference(history.columns):
            raise ValueError(f"Speed-index history is missing: {sorted(missing)}")
        settings = context.parameters.get("recent_speed", {})
        half_life = float(settings.get("half_life_days", 180.0))
        max_days = int(settings.get("max_lookback_days", 1095))
        if half_life <= 0 or max_days <= 0:
            raise ValueError("Recent-speed decay and lookback must be positive")

        target_dates = pd.to_datetime(targets["race_date"], errors="coerce")
        if target_dates.isna().any():
            raise ValueError("Recent-speed target race_date is invalid")
        performance = history[["horse_id", "race_date", "speed_index"]].copy()
        performance["race_date"] = pd.to_datetime(
            performance["race_date"], errors="coerce"
        )
        performance["speed_index"] = pd.to_numeric(
            performance["speed_index"], errors="coerce"
        )
        performance = performance.dropna(subset=["horse_id", "race_date"])

        histories: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        groups = performance.groupby("horse_id", sort=False)
        group_count = max(groups.ngroups, 1)
        for position, (horse_id, group) in enumerate(groups):
            if self._check_cancelled and position % 250 == 0:
                self._check_cancelled()
            if self._progress and position % 250 == 0:
                self._progress(0.2 * position / group_count)
            ordered = group.sort_values("race_date", kind="stable")
            histories[horse_id] = (
                ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                ordered["speed_index"].to_numpy(dtype=float),
            )

        rows: list[dict[str, object]] = []
        target_count = max(len(targets), 1)
        for position, target in enumerate(targets.itertuples(index=False)):
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            if self._progress and position % 1000 == 0:
                self._progress(0.2 + 0.8 * position / target_count)
            arrays = histories.get(target.horse_id)
            target_date = np.datetime64(target_dates.iloc[position].to_datetime64(), "ns")
            row: dict[str, object] = {
                "race_id": target.race_id,
                "horse_id": target.horse_id,
            }
            if arrays is None:
                row.update(self._empty())
                rows.append(row)
                continue
            dates, values = arrays
            end = int(np.searchsorted(dates, target_date, side="left"))
            prior_dates = dates[:end]
            prior_values = values[:end]
            valid_prior = ~np.isnan(prior_values)
            prior_dates = prior_dates[valid_prior]
            prior_values = prior_values[valid_prior]
            if not len(prior_values):
                row.update(self._empty())
                rows.append(row)
                continue

            elapsed = (target_date - prior_dates) / np.timedelta64(1, "D")
            decay_mask = elapsed <= max_days
            decay_values = prior_values[decay_mask]
            decay_elapsed = elapsed[decay_mask]
            weights = np.power(0.5, decay_elapsed / half_life)
            total_weight = float(weights.sum())
            last_five = prior_values[-5:]
            row.update({
                "recent_speed_last": float(prior_values[-1]),
                "recent_speed_mean_3": float(prior_values[-3:].mean()),
                "recent_speed_mean_5": float(last_five.mean()),
                "recent_speed_max_5": float(last_five.max()),
                "recent_speed_decay_mean": (
                    float(np.dot(decay_values, weights) / total_weight)
                    if total_weight > 0 else np.nan
                ),
                "recent_speed_std_5": float(last_five.std(ddof=0)),
                "recent_speed_available_count_5": float(len(last_five)),
                "recent_speed_effective_count": total_weight,
            })
            rows.append(row)
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)

    def _empty(self) -> dict[str, float]:
        values = {column: np.nan for column in self.output_columns}
        values["recent_speed_available_count_5"] = 0.0
        values["recent_speed_effective_count"] = 0.0
        return values
