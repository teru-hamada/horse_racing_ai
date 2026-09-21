from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


REQUIRED_COLUMNS = (
    "race_id", "horse_id", "race_date", "course_name", "surface", "distance",
    "track_condition", "carried_weight", "finish_position", "time_seconds",
)


def prepare_speed_performances(records: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(records.columns)
    if missing:
        raise ValueError(f"Speed-index records are missing columns: {sorted(missing)}")
    columns = list(REQUIRED_COLUMNS)
    if "collected_at" in records.columns:
        columns.append("collected_at")
    frame = records.loc[:, columns].copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    for column in ("distance", "carried_weight", "finish_position", "time_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["race_id", "horse_id"]].isna().any().any():
        raise ValueError("Speed-index race_id and horse_id cannot be missing")
    duplicates = frame.duplicated(["race_id", "horse_id"], keep=False)
    if duplicates.any():
        if "collected_at" not in frame.columns:
            frame = frame.drop_duplicates()
            if frame.duplicated(["race_id", "horse_id"]).any():
                raise ValueError("Conflicting duplicate speed-index rows lack collected_at")
        else:
            frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce")
            frame = frame.sort_values(
                ["race_id", "horse_id", "collected_at"], kind="stable",
                na_position="first",
            ).drop_duplicates(["race_id", "horse_id"], keep="last")
    return frame.drop(columns=["collected_at"], errors="ignore").sort_values(
        ["race_date", "race_id", "horse_id"], kind="stable"
    ).reset_index(drop=True)


@dataclass
class _ReferenceSeries:
    days: list[int]
    paces: list[float]

    def append(self, day: int, pace: float) -> None:
        self.days.append(day)
        self.paces.append(pace)

    def estimate(
        self, target_day: int, half_life_days: float, max_lookback_days: int
    ) -> tuple[float, int, float]:
        start = bisect_left(self.days, target_day - max_lookback_days)
        days = np.asarray(self.days[start:], dtype=float)
        values = np.asarray(self.paces[start:], dtype=float)
        if not len(values):
            return np.nan, 0, 0.0
        weights = np.power(0.5, (target_day - days) / half_life_days)
        total = float(weights.sum())
        return float(np.dot(values, weights) / total), len(values), total


class SpeedIndexGenerator(FeatureGenerator):
    """Rate each completed performance using standards known before its date."""

    name = "speed_index"
    version = "1.0.0"
    output_columns = (
        "speed_index",
        "speed_index_raw",
        "speed_index_was_clipped",
        "speed_standard_time",
        "speed_time_difference",
        "speed_weight_adjustment",
        "speed_track_adjustment",
        "speed_reference_race_count",
        "speed_reference_effective_count",
        "speed_fallback_level",
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
        prepared = prepare_speed_performances(history)
        settings = context.parameters.get("speed_index", {})
        half_life = float(settings.get("half_life_days", 730.0))
        max_days = int(settings.get("max_lookback_days", 3650))
        seconds_scale = float(settings.get("seconds_scale_at_1600m", 10.0))
        weight_points = float(settings.get("weight_points_per_kg", 1.5))
        if half_life <= 0 or max_days <= 0 or seconds_scale <= 0:
            raise ValueError("Speed-index decay, lookback and scale must be positive")

        target_keys = set(zip(targets["race_id"], targets["horse_id"]))
        result_by_key: dict[tuple[object, object], dict[str, object]] = {}
        references: dict[tuple[object, ...], _ReferenceSeries] = {}

        def reference(key: tuple[object, ...]) -> _ReferenceSeries:
            return references.setdefault(key, _ReferenceSeries([], []))

        dated = prepared[prepared["race_date"].notna()]
        date_groups = list(dated.groupby("race_date", sort=True))
        total_dates = max(len(date_groups), 1)
        for date_position, (race_date, date_frame) in enumerate(date_groups):
            if self._check_cancelled and date_position % 10 == 0:
                self._check_cancelled()
            if self._progress and date_position % 10 == 0:
                self._progress(date_position / total_dates)
            target_day = int(race_date.to_datetime64().astype("datetime64[D]").astype(int))
            pending_references: list[tuple[tuple[object, ...], float]] = []

            for race_id, race in date_frame.groupby("race_id", sort=False):
                first = race.iloc[0]
                course = first["course_name"]
                surface = first["surface"]
                distance = first["distance"]
                condition = first["track_condition"]
                valid_race = (
                    pd.notna(course) and pd.notna(surface) and pd.notna(distance)
                    and float(distance) > 0
                )
                estimate = (np.nan, 0, 0.0, "unavailable", 0.0)
                if valid_race:
                    distance_value = float(distance)
                    band = int(distance_value // 200 * 200)
                    keys = [
                        (("condition", course, surface, distance_value, condition), 10, "course_surface_distance_condition"),
                        (("exact", course, surface, distance_value), 20, "course_surface_distance"),
                        (("course_band", course, surface, band), 30, "course_surface_distance_band"),
                        (("surface_band", surface, band), 50, "surface_distance_band"),
                        (("surface", surface), 100, "surface"),
                    ]
                    chosen = None
                    for key, minimum, label in keys:
                        pace, count, effective = reference(key).estimate(
                            target_day, half_life, max_days
                        )
                        if count >= minimum:
                            chosen = (pace, count, effective, label)
                            break
                    if chosen is not None:
                        pace, count, effective, level = chosen
                        standard_time = pace * distance_value / 1000.0
                        track_adjustment = 0.0
                        if level == "course_surface_distance_condition":
                            base_pace, base_count, _ = reference(
                                ("exact", course, surface, distance_value)
                            ).estimate(target_day, half_life, max_days)
                            if base_count >= 20:
                                track_adjustment = (
                                    pace - base_pace
                                ) * distance_value / 1000.0
                        estimate = (
                            standard_time, count, effective, level, track_adjustment
                        )

                standard_time, count, effective, level, track_adjustment = estimate
                for row in race.itertuples(index=False):
                    key = (row.race_id, row.horse_id)
                    if key not in target_keys:
                        continue
                    actual_time = float(row.time_seconds) if pd.notna(row.time_seconds) else np.nan
                    carried_weight = (
                        float(row.carried_weight) if pd.notna(row.carried_weight) else np.nan
                    )
                    difference = (
                        float(standard_time - actual_time)
                        if pd.notna(standard_time) and pd.notna(actual_time) else np.nan
                    )
                    weight_adjustment = (
                        (carried_weight - 55.0) * weight_points
                        if pd.notna(carried_weight) else np.nan
                    )
                    raw_score = (
                        100.0
                        + difference * seconds_scale * 1600.0 / float(distance)
                        + weight_adjustment
                        if valid_race and pd.notna(difference) and pd.notna(weight_adjustment)
                        else np.nan
                    )
                    score = (
                        float(np.clip(raw_score, 0.0, 200.0))
                        if pd.notna(raw_score) else np.nan
                    )
                    result_by_key[key] = {
                        "race_id": row.race_id,
                        "horse_id": row.horse_id,
                        "speed_index": score,
                        "speed_index_raw": raw_score,
                        "speed_index_was_clipped": (
                            bool(score != raw_score) if pd.notna(raw_score) else False
                        ),
                        "speed_standard_time": standard_time,
                        "speed_time_difference": difference,
                        "speed_weight_adjustment": weight_adjustment,
                        "speed_track_adjustment": track_adjustment,
                        "speed_reference_race_count": float(count),
                        "speed_reference_effective_count": float(effective),
                        "speed_fallback_level": level,
                    }

                winners = race[
                    race["finish_position"].eq(1)
                    & race["time_seconds"].notna()
                    & race["time_seconds"].gt(0)
                ]
                if valid_race and not winners.empty:
                    winning_pace = float(winners.iloc[0]["time_seconds"]) / float(distance) * 1000.0
                    band = int(float(distance) // 200 * 200)
                    for key in (
                        ("condition", course, surface, float(distance), condition),
                        ("exact", course, surface, float(distance)),
                        ("course_band", course, surface, band),
                        ("surface_band", surface, band),
                        ("surface", surface),
                    ):
                        pending_references.append((key, winning_pace))

            # Update only after every race on the date has been scored.
            for key, pace in pending_references:
                reference(key).append(target_day, pace)

        rows = []
        for key in zip(targets["race_id"], targets["horse_id"]):
            rows.append(result_by_key.get(key, {
                "race_id": key[0], "horse_id": key[1],
                **{column: np.nan for column in self.output_columns[:-1]},
                "speed_fallback_level": "unavailable",
            }))
        if self._progress:
            self._progress(1.0)
        return pd.DataFrame(rows, columns=FEATURE_ID_COLUMNS + self.output_columns)
