from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..base import FEATURE_ID_COLUMNS, FeatureGenerator
from ..context import FeatureContext


ENTRY_COLUMNS = (
    "race_id", "horse_id", "race_date", "course_name", "race_number",
    "surface", "distance", "track_condition", "horse_number", "frame_number",
    "sex", "age", "carried_weight",
)


def prepare_race_entries(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize fields known when the race entry table is available."""

    missing = set(ENTRY_COLUMNS).difference(records.columns)
    if missing:
        raise ValueError(f"Race entries are missing columns: {sorted(missing)}")
    columns = list(ENTRY_COLUMNS)
    if "collected_at" in records:
        columns.append("collected_at")
    entries = records.loc[:, columns].copy()
    entries["race_date"] = pd.to_datetime(entries["race_date"], errors="coerce")
    if entries[["race_id", "horse_id", "race_date"]].isna().any().any():
        raise ValueError("Race-entry keys and race_date cannot be missing")
    for column in (
        "race_number", "distance", "horse_number", "frame_number", "age",
        "carried_weight",
    ):
        entries[column] = pd.to_numeric(entries[column], errors="coerce")
    if entries.duplicated(["race_id", "horse_id"]).any():
        if "collected_at" not in entries:
            raise ValueError("Race entries contain duplicate race_id + horse_id keys")
        entries["collected_at"] = pd.to_datetime(entries["collected_at"], errors="coerce")
        entries = entries.sort_values(
            ["race_id", "horse_id", "collected_at"], kind="stable", na_position="first"
        ).drop_duplicates(["race_id", "horse_id"], keep="last")
    return entries.drop(columns="collected_at", errors="ignore").reset_index(drop=True)


class RaceEntryGenerator(FeatureGenerator):
    """Create entry-table features plus the supplied track condition."""

    name = "race_entry"
    version = "1.1.0"
    output_columns = (
        "race_entry_course",
        "race_entry_race_number",
        "race_entry_surface",
        "race_entry_track_condition",
        "race_entry_distance",
        "race_entry_age",
        "race_entry_sex",
        "race_entry_carried_weight",
        "race_entry_frame_number",
        "race_entry_horse_number",
        "race_entry_field_size",
        "race_entry_frame_position_ratio",
        "race_entry_horse_number_ratio",
        "race_entry_carried_weight_diff_field_mean",
        "race_entry_distance_change_from_last",
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
        entries = prepare_race_entries(history)
        target_keys = targets.loc[:, ["race_id", "horse_id"]].copy()
        if target_keys.duplicated().any():
            raise ValueError("Race-entry targets contain duplicate keys")
        current = target_keys.merge(
            entries, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
        if current["race_date"].isna().any():
            raise ValueError("Some race-entry targets do not exist in history")

        race_groups = current.groupby("race_id", sort=False)
        current["_field_size"] = race_groups["horse_id"].transform("size")
        current["_mean_weight"] = race_groups["carried_weight"].transform("mean")

        histories: dict[object, tuple[np.ndarray, np.ndarray]] = {}
        for horse_id, group in entries.groupby("horse_id", sort=False):
            ordered = group.sort_values(["race_date", "race_id"], kind="stable")
            histories[horse_id] = (
                ordered["race_date"].to_numpy(dtype="datetime64[ns]"),
                ordered["distance"].to_numpy(dtype=float),
            )

        previous_distances = np.full(len(current), np.nan)
        total = max(len(current), 1)
        for position, row in enumerate(current.itertuples(index=False)):
            if self._check_cancelled and position % 1000 == 0:
                self._check_cancelled()
            if self._progress and position % 1000 == 0:
                self._progress(position / total)
            dates, distances = histories[row.horse_id]
            target_date = np.datetime64(row.race_date.to_datetime64(), "ns")
            end = int(np.searchsorted(dates, target_date, side="left"))
            if end:
                valid = distances[:end][~np.isnan(distances[:end])]
                if len(valid):
                    previous_distances[position] = valid[-1]

        result = pd.DataFrame({
            "race_id": current["race_id"],
            "horse_id": current["horse_id"],
            "race_entry_course": current["course_name"].astype("string"),
            "race_entry_race_number": current["race_number"],
            "race_entry_surface": current["surface"].astype("string"),
            "race_entry_track_condition": current["track_condition"].astype("string"),
            "race_entry_distance": current["distance"],
            "race_entry_age": current["age"],
            "race_entry_sex": current["sex"].astype("string"),
            "race_entry_carried_weight": current["carried_weight"],
            "race_entry_frame_number": current["frame_number"],
            "race_entry_horse_number": current["horse_number"],
            "race_entry_field_size": current["_field_size"].astype(float),
            "race_entry_frame_position_ratio": np.where(
                current["_field_size"].gt(0),
                current["frame_number"] / current["_field_size"],
                np.nan,
            ),
            "race_entry_horse_number_ratio": np.where(
                current["_field_size"].gt(0),
                current["horse_number"] / current["_field_size"],
                np.nan,
            ),
            "race_entry_carried_weight_diff_field_mean": (
                current["carried_weight"] - current["_mean_weight"]
            ),
            "race_entry_distance_change_from_last": (
                current["distance"].to_numpy(dtype=float) - previous_distances
            ),
        })
        if self._progress:
            self._progress(1.0)
        return result.loc[:, FEATURE_ID_COLUMNS + self.output_columns]
