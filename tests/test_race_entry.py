from __future__ import annotations

import math

import pandas as pd

from src.public_api import FeatureContext, RaceEntryGenerator


def _entries() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_id": "old", "horse_id": "h1", "race_date": "2025-12-01",
         "course_name": "東京", "race_number": 1, "surface": "芝", "distance": 1600,
         "track_condition": "良",
         "horse_number": 1, "frame_number": 1, "sex": "牡", "age": 3,
         "carried_weight": 55},
        {"race_id": "new", "horse_id": "h1", "race_date": "2026-01-01",
         "course_name": "中山", "race_number": 11, "surface": "芝", "distance": 2000,
         "track_condition": "稍重",
         "horse_number": 2, "frame_number": 2, "sex": "牡", "age": 4,
         "carried_weight": 57},
        {"race_id": "new", "horse_id": "h2", "race_date": "2026-01-01",
         "course_name": "中山", "race_number": 11, "surface": "芝", "distance": 2000,
         "track_condition": "稍重",
         "horse_number": 1, "frame_number": 1, "sex": "牝", "age": 3,
         "carried_weight": 55},
    ])


def test_race_entry_creates_current_and_relative_conditions():
    entries = _entries()
    targets = entries.loc[entries["race_id"].eq("new"), ["race_id", "horse_id", "race_date"]]
    result = RaceEntryGenerator().transform(
        targets, entries, FeatureContext("race_entry", "1.1.0", {})
    )
    h1 = result[result["horse_id"].eq("h1")].iloc[0]

    assert h1["race_entry_course"] == "中山"
    assert h1["race_entry_track_condition"] == "稍重"
    assert h1["race_entry_field_size"] == 2
    assert h1["race_entry_horse_number_ratio"] == 1
    assert h1["race_entry_carried_weight_diff_field_mean"] == 1
    assert h1["race_entry_distance_change_from_last"] == 400


def test_same_day_distance_is_not_used_as_previous_race():
    entries = _entries()
    targets = entries.loc[
        entries["race_id"].eq("new") & entries["horse_id"].eq("h2"),
        ["race_id", "horse_id", "race_date"],
    ]
    row = RaceEntryGenerator().transform(
        targets, entries, FeatureContext("race_entry", "1.1.0", {})
    ).iloc[0]

    assert math.isnan(row["race_entry_distance_change_from_last"])


def test_race_day_only_columns_are_not_generated():
    output = set(RaceEntryGenerator.output_columns)

    assert output.isdisjoint({
        "weather", "odds", "popularity",
        "body_weight", "body_weight_change",
    })
