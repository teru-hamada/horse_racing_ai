# 検証対象：出馬表条件、頭数による相対位置、レース内の斤量差、前走距離差を確認する。
# 同日データを前走扱いしないことと、当日だけの情報を出力しないことを検証する。
#
from __future__ import annotations

import math

import pandas as pd

from src.public_api import FeatureContext, RaceEntryGenerator


# 出馬表条件の計算に必要な列を持つテストデータを作る。
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


# コース・馬場・頭数、馬番の相対位置、平均との差斤量、前走距離差の具体値を確認する。
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


# 同日の出走距離を前走として参照せず、過去走がなければ距離差はNaNとなる。
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


# 天気・オッズ・人気・馬体重など、採用しない当日列を出力定義へ含めない。
def test_race_day_only_columns_are_not_generated():
    output = set(RaceEntryGenerator.output_columns)

    assert output.isdisjoint({
        "weather", "odds", "popularity",
        "body_weight", "body_weight_change",
    })
