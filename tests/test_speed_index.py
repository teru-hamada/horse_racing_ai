from __future__ import annotations

import math

import pandas as pd

from src.public_api import FeatureContext, SpeedIndexGenerator


def _history() -> pd.DataFrame:
    rows = []
    for number in range(1, 13):
        race_date = f"2026-01-{number:02d}"
        for horse_number, finish, time_seconds, weight in (
            (1, 1, 100.0, 55.0),
            (2, 2, 102.0, 57.0),
        ):
            rows.append({
                "race_id": f"r{number}",
                "horse_id": f"h{number}-{horse_number}",
                "race_date": race_date,
                "course_name": "東京",
                "surface": "芝",
                "distance": 1600,
                "track_condition": "良",
                "carried_weight": weight,
                "finish_position": finish,
                "time_seconds": time_seconds,
                "collected_at": "2026-02-01",
            })
    return pd.DataFrame(rows)


def _generate(history: pd.DataFrame) -> pd.DataFrame:
    context = FeatureContext("speed_index", "1.0.0", {})
    return SpeedIndexGenerator().transform(history, history, context)


def test_speed_index_uses_only_prior_standard_times():
    result = _generate(_history())
    first = result[result["race_id"].eq("r1")].iloc[0]
    race_11 = result[result["race_id"].eq("r11")].sort_values("horse_id")

    assert math.isnan(first["speed_index"])
    assert first["speed_fallback_level"] == "unavailable"
    assert (race_11["speed_fallback_level"] == "course_surface_distance_condition").all()
    assert (race_11["speed_reference_race_count"] == 10).all()
    assert math.isclose(race_11["speed_standard_time"].iloc[0], 100.0)
    assert math.isclose(race_11["speed_index"].iloc[0], 100.0)
    assert math.isclose(race_11["speed_index"].iloc[1], 83.0)
    assert not race_11["speed_index_was_clipped"].any()


def test_extreme_slow_time_is_clipped_but_raw_value_is_retained():
    history = _history()
    history.loc[
        history["race_id"].eq("r11") & history["finish_position"].eq(2),
        "time_seconds",
    ] = 200.0

    result = _generate(history)
    slow = result[
        result["race_id"].eq("r11") & result["horse_id"].eq("h11-2")
    ].iloc[0]

    assert slow["speed_index"] == 0.0
    assert slow["speed_index_raw"] < 0
    assert bool(slow["speed_index_was_clipped"])


def test_other_race_on_same_day_does_not_change_standard():
    history = _history()
    extra = history[history["race_id"].eq("r11")].copy()
    extra["race_id"] = "r11-extra"
    extra["horse_id"] = ["extra-1", "extra-2"]
    extra.loc[extra["finish_position"].eq(1), "time_seconds"] = 150.0
    combined = pd.concat([history, extra], ignore_index=True)

    before = _generate(history)
    after = _generate(combined)
    before_standard = before[before["race_id"].eq("r11")]["speed_standard_time"]
    after_standard = after[after["race_id"].eq("r11")]["speed_standard_time"]

    assert before_standard.tolist() == after_standard.tolist()


def test_speed_index_does_not_use_market_or_body_weight_change():
    history = _history()
    history["odds"] = 1.1
    history["popularity"] = 1
    history["body_weight_change"] = 20

    result = _generate(history)

    assert not set(result.columns).intersection(
        {"odds", "popularity", "body_weight_change"}
    )
