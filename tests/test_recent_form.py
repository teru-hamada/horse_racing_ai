from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from src.public_api import (
    FeatureContext,
    RecentFormGenerator,
    load_features,
    save_features,
)


def _history() -> pd.DataFrame:
    rows = []
    finishes = {
        "2026-01-01": [4, 1, 2, 3],
        "2026-01-02": [2, 1, 3, 4],
        "2026-01-03": [1, 2, 3, 4],
    }
    last3f = {
        "2026-01-01": [39.0, 36.0, 37.0, 38.0],
        "2026-01-02": [36.5, 36.0, 37.0, 38.0],
        "2026-01-03": [35.0, 36.0, 37.0, 38.0],
    }
    for race_number, race_date in enumerate(finishes, start=1):
        for horse_number in range(4):
            rows.append({
                "race_id": f"r{race_number}",
                "horse_id": "h1" if horse_number == 0 else f"other-{race_number}-{horse_number}",
                "race_date": race_date,
                "finish_position": finishes[race_date][horse_number],
                "last_3f_time": last3f[race_date][horse_number],
                "odds": 999.0,
                "popularity": 1,
                "body_weight": 500,
            })
    return pd.DataFrame(rows)


def _generate(target_date: str, horse_id: str = "h1") -> pd.DataFrame:
    targets = pd.DataFrame([{
        "race_id": "target",
        "horse_id": horse_id,
        "race_date": target_date,
    }])
    context = FeatureContext(
        "baseline",
        "1.1.0",
        {"recent_form": {"half_life_days": 180, "max_lookback_days": 1095}},
    )
    return RecentFormGenerator().transform(targets, _history(), context)


def test_recent_form_aggregates_prior_races():
    row = _generate("2026-01-04").iloc[0]

    assert row["recent_starts_total"] == 3
    assert row["recent_days_since_last"] == 1
    assert row["recent_finish_last"] == 1
    assert row["recent_finish_mean_3"] == 7 / 3
    assert math.isclose(row["recent_relative_finish_mean_3"], 4 / 9)
    assert row["recent_win_rate_5"] == 1 / 3
    assert row["recent_top3_rate_5"] == 2 / 3
    assert row["recent_last3f_rank_rate_last"] == 0
    assert row["recent_finish_available_count_5"] == 3
    assert 2.9 < row["recent_effective_starts"] < 3.0


def test_same_day_result_is_not_used():
    before_change = _generate("2026-01-03")
    changed = _history()
    changed.loc[changed["race_id"].eq("r3") & changed["horse_id"].eq("h1"), "finish_position"] = 4
    targets = pd.DataFrame([{
        "race_id": "r3",
        "horse_id": "h1",
        "race_date": "2026-01-03",
    }])
    context = FeatureContext("baseline", "1", {})
    after_change = RecentFormGenerator().transform(targets, changed, context)

    pd.testing.assert_frame_equal(
        before_change.drop(columns=["race_id", "horse_id"]),
        after_change.drop(columns=["race_id", "horse_id"]),
    )
    assert before_change["recent_starts_total"].iloc[0] == 2


def test_new_horse_keeps_aggregates_missing_and_counts_zero():
    row = _generate("2026-01-04", horse_id="new-horse").iloc[0]

    assert row["recent_starts_total"] == 0
    assert row["recent_effective_starts"] == 0
    assert row["recent_finish_available_count_5"] == 0
    assert math.isnan(row["recent_days_since_last"])
    assert math.isnan(row["recent_top3_rate_decay"])


def test_recent_form_does_not_output_forbidden_market_or_weight_features():
    columns = set(_generate("2026-01-04").columns)
    assert not columns.intersection({
        "odds", "popularity", "body_weight", "body_weight_change", "speed_index"
    })


def test_latest_collected_duplicate_is_used():
    history = _history()
    original = history.iloc[[0]].copy()
    original["collected_at"] = "2026-01-05"
    corrected = original.copy()
    corrected["finish_position"] = 1
    corrected["collected_at"] = "2026-01-06"
    history["collected_at"] = "2026-01-05"
    history = pd.concat([history, corrected], ignore_index=True)
    targets = pd.DataFrame([{
        "race_id": "target", "horse_id": "h1", "race_date": "2026-01-02"
    }])

    result = RecentFormGenerator().transform(
        targets, history, FeatureContext("baseline", "1", {})
    )

    assert result["recent_finish_last"].iloc[0] == 1


def test_recent_form_can_be_saved_in_versioned_feature_store(tmp_path):
    feature_db = tmp_path / "features.duckdb"
    features = _generate("2026-01-04")
    features["race_date"] = "2026-01-04"
    features["feature_set_name"] = "baseline"
    features["feature_set_version"] = "1.1.0"
    features["history_cutoff"] = datetime(2026, 1, 4)
    features["feature_run_id"] = "recent-form-test"
    features["generated_at"] = datetime(2026, 1, 5)

    save_features(features, feature_db)
    loaded = load_features("baseline", "1.1.0", feature_db)

    assert len(loaded) == 1
    assert loaded["recent_finish_last"].iloc[0] == 1
