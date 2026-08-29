from __future__ import annotations

from datetime import date

import pandas as pd

from src.public_api import compare_prediction_date


def _predictions() -> pd.DataFrame:
    rows = []
    for race_number, race_id in enumerate(("r1", "r2"), start=1):
        for horse_number in range(1, 5):
            rows.append({
                "race_id": race_id,
                "course_name": "札幌",
                "race_number": race_number,
                "race_name": f"テスト{race_number}",
                "horse_id": f"{race_id}-h{horse_number}",
                "horse_number": horse_number,
                "prediction_rank": horse_number,
                "top3_probability": 1.0 / horse_number,
            })
    return pd.DataFrame(rows)


def test_date_comparison_keeps_success_when_one_race_is_unavailable():
    calls = []

    def fetch_result(race_id: str, race_date: date) -> pd.DataFrame:
        calls.append((race_id, race_date))
        if race_id == "r2":
            raise ValueError("結果未確定")
        return pd.DataFrame([
            {"horse_id": f"r1-h{number}", "horse_number": number,
             "finish_position": number}
            for number in range(1, 5)
        ])

    comparison, summary = compare_prediction_date(
        _predictions(), date(2026, 8, 29), fetch_result
    )

    assert [race_id for race_id, _ in calls] == ["r1", "r2"]
    assert comparison["race_id"].unique().tolist() == ["r1"]
    assert comparison["top3_hit"].sum() == 3
    assert summary == {
        "requested_races": 2,
        "compared_races": 1,
        "failed_races": 1,
        "top3_hit_count": 3,
        "perfect_top3_races": 1,
        "failures": [{"race_id": "r2", "message": "結果未確定"}],
    }
