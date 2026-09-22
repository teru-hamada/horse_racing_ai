from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from importlib import import_module

from src.public_api import compare_prediction_date, compare_recommended_bets

comparison_module = import_module('src.50_result_comparison.prediction_comparison')


@pytest.mark.parametrize("bet_type,selection,judgment,payout", [
    ("win", "7", "返還", 100),
    ("place", "7", "返還", 100),
    ("quinella", "1-7", "返還", 100),
    ("wide", "7-1", "返還", 100),
    ("exacta", "7-1", "返還", 100),
    ("trio", "1-2-7", "返還", 100),
    ("trifecta", "7-2-1", "返還", 100),
    ("bracket_quinella", "4-4", "返還", 100),
    ("bracket_quinella", "4-1", "的中", 600),
    ("bracket_quinella", "4-7", "不的中", 0),
    ("bracket_quinella", "2-8", "返還", 100),
    ("win", "15", "不的中", 0),  # did not finish: no refund
    ("win", "1", "的中", 300),
])
def test_refund_settlement_distinguishes_horse_frame_and_same_frame(bet_type, selection, judgment, payout):
    bets = pd.DataFrame([{"race_id": "r1", "bet_type": bet_type, "selection": selection}])
    result = compare_recommended_bets(bets, {"r1": {"win:1": 300, "bracket_quinella:1-4": 600}},
        {"r1": {"horse_numbers": [7], "frame_numbers": [8], "same_frame_numbers": [4]}})
    assert result.iloc[0].bet_result == judgment
    assert result.iloc[0].payout_per_100 == payout


@pytest.mark.parametrize("status", ["scratched", "excluded", "did_not_finish", "disqualified"])
def test_comparison_retains_non_finisher_status_and_does_not_count_it_as_hit(status):
    prediction = _predictions().query("race_id == 'r1'")
    actual = prediction[["horse_id", "horse_number"]].assign(
        finish_position=[1, None, 2, 3], result_status=["finished", status, "finished", "finished"])
    if status in {"scratched", "excluded"}:
        actual.attrs["official_refunds"] = {"horse_numbers": [2]}
    result = comparison_module.compare_prediction_with_finish(prediction, actual)
    row = result.set_index("horse_number").loc[2]
    assert row.result_status == status
    assert pd.isna(row.finish_position)
    assert not row.actual_top3
    assert not row.top3_hit


@pytest.mark.parametrize("status,position,refunds", [
    ("unknown", None, {}), ("finished", None, {}),
    ("finished", 0, {}), ("finished", 1.5, {}),
    ("excluded", 2, {"horse_numbers": [2]}),
    ("excluded", None, {}), ("scratched", None, {"horse_numbers": [3]}),
    ("did_not_finish", None, {"horse_numbers": [2]}),
    (pd.NA, None, {}),
])
def test_unknown_or_inconsistent_results_still_fail(status, position, refunds):
    actual = pd.DataFrame({"horse_number": [1, 2], "finish_position": [1, position],
                           "result_status": pd.array(["finished", status], dtype="string")})
    actual.attrs["official_refunds"] = refunds
    with pytest.raises(ValueError):
        comparison_module.validated_result_status(actual)


def test_recommended_bets_use_official_payouts_and_preserve_order():
    bets = pd.DataFrame([
        {"race_id": "r1", "bet_type": kind, "selection": selection}
        for kind, selection in [
            ("place", "1"), ("place", "4"), ("wide", "3-1"),
            ("exacta", "2-1"), ("trifecta", "1-2-3"),
        ]
    ] + [{"race_id": "r2", "bet_type": "place", "selection": "1"}])
    result = compare_recommended_bets(bets, {"r1": {
        "place:1": 150, "wide:1-3": 420, "exacta:1-2": 800,
    }})
    assert result["bet_result"].tolist() == [
        "的中", "不的中", "的中", "不的中", "未確認", "未確認",
    ]
    assert result["payout_per_100"].iloc[:4].tolist() == [150, 0, 420, 0]
    assert result["payout_per_100"].iloc[4:].isna().all()
    assert "bet_result" not in bets


def test_empty_recommended_bets():
    result = compare_recommended_bets(pd.DataFrame(), {})
    assert result.empty
    assert {"bet_result", "payout_per_100"}.issubset(result.columns)


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
        "official_payouts": {},
        "official_refunds": {},
    }
