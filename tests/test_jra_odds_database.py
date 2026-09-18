from dataclasses import replace
from datetime import date
from importlib import import_module

import pandas as pd
import pytest


parser = import_module("src.20_scrapers_database_creation.jra_odds_parser")
storage = import_module("src.00_common.storage")
betting = import_module("src.30_ai_modeling.betting")


def test_parse_win_place_html(tmp_path):
    path = tmp_path / "202606040501_win_place.html"
    path.write_text(
        """
        <table class="tanpuku"><tbody><tr>
          <td class="num">1</td><td class="odds_tan">7.3</td>
          <td class="odds_fuku"><span class="min">1.1</span>
          <span class="max">4.1</span></td>
        </tr></tbody></table>
        """,
        encoding="utf-8",
    )
    result = parser.parse_odds_html(path, date(2026, 9, 19))

    assert set(result["bet_type"]) == {"win", "place"}
    assert result.set_index("bet_type").loc["win", "odds_min"] == 7.3
    assert result.set_index("bet_type").loc["place", "odds_max"] == 4.1


def test_parse_combination_html(tmp_path):
    path = tmp_path / "202606040501_trio.html"
    path.write_text(
        """
        <table class="fuku3"><caption>1-2</caption><tbody>
          <tr><th>3</th><td>122.8</td></tr>
        </tbody></table>
        """,
        encoding="utf-8",
    )
    result = parser.parse_odds_html(path, date(2026, 9, 19)).iloc[0]

    assert (result.selection_1, result.selection_2, result.selection_3) == (1, 2, 3)
    assert result.odds_min == 122.8


def test_parse_bracket_quinella_html(tmp_path):
    path = tmp_path / "202606040501_bracket_quinella.html"
    path.write_text(
        """
        <table class="basic narrow-xy waku">
          <caption class="waku1"><img alt="枠1白"></caption>
          <tbody><tr><th>2</th><td>8.4</td></tr></tbody>
        </table>
        """,
        encoding="utf-8",
    )
    result = parser.parse_odds_html(path, date(2026, 9, 19)).iloc[0]

    assert result.bet_type == "bracket_quinella"
    assert (result.selection_1, result.selection_2) == (1, 2)
    assert result.odds_min == 8.4


def test_save_and_replace_race_odds(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage, "PATHS", replace(storage.PATHS, database=tmp_path / "racing.duckdb")
    )
    frame = pd.DataFrame(
        [{
            "race_id": "r1", "race_date": date(2026, 9, 19),
            "bet_type": "place", "selection_1": 1, "selection_2": None,
            "selection_3": None, "odds_min": 2.0, "odds_max": 2.4,
            "source_html": "one.html",
        }]
    )
    storage.save_race_odds(frame, "run-1")
    storage.save_race_odds(frame.assign(odds_min=3.0), "run-2")

    loaded = storage.load_race_odds(["r1"])
    assert len(loaded) == 1
    assert loaded.iloc[0].odds_min == 3.0
    assert loaded.iloc[0].collection_run_id == "run-2"


def test_bet_recommendation_uses_odds_after_prediction():
    predictions = pd.DataFrame(
        [
            {"race_id": "r1", "course_name": "中山", "race_number": 1,
             "race_name": "テスト", "horse_number": 1, "top3_probability": 0.8},
            {"race_id": "r1", "course_name": "中山", "race_number": 1,
             "race_name": "テスト", "horse_number": 2, "top3_probability": 0.4},
        ]
    )
    odds = pd.DataFrame(
        [
            {"race_id": "r1", "bet_type": "place", "selection_1": 1,
             "selection_2": None, "selection_3": None, "odds_min": 2.0},
            {"race_id": "r1", "bet_type": "place", "selection_1": 2,
             "selection_2": None, "selection_3": None, "odds_min": 3.0},
        ]
    )
    result = betting.calculate_bet_recommendations(predictions, odds)

    assert len(result) == 1
    assert result.iloc[0].selection == "1"
    assert result.iloc[0].recovery_rate_percent == pytest.approx(160.0)
    assert result.iloc[0].probability_method == "3着内確率（モデル出力）"


def test_bet_recommendation_filters_low_probability_and_prioritizes_place_wide():
    predictions = pd.DataFrame([
        {"race_id": "r1", "course_name": "中山", "race_number": 1,
         "race_name": "テスト", "horse_number": 1, "top3_probability": 0.6},
        {"race_id": "r1", "course_name": "中山", "race_number": 1,
         "race_name": "テスト", "horse_number": 2, "top3_probability": 0.399},
        {"race_id": "r1", "course_name": "中山", "race_number": 1,
         "race_name": "テスト", "horse_number": 3, "top3_probability": 0.001},
    ])
    odds = pd.DataFrame([
        {"race_id": "r1", "bet_type": "place", "selection_1": 1,
         "selection_2": None, "selection_3": None, "odds_min": 2.0},
        {"race_id": "r1", "bet_type": "win", "selection_1": 2,
         "selection_2": None, "selection_3": None, "odds_min": 10.0},
        {"race_id": "r1", "bet_type": "trifecta", "selection_1": 3,
         "selection_2": 2, "selection_3": 1, "odds_min": 10000.0},
    ])

    result = betting.calculate_bet_recommendations(
        predictions, odds, best_only=False
    )

    assert result.iloc[0].bet_type == "place"
    assert "trifecta" not in set(result["bet_type"])
