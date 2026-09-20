from datetime import date
from dataclasses import replace

import pandas as pd
import pytest

from src import database_smoke as smoke


@pytest.fixture
def frames():
    common = {"race_id": "202609040701", "race_date": date(2026, 9, 21)}
    card = pd.DataFrame([
        {**common, "horse_id": str(100 + n), "horse_name": f"Horse {n}",
         "horse_number": n, "frame_number": n} for n in (1, 2)
    ])
    odds = pd.DataFrame([
        {**common, "bet_type": kind, "selection_1": n, "selection_2": None,
         "selection_3": None, "odds_min": 2.5, "odds_max": 3.0}
        for n in (1, 2) for kind in ("win", "place")
    ])
    return card, odds


def verify(frames, output):
    return smoke.verify_database(*frames, date(2026, 9, 21), "202609040701", output)


def test_real_storage_round_trip_is_isolated_and_idempotent(tmp_path, monkeypatch, frames):
    production = tmp_path / "production.duckdb"
    production.write_bytes(b"must not open or modify")
    monkeypatch.setattr(smoke.storage, "PATHS", replace(smoke.storage.PATHS, database=production))
    report = verify(frames, tmp_path)
    assert report["status"] == "ok"
    assert all(report["checks"].values())
    assert report["matched_win_runners"] == 2
    assert report["registered_odds_by_bet_type"] == {"place": 2, "win": 2}
    assert production.read_bytes() == b"must not open or modify"
    assert (tmp_path / "smoke.duckdb").exists()
    assert (tmp_path / "db_win_join.csv").exists()
    with pytest.raises(ValueError):
        verify(frames, tmp_path)


@pytest.mark.parametrize("problem,failed_check", [
    ("missing_numbers", "registration_1_horse_and_frame_numbers"),
    ("duplicate_numbers", "registration_1_horse_and_frame_numbers"),
    ("missing_win", "win_odds_match_all_runners"),
    ("wrong_horse", "win_odds_match_all_runners"),
    ("wrong_race", "registration_1_odds_counts"),
    ("wrong_date", "registration_1_race_and_date"),
    ("duplicate_odds", "unique_odds_selections"),
    ("empty_odds", "registration_1_odds_counts"),
])
def test_detects_unusable_registered_data(tmp_path, frames, problem, failed_check):
    card, odds = frames
    if problem == "missing_numbers":
        card["horse_number"] = None
    elif problem == "duplicate_numbers":
        card["horse_number"] = 1
    elif problem == "missing_win":
        odds = odds[~(odds.bet_type.eq("win") & odds.selection_1.eq(2))]
    elif problem == "wrong_horse":
        odds.loc[odds.selection_1.eq(2), "selection_1"] = 3
    elif problem == "wrong_race":
        odds["race_id"] = "202609040702"
    elif problem == "wrong_date":
        odds["race_date"] = date(2026, 9, 20)
    elif problem == "duplicate_odds":
        odds = pd.concat([odds, odds.iloc[:1]], ignore_index=True)
    elif problem == "empty_odds":
        odds = odds.iloc[:0]
    report = verify((card, odds), tmp_path)
    assert report["status"] == "incomplete"
    assert failed_check in report["failed_checks"]


def test_detects_duplicate_append_on_reregistration(tmp_path, monkeypatch, frames):
    original = smoke.storage.save_race_odds
    def append_after_save(frame, run_id, **kwargs):
        original(frame, run_id, **kwargs)
        if run_id.endswith("2"):
            with smoke.storage.connect(kwargs["database"]) as con:
                con.execute("INSERT INTO race_odds SELECT * FROM race_odds")
    monkeypatch.setattr(smoke.storage, "save_race_odds", append_after_save)
    report = verify(frames, tmp_path)
    assert report["status"] == "incomplete"
    assert "reregistration_unchanged" in report["failed_checks"]


@pytest.mark.parametrize("database_error", [False, True])
def test_cli_database_mode_reports_success_and_errors(tmp_path, monkeypatch, frames, database_error):
    import json
    from src import html_fetch_smoke as html

    card, odds = frames
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_download", lambda *a, **kw: "<html></html>")
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_enrich_upcoming_html_with_odds", lambda self, text, *a, **kw: text)
    monkeypatch.setattr(html.NetkeibaHtmlCollector, "_parse_page", lambda *a, **kw: card)
    monkeypatch.setattr(html.DiagnosticOddsCollector, "collect_date", lambda *a, **kw: 1)
    monkeypatch.setattr(html, "parse_odds_directory", lambda *a: odds)
    if database_error:
        def fail(*a, **kw):
            raise RuntimeError("simulated database failure")
        monkeypatch.setattr(smoke.storage, "save_race_records", fail)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "output"
    code = html.main(["--race-date", "2026-09-21", "--race-id", "202609040701",
                      "--output", str(output), "--verify-db"])
    assert code == (1 if database_error else 0)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["database"]["status"] == ("error" if database_error else "ok")
    assert "database" in summary.read_text(encoding="utf-8")
