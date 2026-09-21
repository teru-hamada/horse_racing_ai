from importlib import import_module as _import_module
from calendar import monthrange
from datetime import date, datetime, timezone
import json

import pandas as pd
import pytest

daily = _import_module('src.10_workflows.daily_racing')
parse_meeting_day = _import_module('src.20_scrapers_html_collection.race_calendar').parse_meeting_day
build_prediction_site = _import_module('src.60_publication.static_site').build_prediction_site


def calendar_html(year=2026, month=9):
    cells = []
    for day in range(1, monthrange(year, month)[1] + 1):
        body = f'<span class="Day">{day}</span>'
        if day == 19:
            body = f'<a href="race_list.html?kaisai_date={year}{month:02d}{day:02d}">{body}<span class="JyoName">中山</span></a>'
        cells.append(f'<td class="RaceCellBox">{body}</td>')
    return f'<select id="cal_select_year"><option value="{year}" selected>{year}</option></select><table class="Calendar_Table"><tr>'+''.join(cells)+'</tr></table>'


def test_calendar_requires_complete_matching_month():
    html = calendar_html()
    assert parse_meeting_day(html, date(2026, 9, 19))
    assert not parse_meeting_day(html, date(2026, 9, 20))
    for invalid in ("<html>maintenance</html>", html.replace('>30</span>', '>29</span>'),
                    html.replace('20260919', '20260819'), html.replace('class="JyoName"', 'class="unknown"')):
        with pytest.raises(ValueError): parse_meeting_day(invalid, date(2026, 9, 20))


def test_japan_day_uses_local_date_and_year_boundary():
    assert daily.japan_today(datetime(2026, 12, 31, 18, tzinfo=timezone.utc)) == date(2027, 1, 1)
    assert daily.japan_today(datetime(2026, 9, 20, 18, tzinfo=timezone.utc)) == date(2026, 9, 21)


@pytest.mark.parametrize("today,previous,expected,publication", [
    (False, False, "ok", False),
    (False, True, "ok", True),
    (True, False, "ok", True),
    ("error", True, "error", False),
    (True, "error", "error", False),
    ("incomplete", True, "incomplete", False),
])
def test_daily_calendar_errors_never_become_nonmeeting(tmp_path, monkeypatch, today, previous, expected, publication):
    called = []
    def meeting(day, *args):
        status = today if day.day == 21 else previous
        if status == "error": raise ValueError("calendar unavailable")
        return bool(status)
    monkeypatch.setattr(daily, "check_meeting_day", meeting)
    monkeypatch.setattr(daily, "run_date", lambda *a: {"status": "incomplete" if today == "incomplete" else "ok"})
    def compare(*args):
        called.append(args[0])
        return {"status": "ok"}
    monkeypatch.setattr(daily, "compare_previous", compare)
    output = tmp_path / "run"
    report = daily.run_daily(date(2026, 9, 21), tmp_path, tmp_path, output, True)
    assert report["status"] == expected
    assert report["publication_required"] is publication
    assert bool(called) == (previous is True)
    if not publication:
        if expected == "ok":
            assert daily.prepare_daily_publication(output, tmp_path / "missing_site") is False
        else:
            with pytest.raises(ValueError): daily.prepare_daily_publication(output, tmp_path / "missing_site")


@pytest.fixture
def prior_site(tmp_path, monkeypatch):
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    row = {"race_id": "202606040510", "race_date": "2026-09-19", "horse_id": "h1", "horse_number": 1,
           "model_run_id": "m", "top3_probability": 0.5, "course_name": "中山", "race_number": 10,
           "race_name": "Test", "horse_name": "Horse", "jockey_name": "Jockey", "prediction_rank": 1,
           "odds": 3.0, "expected_value_index": 1.5}
    prediction = pd.DataFrame([row])
    prediction.to_csv(site / "predictions/2026-09-19.csv", index=False)
    pd.DataFrame([{"race_id": row["race_id"], "bet_type": "win", "selection": "1",
                   "bet_type_label": "単勝", "estimated_probability": 0.5, "odds_used": 3.0,
                   "recovery_rate_percent": 150.0, "recommendation_score": 150.0,
                   "expected_profit_per_100": 50.0,
                   "is_primary_bet_type": False, "bet_type_reliability": 1.0}]).to_csv(site / "predictions/2026-09-19_bets.csv", index=False)
    build_prediction_site(prediction, "2026-09-19", "m", site)
    monkeypatch.setattr(daily.DateCollector, "race_ids_for_date", lambda *a, **kw: [row["race_id"]])
    def fetch(*args, **kwargs):
        actual = pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": 1}])
        actual.attrs["official_payouts"] = {"win:1": 300}
        return actual
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", fetch)
    return site


def test_previous_results_update_without_repredicting_and_preserve_original(tmp_path, prior_site):
    import logging
    output = tmp_path / "run"
    output.mkdir()
    original = (prior_site / "predictions/2026-09-19.csv").read_bytes()
    result = daily.compare_previous(date(2026, 9, 19), prior_site, output / "comparison", logging.getLogger())
    assert result["status"] == "ok"
    (output / "daily_report.json").write_text(json.dumps({"status": "ok", "publication_required": True,
        "tasks": {"prediction": {"status": "no_races"}, "comparison": {"status": "ok", "race_date": "2026-09-19"}}}), encoding="utf-8")
    assert daily.prepare_daily_publication(output, prior_site)
    assert (prior_site / "predictions/2026-09-19.csv").read_bytes() == original
    assert (prior_site / "predictions/2026-09-19_comparison.csv").exists()
    payouts = pd.read_csv(prior_site / "predictions/2026-09-19_bets_results.csv")
    assert payouts.payout_per_100.tolist() == [300]
    snapshot = {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
    daily.prepare_daily_publication(output, prior_site)
    assert snapshot == {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}


def test_result_missing_finish_stops_publication(tmp_path, monkeypatch, prior_site):
    import logging
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", lambda *a, **kw:
                        pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": None}]))
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / "comparison", logging.getLogger())
    assert result["status"] == "incomplete"
    assert result["summary"]["failed_races"] == 1


def test_missing_prior_prediction_is_error(tmp_path):
    import logging
    with pytest.raises(ValueError, match="CSV"):
        daily.compare_previous(date(2026, 9, 19), tmp_path, tmp_path / "comparison", logging.getLogger())


def test_missing_payout_stops_comparison(tmp_path, monkeypatch, prior_site):
    import logging
    monkeypatch.setattr(daily.JraResultFetcher, "fetch_result_for_comparison", lambda *a, **kw:
                        pd.DataFrame([{"horse_id": None, "horse_number": 1, "finish_position": 1}]))
    result = daily.compare_previous(date(2026, 9, 19), prior_site, tmp_path / "comparison", logging.getLogger())
    assert result["status"] == "incomplete"


def test_changed_previous_inputs_prevent_both_publications(tmp_path, monkeypatch, prior_site):
    import logging
    output = tmp_path / "run"
    output.mkdir()
    daily.compare_previous(date(2026, 9, 19), prior_site, output / "comparison", logging.getLogger())
    (output / "daily_report.json").write_text(json.dumps({"status": "ok", "publication_required": True,
        "tasks": {"prediction": {"status": "ok"}, "comparison": {"status": "ok", "race_date": "2026-09-19"}}}), encoding="utf-8")
    def prepare_today(source, staged):
        (staged / "predictions/today.html").write_bytes(b"new prediction")
    monkeypatch.setattr(daily, "prepare_publication", prepare_today)
    with (prior_site / "predictions/2026-09-19.csv").open("ab") as stream:
        stream.write(b"\n")
    before = {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match="変更"):
        daily.prepare_daily_publication(output, prior_site)
    assert before == {p.name:p.read_bytes() for p in prior_site.rglob('*') if p.is_file()}
