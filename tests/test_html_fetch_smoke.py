from datetime import date
import json

import pandas as pd
import pytest

from src import html_fetch_smoke as smoke


@pytest.mark.parametrize("numbers,frames,status", [
    ([1, 2], [1, 2], "ok"),
    ([None, None], [None, None], "incomplete"),
    ([1, 1], [1, 2], "incomplete"),
    ([1, 2.5], [1, 9], "incomplete"),
    ([], [], "incomplete"),
])
def test_card_requires_valid_unique_numbers(numbers, frames, status):
    report = smoke.inspect_card(pd.DataFrame({"horse_number": numbers, "frame_number": frames}))
    assert report["status"] == status


@pytest.mark.parametrize("card_error,odds_present,status", [
    (False, True, "ok"), (False, False, "incomplete"), (True, True, "error"),
])
def test_probe_preserves_diagnostics_and_checks_jra_after_card_failure(
    tmp_path, monkeypatch, card_error, odds_present, status,
):
    def download(self, url, cache_path, force=False):
        assert force
        if card_error:
            raise RuntimeError("HTTP 403")
        cache_path.write_text("<html>card</html>", encoding="utf-8")
        return "<html>card</html>"

    def collect(self, race_date, race_ids, output_directory, force=False):
        assert race_ids == ["202609040701"] and force
        return int(odds_present)

    monkeypatch.setattr(smoke.NetkeibaHtmlCollector, "_download", download)
    monkeypatch.setattr(smoke.NetkeibaHtmlCollector, "_enrich_upcoming_html_with_odds", lambda self, html, *a, **kw: html)
    monkeypatch.setattr(smoke.NetkeibaHtmlCollector, "_parse_page", lambda *a, **kw: pd.DataFrame({
        "horse_number": [1, 2], "frame_number": [1, 2], "odds": [2.5, None],
    }))
    monkeypatch.setattr(smoke.DiagnosticOddsCollector, "collect_date", collect)
    monkeypatch.setattr(smoke, "parse_odds_directory", lambda *a: pd.DataFrame(
        [{"bet_type": "win", "odds_min": 2.5}] if odds_present else [],
        columns=["bet_type", "odds_min"],
    ))
    summary = tmp_path / "github_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "probe"
    report = smoke.run_probe(date(2026, 9, 21), "202609040701", output)
    assert report["status"] == status
    assert report["jra_odds"]["rows"] == int(odds_present)
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert summary.read_text(encoding="utf-8") == (output / "summary.md").read_text(encoding="utf-8")
    assert (output / "fetch.log").exists()


@pytest.mark.parametrize("race_id", ["$(echo injected)", "202509040701", "202699040701"])
def test_invalid_inputs_rejected_before_network(monkeypatch, race_id):
    monkeypatch.setattr(smoke, "run_probe", lambda *a: pytest.fail("must not fetch"))
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--race-date", "2026-09-21", "--race-id", race_id])
    assert exc.value.code == 2
