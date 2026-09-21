from datetime import date
import json

import pandas as pd
import pytest

from src import date_prediction_smoke as daily


@pytest.mark.parametrize("statuses,expected", [
    (["ok", "ok"], "ok"), (["incomplete", "ok"], "incomplete"),
    (["error", "ok"], "error"), ([], "incomplete"),
])
def test_date_continues_and_aggregates_only_success(tmp_path, monkeypatch, statuses, expected):
    ids = [f"2026060405{n:02d}" for n in range(1, len(statuses) + 1)]
    monkeypatch.setattr(daily.DateCollector, "race_ids_for_date", lambda *a, **kw: ids + ids[:1])
    called = []
    def predict(bundle, output, race_date, race_id):
        called.append(race_id)
        status = statuses[ids.index(race_id)]
        output.mkdir(parents=True)
        if status == "error":
            raise RuntimeError("simulated failure")
        if status == "ok":
            pd.DataFrame([{"race_id": race_id, "model_run_id": "m", "horse_id": "h"}]).to_csv(output / "predictions.csv", index=False)
            pd.DataFrame(columns=["race_id", "selection"]).to_csv(output / "bets.csv", index=False)
        return {"status": status, "prediction_status": status}
    monkeypatch.setattr(daily, "run_prediction", predict)
    monkeypatch.setattr(daily, "build_prediction_site", lambda *a, **kw: None)
    summary = tmp_path / "github.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    output = tmp_path / "results"
    report = daily.run_date(tmp_path, output, date(2026, 9, 19))
    assert report["status"] == expected
    assert called == ids
    assert len(report["races"]) == len(ids)
    assert json.loads((output / "date_report.json").read_text(encoding="utf-8"))["status"] == expected
    assert (output / "index.html").exists()
    assert summary.exists()
    if "ok" in statuses:
        result = pd.read_csv(output / "predictions.csv", dtype={"race_id": str})
        assert set(result.race_id) == {r for r, s in zip(ids, statuses) if s == "ok"}


def test_discovery_failure_is_reported(tmp_path, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("discovery failed")
    monkeypatch.setattr(daily.DateCollector, "race_ids_for_date", fail)
    result = daily.run_date(tmp_path, tmp_path / "results", date(2026, 9, 19))
    assert result["status"] == "error"
    assert "discovery failed" in result["error"]


def test_stage_details_include_missing_odds(tmp_path):
    (tmp_path / "fetch").mkdir()
    (tmp_path / "fetch/report.json").write_text(json.dumps({
        "card": {"status": "ok", "runners": 12},
        "jra_odds": {"status": "incomplete", "rows": 0},
        "database": {"status": "incomplete", "checks": {"win_odds_match_all_runners": False}},
    }), encoding="utf-8")
    row = daily.race_row("202606040501", {"status": "incomplete"}, tmp_path)
    assert row["jra_odds"] == "incomplete"
    assert row["prediction"] == "skipped"
    assert "単勝" in row["reason"]
