import json

import pandas as pd
import pytest

from src.publish_predictions import prepare_publication


@pytest.fixture
def results(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    row = {"race_id": "202606040510", "status": "ok", "runners": 1,
           "prediction_rows": 1, "bet_rows": 0, "model_id": "m"}
    row.update({stage: "ok" for stage in ("card", "jra_odds", "database", "prediction", "odds_match", "betting")})
    report = {"race_date": "2026-09-19", "status": "ok", "race_ids": [row["race_id"]], "races": [row]}
    (output / "date_report.json").write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame([{
        "race_id": row["race_id"], "race_date": report["race_date"], "horse_id": "h",
        "horse_number": 1, "model_run_id": "m", "top3_probability": 0.5,
        "course_name": "中山", "race_number": 10, "race_name": "Test", "horse_name": "Horse",
        "jockey_name": "Jockey", "prediction_rank": 1, "odds": 3.0, "expected_value_index": 1.5,
    }]).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(columns=["race_id", "bet_type", "selection"]).to_csv(output / "bets.csv", index=False)
    return output


def test_preserves_other_dates_and_identical_results_do_not_change_files(tmp_path, results):
    site = tmp_path / "docs"
    (site / "predictions").mkdir(parents=True)
    previous = site / "predictions/2026-09-18.html"
    previous.write_bytes(b"previous date")
    assert "predictions/2026-09-19.html" in prepare_publication(results, site)
    assert previous.read_bytes() == b"previous date"
    assert "2026-09-18.html" in (site / "index.html").read_text(encoding="utf-8")
    page = site / "predictions/2026-09-19.html"
    import re
    page.write_text(re.sub(r'<p class="meta">[^<]*</p>', '<p class="meta">生成日時: OLD TIME</p>', page.read_text(encoding="utf-8")), encoding="utf-8")
    snapshot = {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*") if p.is_file()}
    assert prepare_publication(results, site) == []
    assert snapshot == {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["partial", "stage", "missing_race", "count", "nan_probability"])
def test_rejects_incomplete_or_inconsistent_results_without_touching_site(tmp_path, results, problem):
    path = results / "date_report.json"
    report = json.loads(path.read_text())
    if problem == "partial": report["status"] = "incomplete"
    if problem == "stage": report["races"][0]["database"] = "incomplete"
    if problem == "missing_race": report["race_ids"].append("202606040511")
    if problem == "count": report["races"][0]["runners"] = 2
    if problem == "nan_probability":
        data = pd.read_csv(results / "predictions.csv")
        data["top3_probability"] = float("nan")
        data.to_csv(results / "predictions.csv", index=False)
    path.write_text(json.dumps(report), encoding="utf-8")
    site = tmp_path / "docs"
    site.mkdir()
    (site / "index.html").write_bytes(b"keep")
    with pytest.raises(ValueError): prepare_publication(results, site)
    assert list(site.iterdir()) == [site / "index.html"]
    assert (site / "index.html").read_bytes() == b"keep"
