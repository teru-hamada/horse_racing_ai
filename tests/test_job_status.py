from importlib import import_module as _import_module
from datetime import datetime, timezone
from io import BytesIO
import json
from zipfile import ZipFile
from urllib.request import HTTPRedirectHandler

import pytest

build_status = _import_module('src.60_publication.job_status').build_status
fetch_runs = _import_module('src.60_publication.job_status').fetch_runs
render_status = _import_module('src.60_publication.job_status').render_status
result_label = _import_module('src.60_publication.job_status').result_label
module = _import_module('src.60_publication.job_status')


@pytest.mark.parametrize("status,conclusion,label", [
    ("completed", "success", "成功"), ("completed", "failure", "失敗"),
    ("completed", "timed_out", "失敗"), ("completed", "cancelled", "中止"),
    ("in_progress", None, "実行中"), ("queued", None, "待機中"),
])
def test_result(status, conclusion, label):
    assert result_label({"status": status, "conclusion": conclusion}) == label


def test_snapshot_has_jst_and_no_error_details(tmp_path):
    now = datetime(2026, 9, 20, 18, tzinfo=timezone.utc)
    runs = [{"run_started_at": "2026-09-20T18:00:00Z", "event": "schedule",
             "status": "completed", "conclusion": "failure", "run_attempt": 2,
             "error": "SECRET ERROR", "display_title": "<script>unsafe</script>"}]
    (tmp_path / "index.html").write_text("<main>INDEX</main>", encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    old = predictions / "2026-09-21.html"
    old.write_text("<main>OLD PREDICTION</main>", encoding="utf-8")
    for _ in range(2):
        build_status(tmp_path, runs, now)
    page = (tmp_path / "job-status.html").read_text(encoding="utf-8")
    assert "2026-09-21 03:00:00" in page
    assert "失敗" in page
    assert "実行方法" not in page and "試行回数" not in page
    assert "予想レース数" in page and "レース結果照合数" in page
    assert "SECRET" not in page and "<script>" not in page
    assert (tmp_path / "index.html").read_text(encoding="utf-8").count('href="job-status.html"') == 1
    assert 'href="../job-status.html"' in old.read_text(encoding="utf-8")
    assert "OLD PREDICTION" in old.read_text(encoding="utf-8")
    assert "実行記録はありません" in render_status([], now)


def test_api_failure_does_not_become_empty_history(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("API unavailable")
    monkeypatch.setattr("src.60_publication.job_status.urlopen", fail)
    with pytest.raises(OSError):
        fetch_runs("owner/repo", "main", "token")


def test_status_only_displays_seven_runs_and_preserves_unknown_counts():
    runs = [{"run_started_at": f"2026-09-{day:02d}T18:00:00Z", "status": "completed",
             "conclusion": "success", "predicted_races": 12, "compared_races": 10}
            for day in range(24, 14, -1)]
    runs[0]["predicted_races"] = None
    runs[0]["compared_races"] = 0
    page = render_status(runs, datetime.now(timezone.utc))
    assert page.count("<tr>") == 8  # header plus seven runs
    assert "<td>—</td><td>0</td>" in page
    assert "<td>12</td><td>10</td>" in page
    assert "2026-09-18 03:00:00" not in page
    assert "最新7件" in page


def test_counts_use_completed_predictions_and_comparisons(tmp_path):
    (tmp_path / "prediction").mkdir()
    (tmp_path / "comparison").mkdir()
    (tmp_path / "prediction/date_report.json").write_text(json.dumps({
        "race_ids": ["r1", "r2", "r3"],
        "races": [{"prediction": "ok", "status": "ok"},
                  {"prediction": "ok", "status": "incomplete"},
                  {"prediction": "error", "status": "error"}],
    }), encoding="utf-8")
    (tmp_path / "comparison/report.json").write_text(json.dumps({
        "summary": {"requested_races": 12, "compared_races": 10, "failed_races": 2},
    }), encoding="utf-8")
    assert module.collect_counts(tmp_path) == {"predicted_races": 2, "compared_races": 10}


@pytest.mark.parametrize("report,expected", [
    (None, {"predicted_races": None, "compared_races": None}),
    ({"status": "ok", "tasks": {"prediction": {"status": "no_races"}}},
     {"predicted_races": 0, "compared_races": 0}),
    ({"status": "error", "tasks": {"prediction": {"status": "error"},
                                    "comparison": {"status": "no_races"}}},
     {"predicted_races": None, "compared_races": 0}),
    ({"status": "running", "tasks": {}}, {"predicted_races": None, "compared_races": None}),
])
def test_unavailable_counts_are_not_assumed_zero(tmp_path, report, expected):
    if report is not None:
        (tmp_path / "daily_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert module.collect_counts(tmp_path) == expected


def test_artifact_counts_use_current_attempt_and_expose_only_counts(monkeypatch):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("race_counts.json", json.dumps({
            "predicted_races": 12, "compared_races": 10, "error": "SECRET",
        }))
    calls = []
    def fetch(request, **kwargs):
        calls.append(request)
        if request.full_url.endswith("/50/zip"):
            return BytesIO(buffer.getvalue())
        assert "name=daily-racing-counts-42-2" in request.full_url
        return BytesIO(json.dumps({"artifacts": [{
            "name": "daily-racing-counts-42-2", "id": 50, "expired": False,
        }]}).encode())
    monkeypatch.setattr(module, "urlopen", fetch)
    assert module.fetch_counts("owner/repo", {"id": 42, "run_attempt": 2}, "token") == {
        "predicted_races": 12, "compared_races": 10,
    }
    request = calls[-1]
    assert request.get_header("Authorization") == "Bearer token"
    redirected = HTTPRedirectHandler().redirect_request(request, None, 302, "Found", {}, "https://storage.example/file")
    assert redirected.get_header("Authorization") is None


@pytest.mark.parametrize("artifacts", [[], [
    {"name": "daily-racing-counts-42-1", "id": 50, "expired": False},
], [{"name": "daily-racing-counts-42-2", "id": 50, "expired": True}]])
def test_missing_expired_or_previous_attempt_counts_are_unknown(monkeypatch, artifacts):
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: BytesIO(json.dumps({"artifacts": artifacts}).encode()))
    assert module.fetch_counts("owner/repo", {"id": 42, "run_attempt": 2}, "token") == {}


def test_fetch_limits_counts_to_seven_eligible_runs_and_survives_missing_artifacts(monkeypatch):
    runs = [{"id": n, "event": "push" if n == 10 else "schedule"} for n in range(10, 0, -1)]
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: BytesIO(json.dumps({"workflow_runs": runs}).encode()))
    calls = []
    def counts(repo, run, token):
        calls.append(run["id"])
        if run["id"] == 9:
            raise OSError("unavailable")
        return {"predicted_races": 12, "compared_races": 0}
    monkeypatch.setattr(module, "fetch_counts", counts)
    result = fetch_runs("owner/repo", "main", "token")
    assert calls == [9, 8, 7, 6, 5, 4, 3]
    assert "predicted_races" not in result[0]
    assert result[1]["predicted_races"] == 12
