from importlib import import_module as _import_module
from datetime import datetime, timezone

import pytest

build_status = _import_module('src.60_publication.job_status').build_status
fetch_runs = _import_module('src.60_publication.job_status').fetch_runs
render_status = _import_module('src.60_publication.job_status').render_status
result_label = _import_module('src.60_publication.job_status').result_label


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
    assert "定期実行" in page and "失敗" in page
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
