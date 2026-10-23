"""Network-free checks of fresh-download staging and failure recovery."""
from datetime import date
import importlib.util
import json
from pathlib import Path

import pytest
import requests


@pytest.fixture
def validation():
    path = Path(__file__).resolve().parents[1] / 'scripts/validate_historical_import.py'
    spec = importlib.util.spec_from_file_location('historical_import_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Collector:
    RESULT_FALLBACK_URL = 'https://example.test/race/{race_id}/'

    def __init__(self, fail=False):
        self.calls = []
        self.diagnostics = []
        self.fail = fail

    def race_ids_for_date(self, day, run, kind, force):
        assert force is True
        self.diagnostics.append({'cache': False})
        return ['202601010501', '202601010502']

    def _download(self, url, path, force):
        self.calls.append((url, force))
        assert force is True
        if self.fail:
            raise requests.ConnectionError('simulated unavailable server')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('new network response', encoding='utf-8')
        self.diagnostics.append({'cache': False})


def test_retry_uses_persisted_checkpoint_and_only_requests_failed_race(validation, tmp_path):
    database = tmp_path / 'copy.duckdb'
    database.write_bytes(b'unchanged database')
    collector = Collector()
    ids, report = validation.stage_fresh_day(
        collector, date(2026, 8, 8), tmp_path, database, simulate_failure=True)
    checkpoint = json.loads((tmp_path / 'download_checkpoint_20260808.json').read_text())
    assert checkpoint['status'] == 'incomplete'
    assert checkpoint['failed_races'] == [ids[0]]
    assert checkpoint['database_unchanged_before_completion']
    assert report['status'] == 'ok'
    assert report['retry_requested'] == [ids[0]]
    assert [url for url, force in collector.calls] == [
        Collector.RESULT_FALLBACK_URL.format(race_id=ids[1]),
        Collector.RESULT_FALLBACK_URL.format(race_id=ids[0]),
    ]
    assert report['all_downloads_network_only']
    assert database.read_bytes() == b'unchanged database'


def test_exhausted_retries_leave_database_unchanged_and_keep_failed_ids(validation, tmp_path, monkeypatch):
    monkeypatch.setattr(validation.time, 'sleep', lambda _: None)
    database = tmp_path / 'copy.duckdb'
    database.write_bytes(b'unchanged database')
    collector = Collector(fail=True)
    with pytest.raises(ValueError, match='Incomplete downloads'):
        validation.stage_fresh_day(collector, date(2026, 8, 8), tmp_path, database)
    report = json.loads((tmp_path / 'download_resumed_20260808.json').read_text())
    assert report['status'] == 'incomplete'
    assert len(report['failed_races']) == 2
    assert len(collector.calls) == 12  # Three attempts per race, then one resumed batch.
    assert database.read_bytes() == b'unchanged database'


def test_fresh_fetch_overwrites_run_local_html_instead_of_using_it(validation, tmp_path):
    database = tmp_path / 'copy.duckdb'
    database.write_bytes(b'unchanged database')
    stale = tmp_path / 'fetch/result/202601010501.html'
    stale.parent.mkdir(parents=True)
    stale.write_text('stale cache', encoding='utf-8')
    collector = Collector()
    _, report = validation.stage_fresh_day(collector, date(2026, 8, 8), tmp_path, database)
    assert stale.read_text(encoding='utf-8') == 'new network response'
    assert len(collector.calls) == 2
    assert report['all_downloads_network_only']


def test_transient_timeout_is_retried_before_checkpoint_completion(validation, tmp_path, monkeypatch):
    monkeypatch.setattr(validation.time, 'sleep', lambda _: None)
    database = tmp_path / 'copy.duckdb'
    database.write_bytes(b'unchanged database')
    collector = Collector()
    download = collector._download
    first_request = True

    def fail_once(url, path, force):
        nonlocal first_request
        if first_request:
            first_request = False
            raise requests.Timeout('simulated transient timeout')
        return download(url, path, force)

    monkeypatch.setattr(collector, '_download', fail_once)
    _, report = validation.stage_fresh_day(collector, date(2026, 8, 8), tmp_path, database)
    assert report['status'] == 'ok'
    assert report['retry_requested'] == []  # Recovered during the original download pass.
    assert report['http_attempts'][0]['status'] == 'error'
    assert report['http_attempts'][1]['attempt'] == 2
    assert report['http_attempts'][1]['status'] == 'ok'
    assert database.read_bytes() == b'unchanged database'
