"""Complete-day ingestion, persistent restart, model preservation and safe publishing."""
from calendar import monthrange
from datetime import date
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest
import requests

from scripts import private_runtime, private_snapshot, runtime_transfer
from scripts import publish_results_snapshot as publisher
from tests.test_runtime_transfer import make_source

update = import_module('src.10_workflows.result_update')


def calendar():
    cells = []
    for day in range(1, monthrange(2026, 8)[1] + 1):
        body = f'<span class="Day">{day}</span>'
        if day in (8, 9):
            body = f'<a href="race_list.html?kaisai_date=202608{day:02d}">{body}<span class="JyoName">札幌</span></a>'
        cells.append(f'<td class="RaceCellBox">{body}</td>')
    return '<select id="cal_select_year"><option selected value="2026">2026</option></select><table class="Calendar_Table">' + ''.join(cells) + '</table>'


class Collector:
    RESULT_FALLBACK_URL = 'https://example.test/race/{race_id}/'
    calls = []

    def __init__(self, *args):
        self.session = SimpleNamespace(close=lambda: None)

    def _download(self, url, path, force):
        assert force is True
        self.calls.append(url)
        return calendar() if 'calendar' in url else '<html>fresh result</html>'

    def race_ids_for_date(self, day, *args, **kwargs):
        return [f'202601010{day.day - 3}01', f'202601010{day.day - 3}02']


class Parser:
    fail = None

    def __init__(self, *args):
        self.session = SimpleNamespace(close=lambda: None)

    def _parse_page(self, html, race_id, day, kind):
        if race_id == self.fail:
            raise ValueError('injected malformed result')
        return pd.DataFrame([{'race_id': race_id, 'race_date': day,
                              'horse_id': f'20230000{number:02d}', 'horse_number': number,
                              'finish_position': number, 'time_seconds': 70.0 + number}
                             for number in range(1, 4)])


@pytest.fixture
def source(tmp_path):
    root = make_source(tmp_path / 'source')
    database = root / 'data/racing.duckdb'
    database.unlink()
    with duckdb.connect(str(database)) as connection:
        connection.execute(update.storage.SCHEMA_SQL)
        connection.execute("INSERT INTO race_records(race_id,race_date,horse_id,dataset_type,finish_position,time_seconds) VALUES ('202601010401',DATE '2026-08-02','2023000001','historical',1,70)")
    Collector.calls = []
    Parser.fail = None
    return root


def run(source, tmp_path, suffix, start=date(2026, 8, 8), end=date(2026, 8, 8)):
    return update.update(source, tmp_path / ('out' + suffix), tmp_path / ('work' + suffix),
        start=start, end=end, today=date(2026, 10, 23), collector_factory=Collector, parser_factory=Parser)


def test_complete_day_and_second_execution_are_idempotent(source, tmp_path):
    before = runtime_transfer.hashes(source)
    report = run(source, tmp_path, '1')
    assert report['status'] == 'ok' and report['updated_rows'] == 6
    assert report['model_and_features_unchanged'] and not report['model_updated']
    assert report['cursor'] == '2026-08-02'  # Manual Aug 8 must not hide Aug 3-7.
    assert runtime_transfer.hashes(source) == before
    private_snapshot.restore(tmp_path / 'out1', tmp_path / 'restored')
    after = runtime_transfer.hashes(tmp_path / 'restored')
    assert all(after[name] == before[name] for name in update.PROTECTED)
    assert after['data/racing.duckdb'] != before['data/racing.duckdb']
    downloads = len(Collector.calls)
    report2 = run(tmp_path / 'out1', tmp_path, '2')
    assert report2['status'] == 'ok' and not report2['changed']
    assert report2['days'][0]['status'] == 'already_completed'
    assert not (tmp_path / 'out2').exists()
    assert len(Collector.calls) - downloads == 1  # Calendar only; no result re-download.


def test_incomplete_day_is_not_registered_but_other_complete_days_survive(source, tmp_path):
    Parser.fail = '202601010502'
    report = run(source, tmp_path, '1', end=date(2026, 8, 9))
    assert report['status'] == 'incomplete'
    assert report['updated_days'] == 1 and report['updated_rows'] == 6
    private_snapshot.restore(tmp_path / 'out1', tmp_path / 'restored')
    database = tmp_path / 'restored/data/racing.duckdb'
    assert update.day_counts(database, date(2026, 8, 8)) == {}
    assert len(update.day_counts(database, date(2026, 8, 9))) == 2
    state = json.loads((tmp_path / 'out1' / update.STATE_FILE).read_text())
    assert '2026-08-08' not in state['days']
    assert state['days']['2026-08-09']['status'] == 'completed'
    Parser.fail = None
    repaired = run(tmp_path / 'out1', tmp_path, '2')
    assert repaired['status'] == 'ok' and repaired['updated_rows'] == 6


def test_no_meeting_days_advance_cursor_without_rewriting_existing_parts(source, tmp_path):
    first = run(source, tmp_path, '1', date(2026, 8, 3), date(2026, 8, 3))
    assert first['cursor'] == '2026-08-03' and not first['database_changed']
    second = run(tmp_path / 'out1', tmp_path, '2', date(2026, 8, 4), date(2026, 8, 4))
    assert second['cursor'] == '2026-08-04'
    for part in private_snapshot.inspect_manifest(tmp_path / 'out1')['parts']:
        assert private_runtime.digest(tmp_path / 'out1' / part['path']) == private_runtime.digest(tmp_path / 'out2' / part['path'])


def test_all_failed_days_produce_no_candidate(source, tmp_path, monkeypatch):
    monkeypatch.setattr(Collector, '_download', lambda *a, **k: '<html>maintenance</html>')
    report = run(source, tmp_path, '1')
    assert report['status'] == 'incomplete' and not report['changed']
    assert not (tmp_path / 'out1').exists()


def test_transient_download_timeout_recovers_without_losing_day(source, tmp_path, monkeypatch):
    download = Collector._download
    attempts = []

    def timeout_once(self, url, path, force):
        attempts.append(url)
        if len(attempts) == 1:
            raise requests.Timeout('temporary outage')
        return download(self, url, path, force)

    monkeypatch.setattr(Collector, '_download', timeout_once)
    monkeypatch.setattr(update.time, 'sleep', lambda _: None)
    before = runtime_transfer.hashes(source)
    report = run(source, tmp_path, 'retry')
    assert report['status'] == 'ok' and report['updated_rows'] == 6
    assert attempts[0] == attempts[1]
    assert runtime_transfer.hashes(source) == before


def test_exhausted_downloads_leave_day_retryable_and_source_unchanged(source, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError('offline')

    monkeypatch.setattr(Collector, '_download', unavailable)
    monkeypatch.setattr(update.time, 'sleep', lambda _: None)
    before = runtime_transfer.hashes(source)
    report = run(source, tmp_path, 'failed')
    assert report['status'] == 'incomplete' and not report['changed']
    assert not (tmp_path / 'outfailed').exists()
    assert runtime_transfer.hashes(source) == before
    monkeypatch.setattr(Collector, '_download', lambda self, url, path, force: calendar() if 'calendar' in url else '<html>fresh result</html>')
    recovered = run(source, tmp_path, 'recovered')
    assert recovered['status'] == 'ok' and recovered['updated_rows'] == 6


@pytest.mark.parametrize('start,end,max_days', [
    (date(2026, 10, 23), date(2026, 10, 23), 7),
    (date(2026, 8, 9), date(2026, 8, 8), 7),
    (date(2026, 8, 1), date(2026, 8, 8), 7),
    (None, None, 0),
])
def test_invalid_period_is_rejected(start, end, max_days):
    with pytest.raises(ValueError):
        update.plan_dates({'cursor': '2026-08-02'}, start, end, date(2026, 10, 23), max_days)


def test_saved_state_preserves_scan_gaps():
    state = {'cursor': '2026-08-02', 'days': {'2026-08-08': {'status': 'completed'}}}
    assert update.plan_dates(state, None, None, date(2026, 10, 23), 7) == [date(2026, 8, day) for day in range(3, 10)]


def test_stage_only_replaces_snapshot_files_and_keeps_workflow(source, tmp_path, monkeypatch):
    run(source, tmp_path, '1')
    workflow = source / '.github/workflows/update-results.yml'
    workflow.parent.mkdir(parents=True)
    workflow.write_text('private workflow')
    calls = []

    def git(repo, *args):
        calls.append(args)
        if args[:2] == ('rev-parse', 'HEAD'):
            return 'a' * 40
        if args == ('ls-files',):
            return '\n'.join(private_runtime.FILES)
        return ''

    monkeypatch.setattr(publisher, 'git', git)
    publisher.stage_candidate(source, tmp_path / 'out1', 'a' * 40, tmp_path / 'work1/report.json')
    assert workflow.read_text() == 'private workflow'
    assert (source / 'snapshot.json').is_file()
    assert not (source / 'data/racing.duckdb').exists()
    assert all('.github/workflows/update-results.yml' not in args for args in calls)


def test_model_or_feature_change_is_rejected_before_repository_mutation(source, tmp_path, monkeypatch):
    run(source, tmp_path, '1')
    private_snapshot.restore(tmp_path / 'out1', tmp_path / 'bad-source')
    (tmp_path / 'bad-source/data/features.duckdb').write_bytes(b'forbidden feature update')
    private_snapshot.prepare(tmp_path / 'bad-source', tmp_path / 'bad-candidate')
    before = runtime_transfer.hashes(source)
    monkeypatch.setattr(publisher, 'git', lambda *a: 'a' * 40)
    with pytest.raises(ValueError, match='model/bundle/features'):
        publisher.stage_candidate(source, tmp_path / 'bad-candidate', 'a' * 40, tmp_path / 'work1/report.json')
    assert runtime_transfer.hashes(source) == before
