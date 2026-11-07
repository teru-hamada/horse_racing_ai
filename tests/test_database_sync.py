"""GUI batch merging must retain remote results, models and durable retry data."""
from datetime import date
from importlib import import_module
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts import private_runtime, private_snapshot, runtime_transfer
from tests.test_result_update import Collector, Parser
from tests.test_runtime_transfer import make_source

sync = import_module('src.10_workflows.database_sync')
jobs = import_module('src.30_scrapers_database_creation.database_creation_jobs')
get_paths = import_module('src.00_common.config').get_paths


def database(root):
    make_source(root)
    path = root / 'data/racing.duckdb'
    path.unlink()
    with duckdb.connect(str(path)) as connection:
        connection.execute(sync.result_update.storage.SCHEMA_SQL)
        connection.execute("INSERT INTO race_records(race_id,race_date,horse_id,dataset_type,finish_position,time_seconds) VALUES ('202601010401',DATE '2026-08-02','2023000001','historical',1,70)")
    return get_paths(root)


def records(day=9):
    parser = Parser()
    return pd.concat([parser._parse_page('', race_id, date(2026, 8, day), 'historical')
                      for race_id in Collector().race_ids_for_date(date(2026, 8, day))], ignore_index=True)


def enqueue(paths, frame, batch='database_fixture'):
    directory = paths.historical_html / '2026/result'
    directory.mkdir(parents=True, exist_ok=True)
    for race_id in frame.race_id.unique():
        (directory / f'{race_id}.html').write_text('<html>cached GUI result</html>')
    sync.enqueue(frame, batch, 'example/private', paths=paths)
    return sync.read_batch(batch, paths)


@pytest.fixture(autouse=True)
def reset_parser():
    Parser.fail = None
    Collector.calls = []


def test_batch_survives_restart_and_partial_payload_completes_day(tmp_path):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    frame = records()
    directory, metadata = enqueue(local, frame.iloc[:3])
    assert sync.pending_batches(paths=local)[0]['id'] == metadata['id']
    restored, candidate, report = sync.build_candidate(remote.root, directory, metadata, tmp_path / 'merge',
        collector_factory=Collector, parser_factory=Parser)
    assert report['days'][0]['gui_races'] == 1
    assert report['days'][0]['fetched_races'] == 1
    assert sum(sync.result_update.day_counts(restored / 'data/racing.duckdb', date(2026, 8, 9)).values()) == 6
    assert candidate is not None
    for name in sync.result_update.PROTECTED:
        assert private_runtime.digest(restored / name) == private_runtime.digest(remote.root / name)


def test_matching_remote_results_do_not_rewrite_db(tmp_path):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    frame = records()
    sync.result_update.save_day(remote.database, frame, date(2026, 8, 9))
    counts = sync.result_update.day_counts(remote.database, date(2026, 8, 9))
    state = {'format': 1, 'anchor': '2026-08-02', 'cursor': '2026-08-02',
             'days': {'2026-08-09': {'status': 'completed', 'race_rows': counts}}}
    (remote.root / 'result_update_state.json').write_text(json.dumps(state))
    directory, metadata = enqueue(local, frame)
    _, candidate, report = sync.build_candidate(remote.root, directory, metadata, tmp_path / 'merge',
        collector_factory=Collector, parser_factory=Parser)
    assert candidate is None and report['updated_rows'] == 0
    assert report['days'][0]['gui_races'] == report['days'][0]['fetched_races'] == 0


def test_conflicting_old_gui_values_are_resolved_using_fresh_results(tmp_path):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    current = records()
    sync.result_update.save_day(remote.database, current, date(2026, 8, 9))
    stale = current.copy()
    stale['time_seconds'] += 100
    directory, metadata = enqueue(local, stale)
    restored, _, report = sync.build_candidate(remote.root, directory, metadata, tmp_path / 'merge',
        collector_factory=Collector, parser_factory=Parser)
    assert report['updated_rows'] == 0 and report['days'][0]['fetched_races'] == 2
    assert private_runtime.digest(restored / 'data/racing.duckdb') == private_runtime.digest(remote.database)


def test_incomplete_day_never_produces_candidate(tmp_path):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    directory, metadata = enqueue(local, records().iloc[:3])
    Parser.fail = '202601010602'
    before = runtime_transfer.hashes(remote.root)
    with pytest.raises(ValueError, match='malformed'):
        sync.build_candidate(remote.root, directory, metadata, tmp_path / 'merge',
            collector_factory=Collector, parser_factory=Parser)
    assert not (tmp_path / 'merge/candidate').exists()
    assert runtime_transfer.hashes(remote.root) == before
    assert sync.pending_batches(paths=local)


def test_local_merge_pulls_remote_history_without_replacing_models_or_later_import(tmp_path):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    update = sync.result_update
    update.save_day(remote.database, records(8), date(2026, 8, 8))
    update.save_day(remote.database, records(9), date(2026, 8, 9))
    update.storage.save_race_records(records(9), 'database_fixture', 'historical', database=local.database)
    with duckdb.connect(str(local.database)) as connection:
        connection.execute("UPDATE race_records SET time_seconds=999,collection_run_id='later-job' WHERE race_id='202601010601' AND horse_number=1")
    protected = {name: private_runtime.digest(local.root / name) for name in update.PROTECTED}
    sync.merge_local(remote.database, 'database_fixture', paths=local)
    assert len(update.day_counts(local.database, date(2026, 8, 8))) == 2
    with duckdb.connect(str(local.database), read_only=True) as connection:
        assert connection.execute("SELECT time_seconds FROM race_records WHERE collection_run_id='later-job'").fetchone()[0] == 999
    assert all(private_runtime.digest(local.root / name) == digest for name, digest in protected.items())


def test_network_failure_keeps_retry_batch_and_local_data(tmp_path, monkeypatch):
    local = database(tmp_path / 'local')
    directory, metadata = enqueue(local, records())
    before = runtime_transfer.hashes(local.root)
    def unavailable(*args, **kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr(sync.private_sync, 'repository_info', unavailable)
    with pytest.raises(RuntimeError, match='offline'):
        sync.synchronize(metadata['id'], paths=local)
    assert sync.pending_batches(paths=local)[0]['status'] == 'failed'
    assert (directory / 'payload.parquet').exists()
    assert runtime_transfer.hashes(local.root) == before


def test_unsaved_prepared_batch_cannot_be_published_even_after_failed_retry(tmp_path, monkeypatch):
    local = database(tmp_path / 'local')
    sync.enqueue(records(), 'prepared', 'example/private', paths=local, prepared=True)
    monkeypatch.setattr(sync.private_sync, 'repository_info', lambda *a, **k: pytest.fail('must not access remote'))
    for _ in range(2):
        with pytest.raises(ValueError, match='未完了'):
            sync.synchronize('prepared', paths=local)
        assert sync.pending_batches(paths=local) == []


def test_prepared_batch_is_recoverable_after_local_commit_before_status_write(tmp_path):
    local = database(tmp_path / 'local')
    sync.enqueue(records(), 'prepared', 'example/private', paths=local, prepared=True)
    assert sync.pending_batches(paths=local) == []
    sync.result_update.storage.save_race_records(records(), 'prepared', 'historical', database=local.database)
    assert sync.pending_batches(paths=local)[0]['id'] == 'prepared'
    sync.mark_local_saved('prepared', paths=local)
    assert sync.read_batch('prepared', local)[1]['local_saved']


def test_sync_retries_concurrent_remote_update_before_publish(tmp_path, monkeypatch):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    _, metadata = enqueue(local, records())
    commits = iter(['a' * 40, 'b' * 40])
    monkeypatch.setattr(sync.private_sync, 'repository_info', lambda *a, **k: {'branch': 'main', 'commit': next(commits)})
    calls = []
    def git(args, **kwargs):
        calls.append(args)
        if args[0] == 'rev-parse':
            return ('b' if sum(a[0] == 'fetch' for a in calls) > 1 else 'a') * 40
        if args[0] == 'ls-remote':
            return 'b' * 40 + '\trefs/heads/main'
        return ''
    monkeypatch.setattr(sync.private_sync, '_git', git)
    monkeypatch.setattr(sync, 'build_candidate', lambda *a, **k: (remote.root, remote.root, {'status': 'ok'}))
    monkeypatch.setattr(sync.publish_results_snapshot, 'stage_candidate', lambda *a, **k: None)
    result = sync.synchronize(metadata['id'], paths=local)
    assert result['status'] == 'completed' and result['commit'] == 'b' * 40
    assert sum(a[0] == 'fetch' for a in calls) == 2
    assert sum(a[0] == 'push' for a in calls) == 1


@pytest.mark.parametrize('dataset,sync_enabled', [('historical', True), ('upcoming', False)])
def test_database_job_runs_integration_after_local_save_and_preserves_success_on_failure(tmp_path, monkeypatch, dataset, sync_enabled):
    paths = database(tmp_path / 'local')
    frame = records()
    monkeypatch.setattr(jobs, 'PATHS', paths)
    monkeypatch.setattr(jobs, 'NetkeibaDatabaseCreator', lambda **kwargs:
                        type('Scraper', (), {'parse_cached_date_range': lambda self, **k: frame})())
    calls = []
    monkeypatch.setattr(jobs, 'save_race_records', lambda *a, **k: calls.append('local-save'))
    monkeypatch.setattr(jobs, '_save_run', lambda *a, **k: None)
    monkeypatch.setattr(sync, 'enqueue', lambda *a, **k: 'batch')
    monkeypatch.setattr(sync, 'mark_local_saved', lambda *a, **k: None)
    def offline(*args, **kwargs):
        calls.append('remote-merge')
        raise RuntimeError('offline')
    monkeypatch.setattr(sync, 'synchronize', offline)
    job_id = 'job-' + dataset
    job = jobs.DatabaseCreationJob(job_id, dataset, date(2026, 8, 9), date(2026, 8, 9),
                                    sync_repository='example/private' if sync_enabled else None)
    import threading
    monkeypatch.setitem(jobs._JOBS, job_id, job)
    monkeypatch.setitem(jobs._CANCEL_EVENTS, job_id, threading.Event())
    jobs._run_job(job_id)
    assert job.status == 'completed' and job.local_saved
    assert calls[0] == 'local-save'
    if sync_enabled:
        assert calls == ['local-save', 'remote-merge'] and job.sync_status == 'failed'
    else:
        assert 'remote-merge' not in calls


def mock_pull(monkeypatch, remote, *, split=False):
    import shutil
    calls = []
    monkeypatch.setattr(sync.private_sync, 'repository_info', lambda *a, **k: {'branch': 'main', 'commit': 'a' * 40})
    def git(args, *, cwd):
        calls.append(args)
        if args[0] == 'fetch':
            if split:
                packed = cwd.parent / 'packed'
                private_snapshot.prepare(remote.root, packed)
                shutil.copytree(packed, cwd, dirs_exist_ok=True)
            else:
                for name in private_runtime.FILES:
                    target = cwd / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(remote.root / name, target)
        return 'a' * 40 if args[0] == 'rev-parse' else ''
    monkeypatch.setattr(sync.private_sync, '_git', git)
    monkeypatch.setattr(sync, 'build_candidate', lambda *a, **k: pytest.fail('pull must not build an upload'))
    return calls


@pytest.mark.parametrize('split', [False, True])
def test_repository_pull_needs_no_html_and_keeps_local_rows_and_models(tmp_path, monkeypatch, split):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    sync.result_update.save_day(remote.database, records(), date(2026, 8, 9))
    with duckdb.connect(str(local.database)) as con:
        con.execute("UPDATE race_records SET time_seconds=999")
        con.execute("INSERT INTO race_records(race_id,horse_id,dataset_type) VALUES ('future','h','upcoming')")
    protected = {name: private_runtime.digest(local.root / name) for name in sync.result_update.PROTECTED}
    calls = mock_pull(monkeypatch, remote, split=split)
    assert not list(local.historical_html.rglob('*.html'))
    result = sync.import_repository('example/private', paths=local)
    assert result['imported_rows'] == len(records())
    with duckdb.connect(str(local.database), read_only=True) as con:
        assert con.execute("SELECT time_seconds FROM race_records WHERE race_date=DATE '2026-08-02'").fetchone()[0] == 999
        assert con.execute("SELECT count(*) FROM race_records WHERE dataset_type='upcoming'").fetchone()[0] == 1
    again = sync.import_repository('example/private', paths=local)
    assert again['imported_rows'] == 0
    assert all(private_runtime.digest(local.root / name) == value for name, value in protected.items())
    assert not any(args[0] in {'push', 'commit'} for args in calls)


def test_repository_pull_creates_database_and_respects_date_range(tmp_path, monkeypatch):
    remote = database(tmp_path / 'remote')
    local = get_paths(tmp_path / 'empty-local')
    sync.result_update.save_day(remote.database, records(8), date(2026, 8, 8))
    sync.result_update.save_day(remote.database, records(9), date(2026, 8, 9))
    mock_pull(monkeypatch, remote)
    result = sync.import_repository('example/private', paths=local, start_date=date(2026, 8, 9), end_date=date(2026, 8, 9))
    assert result['imported_rows'] == len(records(9))
    with duckdb.connect(str(local.database), read_only=True) as con:
        assert con.execute('SELECT DISTINCT race_date FROM race_records').fetchall() == [(date(2026, 8, 9),)]


@pytest.mark.parametrize('failure', ['network', 'cancel', 'duplicates'])
def test_pull_failure_before_merge_preserves_existing_database(tmp_path, monkeypatch, failure):
    remote, local = database(tmp_path / 'remote'), database(tmp_path / 'local')
    mock_pull(monkeypatch, remote)
    if failure == 'network':
        monkeypatch.setattr(sync.private_sync, '_git', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
    if failure == 'duplicates':
        with duckdb.connect(str(remote.database)) as con:
            con.execute('INSERT INTO race_records SELECT * FROM race_records')
    before = private_runtime.digest(local.database)
    count = []
    def cancel():
        count.append(1)
        if failure == 'cancel' and len(count) == 3:
            raise RuntimeError('cancelled')
    with pytest.raises((RuntimeError, ValueError)):
        sync.import_repository('example/private', paths=local, check_cancel=cancel)
    assert private_runtime.digest(local.database) == before


@pytest.mark.parametrize('import_only', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
def test_database_job_imports_without_html(tmp_path, monkeypatch, import_only, cancel):
    import threading
    paths = database(tmp_path / 'local')
    monkeypatch.setattr(jobs, 'PATHS', paths)
    def scraper(**kwargs):
        if import_only:
            pytest.fail('repository-only mode must not initialize the HTML parser')
        return type('EmptyScraper', (), {'parse_cached_date_range': lambda self, **k: pd.DataFrame()})()
    monkeypatch.setattr(jobs, 'NetkeibaDatabaseCreator', scraper)
    monkeypatch.setattr(jobs, '_save_run', lambda *a, **k: None)
    calls = []
    cancel_event = threading.Event()
    def pull(repository, **kwargs):
        calls.append(kwargs)
        if cancel:
            cancel_event.set()
            kwargs['check_cancel']()
        kwargs['before_merge']()
        return {'commit': 'a' * 40, 'race_count': 2, 'row_count': 6, 'imported_rows': 6,
                'odds_count': 0, 'start_date': date(2026, 8, 9), 'end_date': date(2026, 8, 9)}
    monkeypatch.setattr(sync, 'import_repository', pull)
    job = jobs.DatabaseCreationJob('pull-test', 'historical', date(2026, 1, 1), date(2026, 12, 31),
                                   sync_repository='example/private', import_only=import_only)
    monkeypatch.setitem(jobs._JOBS, job.job_id, job)
    monkeypatch.setitem(jobs._CANCEL_EVENTS, job.job_id, cancel_event)
    jobs._run_job(job.job_id)
    if cancel:
        assert job.status == 'cancelled' and job.sync_status == 'cancelled' and not job.local_saved
        return
    assert job.status == 'completed' and job.local_saved and job.sync_status == 'completed'
    assert job.source == 'private_repository' and job.result['imported_rows'] == 6
    assert calls[0]['start_date'] == (None if import_only else date(2026, 1, 1))
