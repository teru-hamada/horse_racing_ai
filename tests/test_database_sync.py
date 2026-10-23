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
