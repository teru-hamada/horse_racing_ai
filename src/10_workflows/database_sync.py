"""Durable GUI result batches merged into latest private main, never local models."""
from datetime import date, timedelta
from importlib import import_module
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile

import duckdb
import pandas as pd

from scripts import private_runtime, private_snapshot, runtime_transfer, publish_results_snapshot
from . import private_sync, result_update

runtime_lock = import_module('src.00_common.runtime_lock').runtime_lock


def _root(paths):
    return paths.root / '.private-runtime/db-sync'


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_batch(batch_id, paths):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', batch_id):
        raise ValueError('Invalid sync batch ID')
    directory = _root(paths) / batch_id
    if directory.is_symlink() or not directory.resolve().is_relative_to(_root(paths).resolve()):
        raise ValueError('Unsafe sync batch directory')
    metadata = json.loads((directory / 'batch.json').read_text(encoding='utf-8'))
    if metadata.get('id') != batch_id:
        raise ValueError('Invalid sync batch metadata')
    return directory, metadata


def pending_batches(*, paths):
    results = []
    for path in sorted(_root(paths).glob('*/batch.json')):
        try:
            _, metadata = read_batch(path.parent.name, paths)
            if not metadata.get('local_saved', metadata.get('status') != 'prepared'):
                if not _registered(metadata, paths):
                    continue
                metadata['status'] = 'pending'
            if metadata.get('status') in {'pending', 'running', 'failed', 'remote_saved'}:
                results.append(metadata)
        except (OSError, ValueError, duckdb.Error):
            continue
    return results


def enqueue(frame, batch_id, repository, *, paths, prepared=False):
    """Persist only the imported batch and its available HTML for restart/retry."""
    if frame.empty or frame[['race_id', 'horse_id', 'race_date']].isna().any().any() or frame.duplicated(['race_id', 'horse_id']).any():
        raise ValueError('同期対象のキーが欠損・重複しています。')
    if not frame.race_id.astype(str).str.fullmatch(r'\d{12}').all():
        raise ValueError('Invalid race ID')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', batch_id):
        raise ValueError('Invalid sync batch ID')
    base = _root(paths)
    base.mkdir(parents=True, exist_ok=True)
    destination = base / batch_id
    if destination.exists():
        raise ValueError('Sync batch already exists')
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        stage = Path(temporary) / 'batch'
        (stage / 'html').mkdir(parents=True)
        frame.to_parquet(stage / 'payload.parquet', index=False)
        dated = frame.assign(race_date=pd.to_datetime(frame.race_date))
        raw = {}
        for race_id, rows in dated.groupby('race_id'):
            year = rows.race_date.iloc[0].year
            candidates = sorted((paths.historical_html / str(year) / 'result').glob(f'{race_id}*.html'))
            if candidates:
                target = stage / 'html' / f'{race_id}.html'
                shutil.copy2(candidates[-1], target)
                raw[str(race_id)] = private_runtime.digest(target)
        metadata = {'id': batch_id, 'repository': repository, 'status': 'prepared' if prepared else 'pending',
                    'local_saved': not prepared,
                    'dates': sorted(str(value.date()) for value in dated.race_date.unique()),
                    'rows': len(frame), 'races': int(frame.race_id.nunique()),
                    'payload_sha256': private_runtime.digest(stage / 'payload.parquet'),
                    'html_sha256': raw, 'error': ''}
        _write(stage / 'batch.json', metadata)
        stage.rename(destination)
    return batch_id


def _registered(metadata, paths):
    if not paths.database.exists():
        return False
    with duckdb.connect(str(paths.database), read_only=True) as connection:
        count = connection.execute("SELECT count(*) FROM race_records WHERE dataset_type='historical' AND collection_run_id=?", [metadata['id']]).fetchone()[0]
    return count == metadata['rows']


def mark_local_saved(batch_id, *, paths):
    directory, metadata = read_batch(batch_id, paths)
    if not _registered(metadata, paths):
        raise ValueError('ローカルDBの登録件数を確認できません。')
    metadata['status'] = 'pending'
    metadata['local_saved'] = True
    _write(directory / 'batch.json', metadata)


def _same_results(left, right):
    columns = ['horse_id', 'horse_number', 'finish_position', 'time_seconds']
    def canonical(frame):
        result = frame[columns].copy()
        result['horse_id'] = result.horse_id.astype('string')
        for name in columns[1:]:
            result[name] = pd.to_numeric(result[name], errors='coerce').astype('float64')
        return result.sort_values('horse_id').reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(canonical(left), canonical(right), check_dtype=False)
        return True
    except AssertionError:
        return False


def build_candidate(source, batch, metadata, work, *, collector_factory=result_update.DateCollector,
                    parser_factory=result_update.Creator):
    """Use latest remote records for overlaps; verify every new complete day."""
    source, batch, work = Path(source), Path(batch), Path(work)
    work.mkdir(parents=True)
    restored, candidate = work / 'runtime', work / 'candidate'
    runtime_transfer.materialize(source, restored)
    if private_runtime.digest(batch / 'payload.parquet') != metadata['payload_sha256']:
        raise ValueError('同期対象データのハッシュが一致しません。')
    incoming = pd.read_parquet(batch / 'payload.parquet')
    incoming['race_date'] = pd.to_datetime(incoming.race_date)
    database = restored / 'data/racing.duckdb'
    database_hash = private_runtime.digest(database)
    protected = {name: private_runtime.digest(restored / name) for name in result_update.PROTECTED}
    if (source / result_update.STATE_FILE).exists():
        state = json.loads((source / result_update.STATE_FILE).read_text(encoding='utf-8'))
    else:
        with duckdb.connect(str(database), read_only=True) as connection:
            anchor = connection.execute("SELECT max(race_date) FROM race_records WHERE dataset_type='historical'").fetchone()[0]
        state = {'format': 1, 'anchor': str(anchor), 'cursor': str(anchor), 'days': {}}
    if state.get('format') != 1:
        raise ValueError('Unknown remote update state')
    before_state = json.dumps(state, sort_keys=True)
    logger = logging.getLogger('gui-database-sync')
    collector, parser = collector_factory(logger, work / 'fetch'), parser_factory(logger)
    calendars = {}
    report = {'origin': 'gui', 'batch_id': metadata['id'], 'status': 'ok', 'days': [],
              'updated_days': 0, 'updated_rows': 0, 'model_updated': False, 'features_updated': False}
    try:
        for value, rows in incoming.groupby(incoming.race_date.dt.date):
            if value >= result_update.japan_today():
                raise ValueError('当日以降の過去結果は、日付が変わってから同期してください。')
            month = value.strftime('%Y%m')
            if month not in calendars:
                calendars[month] = result_update.request_with_retry(lambda: collector._download(
                    f'https://race.netkeiba.com/top/calendar.html?year={value.year}&month={value.month}',
                    work / f'fetch/calendar_{month}.html', force=True))
            if not result_update.parse_meeting_day(calendars[month], value):
                raise ValueError('GUIの結果と開催カレンダーが一致しません。')
            ids = result_update.request_with_retry(lambda: collector.race_ids_for_date(value, 'gui-sync', 'historical', force=True))
            if not ids or not set(rows.race_id) <= set(ids):
                raise ValueError('GUIの結果と掲載レース一覧が一致しません。')
            with duckdb.connect(str(database), read_only=True) as connection:
                existing = connection.execute("SELECT * FROM race_records WHERE dataset_type='historical' AND race_date=?", [value]).df()
            if not set(existing.race_id) <= set(ids):
                raise ValueError('Remote races disagree with listing')
            frames, changed, from_gui, fetched = [], False, 0, 0
            for race_id in ids:
                remote = existing[existing.race_id.eq(race_id)]
                local = rows[rows.race_id.eq(race_id)]
                if not remote.empty and (local.empty or _same_results(remote, local)):
                    # Preserve newer remote values and provenance for already matching races.
                    frames.append(remote)
                    continue
                path = batch / 'html' / f'{race_id}.html'
                if remote.empty and not local.empty and race_id in metadata['html_sha256']:
                    if private_runtime.digest(path) != metadata['html_sha256'][race_id]:
                        raise ValueError('保存HTMLのハッシュが一致しません。')
                    html = path.read_text(encoding='utf-8')
                    parsed = parser._parse_page(html, race_id, value, 'historical')
                    if not _same_results(parsed, local):
                        raise ValueError('保存HTMLがDB登録時から変更されています。DBを作成し直してください。')
                    result_update.validate_results(local, race_id, value, html)
                    frame = local
                    from_gui += 1
                else:
                    # Missing GUI races and conflicting values are resolved by a fresh result.
                    html = result_update.request_with_retry(lambda: collector._download(
                        collector.RESULT_FALLBACK_URL.format(race_id=race_id),
                        work / f'fetch/result/{race_id}.html', force=True))
                    frame = parser._parse_page(html, race_id, value, 'historical')
                    result_update.validate_results(frame, race_id, value, html)
                    fetched += 1
                    if not remote.empty and _same_results(remote, frame):
                        frames.append(remote)
                        continue
                frames.append(frame)
                changed = True
            combined = pd.concat(frames, ignore_index=True)
            if changed:
                result_update.save_day(database, combined, value)
                report['updated_days'] += 1
                report['updated_rows'] += len(combined)
            if changed or value > date.fromisoformat(state['anchor']) or str(value) in state['days']:
                state['days'][str(value)] = {'status': 'completed', 'race_rows': result_update.day_counts(database, value)}
            report['days'].append({'date': str(value), 'races': len(ids), 'rows': len(combined),
                                   'gui_races': from_gui, 'fetched_races': fetched, 'changed': changed})
    finally:
        collector.session.close()
        parser.session.close()
    cursor = date.fromisoformat(state['cursor'])
    while state['days'].get(str(cursor + timedelta(days=1)), {}).get('status') in {'completed', 'no_races'}:
        cursor += timedelta(days=1)
    state['cursor'] = str(cursor)
    changed = private_runtime.digest(database) != database_hash or json.dumps(state, sort_keys=True) != before_state
    if any(private_runtime.digest(restored / name) != digest for name, digest in protected.items()):
        raise ValueError('GUI result integration changed a model or feature file')
    if changed:
        private_snapshot.prepare(restored, candidate)
        _write(candidate / result_update.STATE_FILE, state)
    report.update(changed=changed, cursor=state['cursor'], model_and_features_unchanged=True)
    _write(work / 'report.json', report)
    return restored, candidate if changed else None, report


def merge_local(remote_database, batch_id, *, paths):
    """Pull historical rows without replacing local models or newer local imports."""
    columns = ','.join(result_update.storage.RACE_RECORD_COLUMNS)
    remote_path = str(Path(remote_database).resolve()).replace("'", "''")
    with duckdb.connect(str(paths.database)) as connection:
        connection.execute(f"ATTACH '{remote_path}' AS merged_source (READ_ONLY)")
        connection.execute('BEGIN TRANSACTION')
        try:
            connection.execute("DELETE FROM race_records WHERE dataset_type='historical' AND collection_run_id=?", [batch_id])
            connection.execute(f"""INSERT INTO race_records ({columns})
                SELECT {','.join('r.' + name for name in result_update.storage.RACE_RECORD_COLUMNS)}
                FROM merged_source.race_records r
                WHERE r.dataset_type='historical' AND NOT EXISTS (
                    SELECT 1 FROM race_records l WHERE l.dataset_type='historical'
                    AND l.race_id=r.race_id AND l.horse_id=r.horse_id)""")
            connection.execute('COMMIT')
        except BaseException:
            connection.execute('ROLLBACK')
            raise


def synchronize(batch_id, *, paths, log=lambda value: None):
    directory, metadata = read_batch(batch_id, paths)
    if metadata['status'] == 'completed':
        return metadata
    with runtime_lock(paths.root):
        try:
            if not metadata.get('local_saved', metadata['status'] != 'prepared') and not _registered(metadata, paths):
                raise ValueError('ローカルDBへの登録が未完了です。')
            metadata.update(status='running', error='', local_saved=True)
            _write(directory / 'batch.json', metadata)
            for attempt in range(3):
                info = private_sync.repository_info(metadata['repository'], root=paths.root)
                if info['branch'] != 'main':
                    raise ValueError('同期先のデフォルトブランチはmainである必要があります。')
                with tempfile.TemporaryDirectory(dir=_root(paths), prefix='merge-') as temporary:
                    temporary = Path(temporary)
                    checkout = temporary / 'repository'
                    checkout.mkdir()
                    git = lambda args: private_sync._git(args, cwd=checkout)
                    git(['init'])
                    git(['remote', 'add', 'origin', f'https://github.com/{metadata["repository"]}.git'])
                    git(['fetch', '--depth', '1', 'origin', info['commit']])
                    git(['checkout', '--detach', 'FETCH_HEAD'])
                    if git(['rev-parse', 'HEAD']) != info['commit']:
                        raise ValueError('Downloaded private commit mismatch')
                    log('非公開mainの最新DBへ統合しています。')
                    restored, candidate, report = build_candidate(checkout, directory, metadata, temporary / 'work')
                    commit = info['commit']
                    if candidate is not None:
                        if git(['ls-remote', 'origin', 'refs/heads/main']).split()[0] != info['commit']:
                            log('他の更新を検出しました。最新DBで統合し直します。')
                            continue
                        publish_results_snapshot.stage_candidate(checkout, candidate, info['commit'], temporary / 'work/report.json')
                        for name, fallback in (('user.name', 'Horse racing database sync'), ('user.email', 'database-sync@localhost')):
                            try:
                                identity = private_sync._git(['config', name], cwd=paths.root)
                            except RuntimeError:
                                identity = fallback
                            git(['config', name, identity])
                        git(['commit', '-m', 'Merge GUI historical result batch'])
                        try:
                            git(['push', 'origin', 'HEAD:refs/heads/main'])
                        except RuntimeError:
                            if private_sync.repository_info(metadata['repository'], root=paths.root)['commit'] != info['commit']:
                                log('保存先が更新されたため統合をやり直します。')
                                continue
                            raise
                        commit = git(['rev-parse', 'HEAD'])
                    metadata.update(status='remote_saved', commit=commit, report=report)
                    _write(directory / 'batch.json', metadata)
                    merge_local(restored / 'data/racing.duckdb', metadata['id'], paths=paths)
                    metadata.update(status='completed', error='')
                    _write(directory / 'batch.json', metadata)
                    log('非公開mainへの保存とローカルDBへの統合が完了しました。')
                    return metadata
            raise RuntimeError('保存先の更新が続いています。後でもう一度同期してください。')
        except Exception as exc:
            metadata.update(status='failed', error=str(exc))
            _write(directory / 'batch.json', metadata)
            raise
