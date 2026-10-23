"""Update complete historical race days in a copy, preserving every model file."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from importlib import import_module
import json
import logging
from pathlib import Path
import shutil
import time
import uuid

from bs4 import BeautifulSoup
import duckdb
import pandas as pd
import requests

from scripts import private_runtime, private_snapshot, runtime_transfer

DateCollector = import_module('src.20_scrapers_html_collection.date_collector').DateCollector
Creator = import_module('src.30_scrapers_database_creation.scrapers_database_creation_netkeiba').NetkeibaDatabaseCreator
parse_meeting_day = import_module('src.20_scrapers_html_collection.race_calendar').parse_meeting_day
storage = import_module('src.00_common.storage')
STATE_FILE = 'result_update_state.json'
PROTECTED = tuple(name for name in private_runtime.FILES if name != 'data/racing.duckdb')


def japan_today():
    return datetime.now(timezone(timedelta(hours=9))).date()


def request_with_retry(operation):
    for attempt in range(3):
        try:
            return operation()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def validate_results(frame, race_id, day, html):
    if frame.empty or not frame.race_id.eq(race_id).all() or not pd.to_datetime(frame.race_date).dt.date.eq(day).all():
        raise ValueError('Result identifiers or date are invalid')
    if frame[['horse_id', 'horse_number']].isna().any().any() or frame.duplicated(['race_id', 'horse_id']).any() or frame.horse_number.duplicated().any():
        raise ValueError('Result runners are missing or duplicated')
    finishers = frame.finish_position.gt(0)
    if not finishers.any() or frame.loc[finishers, 'time_seconds'].isna().any() or not frame.loc[finishers, 'time_seconds'].gt(0).all():
        raise ValueError('Finished runners lack valid results or times')
    soup = BeautifulSoup(html, 'lxml')
    statuses = {}
    for _, runner in frame.loc[~finishers].iterrows():
        row = next((a.find_parent('tr') for a in soup.select('a[href]')
                    if f'/horse/{runner.horse_id}' in a['href']), None)
        cell = row.find('td') if row else None
        label = cell.get_text(strip=True) if cell else ''
        if label not in {'取', '取消', '除', '除外', '中', '中止', '失', '失格'}:
            raise ValueError('Unexplained missing finish position')
        statuses[label] = statuses.get(label, 0) + 1
    return statuses


def save_day(database, frame, day):
    """Replace a verified complete day atomically in the work copy only."""
    normalized = storage.normalize_race_frame(frame)
    normalized['dataset_type'] = 'historical'
    normalized['collection_run_id'] = 'result-update-' + uuid.uuid4().hex
    normalized['collected_at'] = datetime.now(timezone.utc).replace(tzinfo=None)
    with duckdb.connect(str(database)) as connection:
        connection.register('incoming_result_day', normalized)
        connection.execute('BEGIN TRANSACTION')
        try:
            connection.execute("DELETE FROM race_records WHERE dataset_type='historical' AND race_date=?", [day])
            columns = ','.join(storage.RACE_RECORD_COLUMNS)
            connection.execute(f'INSERT INTO race_records ({columns}) SELECT {columns} FROM incoming_result_day')
            count, unique = connection.execute("SELECT count(*),count(distinct (race_id,horse_id)) FROM race_records WHERE dataset_type='historical' AND race_date=?", [day]).fetchone()
            if count != len(frame) or unique != len(frame):
                raise ValueError('Registered result count mismatch')
            connection.execute('COMMIT')
        except BaseException:
            connection.execute('ROLLBACK')
            raise


def day_counts(database, day):
    with duckdb.connect(str(database), read_only=True) as connection:
        return {race_id: count for race_id, count in connection.execute(
            "SELECT race_id,count(*) FROM race_records WHERE dataset_type='historical' AND race_date=? GROUP BY race_id", [day]).fetchall()}


def plan_dates(state, start, end, today, max_days):
    if not 1 <= max_days <= 7:
        raise ValueError('At most seven calendar days can be processed per run')
    yesterday = today - timedelta(days=1)
    if (start is None) != (end is None):
        raise ValueError('Start and end must both be specified')
    first = start or date.fromisoformat(state['cursor']) + timedelta(days=1)
    last = end or min(yesterday, first + timedelta(days=max_days - 1))
    if start is not None and (last < first or last > yesterday):
        raise ValueError('Only completed past calendar days can be requested')
    if last < first:
        return []
    if (last - first).days + 1 > max_days:
        raise ValueError('At most seven calendar days can be processed per run')
    return [first + timedelta(days=index) for index in range((last - first).days + 1)]


def update(source: Path, output: Path, work: Path, *, start=None, end=None,
           max_days=7, today=None, collector_factory=DateCollector, parser_factory=Creator):
    source, output, work = Path(source).resolve(), Path(output).resolve(), Path(work).resolve()
    if output.exists() or work.exists() or output == source or work == source:
        raise ValueError('New output and work directories are required')
    if source in output.parents or source in work.parents:
        raise ValueError('Work and candidate must be outside the source checkout')
    work.mkdir(parents=True)
    restored = work / 'runtime'
    runtime_transfer.materialize(source, restored)
    database = restored / 'data/racing.duckdb'
    protected = {name: private_runtime.digest(restored / name) for name in PROTECTED}
    original_database_hash = private_runtime.digest(database)
    state_path = source / STATE_FILE
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('format') != 1 or not isinstance(state.get('days'), dict):
            raise ValueError('Unsupported update state')
    else:
        with duckdb.connect(str(database), read_only=True) as connection:
            anchor = connection.execute("SELECT max(race_date) FROM race_records WHERE dataset_type='historical'").fetchone()[0]
        if anchor is None:
            raise ValueError('An existing historical DB is required')
        state = {'format': 1, 'anchor': str(anchor), 'cursor': str(anchor), 'days': {}}
    state_before = json.dumps(state, sort_keys=True)
    days = plan_dates(state, start, end, today or japan_today(), max_days)
    report = {'status': 'ok', 'days': [], 'updated_days': 0, 'updated_rows': 0,
              'model_updated': False, 'features_updated': False}
    logger = logging.getLogger('result-update')
    collector = collector_factory(logger, work / 'html')
    parser = parser_factory(logger)
    calendars = {}
    try:
        for day in days:
            key = str(day)
            result = {'date': key}
            try:
                month = day.strftime('%Y%m')
                if month not in calendars:
                    calendars[month] = request_with_retry(lambda: collector._download(
                        f'https://race.netkeiba.com/top/calendar.html?year={day.year}&month={day.month}',
                        work / f'html/calendar_{month}.html', force=True))
                if not parse_meeting_day(calendars[month], day):
                    if day_counts(database, day):
                        raise ValueError('Calendar conflicts with existing historical races')
                    state['days'][key] = {'status': 'no_races'}
                    result['status'] = 'no_races'
                else:
                    ids = request_with_retry(lambda: collector.race_ids_for_date(day, 'result-update', 'historical', force=True))
                    if not ids or len(ids) != len(set(ids)):
                        raise ValueError('Meeting day has an empty or duplicate race listing')
                    existing = day_counts(database, day)
                    previous = state['days'].get(key, {})
                    if previous.get('status') == 'completed' and set(ids) == set(existing) and previous.get('race_rows') == existing:
                        result.update(status='already_completed', races=len(ids), rows=sum(existing.values()))
                    else:
                        if set(existing) - set(ids):
                            raise ValueError('Listed races disagree with previously stored races')
                        frames, statuses = [], {}
                        for index, race_id in enumerate(ids, 1):
                            html = request_with_retry(lambda: collector._download(
                                collector.RESULT_FALLBACK_URL.format(race_id=race_id),
                                work / f'html/result/{race_id}.html', force=True))
                            frame = parser._parse_page(html, race_id, day, 'historical')
                            statuses[race_id] = validate_results(frame, race_id, day, html)
                            frames.append(frame)
                            print(f'{day}: verified race {index}/{len(ids)}', flush=True)
                        incoming = pd.concat(frames, ignore_index=True)
                        save_day(database, incoming, day)
                        counts = day_counts(database, day)
                        state['days'][key] = {'status': 'completed', 'race_rows': counts}
                        result.update(status='completed', races=len(ids), rows=len(incoming), nonfinish_statuses=statuses)
                        report['updated_days'] += 1
                        report['updated_rows'] += len(incoming)
            except Exception as exc:
                result.update(status='error', error=f'{type(exc).__name__}: {exc}')
                report['status'] = 'incomplete'
                logger.error('%s: %s', day, result['error'])
            report['days'].append(result)
    finally:
        collector.session.close()
        parser.session.close()
    cursor = date.fromisoformat(state['cursor'])
    while state['days'].get(str(cursor + timedelta(days=1)), {}).get('status') in {'completed', 'no_races'}:
        cursor += timedelta(days=1)
    state['cursor'] = str(cursor)
    database_changed = private_runtime.digest(database) != original_database_hash
    changed = database_changed or json.dumps(state, sort_keys=True) != state_before
    if any(private_runtime.digest(restored / name) != digest for name, digest in protected.items()):
        raise ValueError('Model, bundle, or feature DB changed during result-only update')
    if changed:
        if not database_changed and (source / private_snapshot.MANIFEST).exists():
            manifest = private_snapshot.inspect_manifest(source)
            output.mkdir(parents=True)
            names = [private_snapshot.MANIFEST, '.gitattributes', *runtime_transfer.METADATA,
                     *(part['path'] for part in manifest['parts'])]
            for name in names:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, target)
        else:
            private_snapshot.prepare(restored, output)
        (output / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report.update(changed=changed, cursor=state['cursor'], database_changed=database_changed,
                  model_and_features_unchanged=True)
    (work / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
