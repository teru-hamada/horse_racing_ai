"""Validate cached or freshly downloaded results in a new DB copy."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from importlib import import_module
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
storage = import_module('src.00_common.storage')
Creator = import_module('src.30_scrapers_database_creation.scrapers_database_creation_netkeiba').NetkeibaDatabaseCreator
DateCollector = import_module('src.20_scrapers_html_collection.date_collector').DateCollector


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_training_input(output: Path):
    """Check generated columns and labels against the saved preprocessing schema."""
    import joblib
    import numpy as np

    frame = pd.read_parquet(output / 'parsed_results.parquet')
    frame = frame.loc[frame.finish_position.gt(0)].copy()
    for name in ('baseline', 'recent_speed', 'race_entry'):
        frame = frame.merge(pd.read_parquet(output / f'{name}.parquet'),
                            on=['race_id', 'horse_id'], validate='one_to_one')
    model_id = json.loads((ROOT / 'prediction_bundle/manifest.json').read_text(encoding='utf-8'))['model_id']
    model_dir = ROOT / 'models/top3' / model_id
    manifest = json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8'))
    align = import_module('src.40_ai_modeling.tasks.top3.feature_set').align_saved_top3_features
    prepare = import_module('src.40_ai_modeling.common.preprocessing').prepare_registered_feature_frame
    aligned = align(frame, manifest['features'])
    matrix = joblib.load(model_dir / 'preprocessor.joblib').transform(
        prepare(aligned, manifest['numeric_features'], manifest['categorical_features']))
    if matrix.shape[0] != len(frame) or not np.isfinite(matrix).all():
        raise ValueError('Invalid preprocessed training input.')
    aligned['target_top3'] = aligned.finish_position.le(3).astype('int8')
    aligned.to_parquet(output / 'training_input_smoke.parquet', index=False)
    return {'rows': len(frame), 'input_columns': matrix.shape[1],
            'positive_labels': int(aligned.target_top3.sum()), 'finite': True}


def stage_fresh_day(collector, day: date, output: Path, database: Path,
                    *, simulate_failure: bool = False):
    """Fetch only into this run's directory, checkpoint failures, then retry them."""
    ids = collector.race_ids_for_date(day, 'validation', 'historical', force=True)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Invalid race listing: {day}')
    checkpoint = {'date': str(day), 'races': {}, 'failure_injection': simulate_failure}
    initial_db_hash = digest(database)
    events = []

    def fetch(race_id):
        path = output / 'fetch/result' / f'{race_id}.html'
        url = collector.RESULT_FALLBACK_URL.format(race_id=race_id)
        for attempt in range(1, 4):
            try:
                collector._download(url, path, force=True)
                events.append({'race_id': race_id, 'attempt': attempt, 'status': 'ok'})
                return {'status': 'ok', 'path': str(path.relative_to(output)), 'sha256': digest(path)}
            except requests.RequestException as exc:
                events.append({'race_id': race_id, 'attempt': attempt,
                               'status': 'error', 'error': str(exc)})
                if attempt < 3:
                    time.sleep(2 * attempt)
        return {'status': 'error', 'error': events[-1]['error']}

    for position, race_id in enumerate(ids):
        if simulate_failure and position == 0:
            checkpoint['races'][race_id] = {'status': 'error', 'error': 'Injected timeout before HTTP request'}
        else:
            checkpoint['races'][race_id] = fetch(race_id)
        print(f'{day}: download {position + 1}/{len(ids)} {race_id}', flush=True)
    failed = [key for key, row in checkpoint['races'].items() if row['status'] != 'ok']
    checkpoint['status'] = 'incomplete' if failed else 'ok'
    checkpoint['failed_races'] = failed
    checkpoint['database_unchanged_before_completion'] = digest(database) == initial_db_hash
    path = output / f'download_checkpoint_{day:%Y%m%d}.json'
    path.write_text(json.dumps(checkpoint, indent=2), encoding='utf-8')
    # Read the persisted checkpoint, as a later execution would do.
    resumed = json.loads(path.read_text(encoding='utf-8'))
    for race_id in failed:
        resumed['races'][race_id] = fetch(race_id)
    resumed['failed_races'] = [key for key, row in resumed['races'].items() if row['status'] != 'ok']
    resumed['status'] = 'incomplete' if resumed['failed_races'] else 'ok'
    resumed['retry_requested'] = failed
    resumed['http_attempts'] = events
    resumed['all_downloads_network_only'] = all(not d.get('cache') for d in collector.diagnostics)
    (output / f'download_resumed_{day:%Y%m%d}.json').write_text(json.dumps(resumed, indent=2), encoding='utf-8')
    if resumed['status'] != 'ok' or not checkpoint['database_unchanged_before_completion']:
        raise ValueError(f'Incomplete downloads: {resumed["failed_races"]}')
    return ids, resumed


def run(output: Path, days: list[date], *, fresh_download=False,
        skip_features=False, simulate_failure=False):
    source = ROOT / 'data/racing.duckdb'
    feature_source = ROOT / 'data/features.duckdb'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = output / 'racing_validation.duckdb'
    if copied == source.resolve() or copied.exists():
        raise ValueError('A new isolated database path is required.')
    originals = {str(p): digest(p) for p in (source, feature_source)}
    report = {'status': 'running', 'days': [str(d) for d in days], 'checks': {},
              'scope': 'Selected dates only; not a continuous backfill.',
              'fresh_download': fresh_download}
    checks = report['checks']
    started = time.monotonic()
    try:
        # Refuse a live WAL rather than copy an inconsistent database snapshot.
        if Path(str(source) + '.wal').exists():
            raise ValueError('Source has a WAL; close the writer before copying.')
        shutil.copy2(source, copied)
        parser = Creator(logging.getLogger('historical-validation'))
        frames, coverage = [], []
        for day in days:
            if fresh_download:
                if (output / 'fetch').exists() and len(days) == 1:
                    raise ValueError('Fresh download requires a new HTML directory.')
                collector = DateCollector(logging.getLogger('fresh-download'), output / 'fetch')
                try:
                    ids, download_report = stage_fresh_day(collector, day, output, copied,
                                                          simulate_failure=simulate_failure)
                finally:
                    collector.session.close()
                report.setdefault('downloads', []).append(download_report)
                checks['network_only_downloads'] = download_report['all_downloads_network_only']
                checks['no_registration_while_download_incomplete'] = download_report['database_unchanged_before_completion']
                if simulate_failure:
                    checks['injected_failure_recovered'] = bool(download_report['retry_requested']) and download_report['status'] == 'ok'
                listing = output / f'fetch/race_list_db/{day:%Y%m%d}.html'
            else:
                listing = output / f'list_{day:%Y%m%d}.html'
                html = listing.read_bytes().decode('euc-jp', errors='replace')
                ids = parser._extract_race_ids(html, day.strftime('%Y%m%d'))
            if not ids:
                raise ValueError(f'No race IDs in independently fetched listing: {day}')
            row = {'date': str(day), 'expected_races': len(ids), 'parsed_races': 0,
                   'listing_sha256': digest(listing), 'files': [], 'missing_races': []}
            for race_id in ids:
                candidates = ([output / f'fetch/result/{race_id}.html'] if fresh_download else
                              list((ROOT / f'data/raw_html/historical/{day.year}/result').glob(f'{race_id}*.html')))
                if not candidates:
                    row['missing_races'].append(race_id)
                    continue
                path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
                frame = parser._parse_page(path.read_text(encoding='utf-8'), race_id, day, 'historical')
                if fresh_download and digest(path) != download_report['races'][race_id]['sha256']:
                    raise ValueError(f'Downloaded result changed after checkpoint: {race_id}')
                if frame.empty or frame[['horse_id', 'horse_number']].isna().any().any():
                    raise ValueError(f'Empty results or missing runner identifiers: {race_id}')
                if frame.duplicated(['race_id', 'horse_id']).any() or frame.horse_number.duplicated().any():
                    raise ValueError(f'Duplicate runner identifiers: {race_id}')
                if not frame.finish_position.gt(0).any():
                    raise ValueError(f'No confirmed finish positions: {race_id}')
                frames.append(frame)
                row['files'].append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
                row['parsed_races'] += 1
            coverage.append(row)
            print(f'{day}: parsed {row["parsed_races"]}/{row["expected_races"]} races', flush=True)
        report['coverage'] = coverage
        checks['all_listed_races_available'] = all(not r['missing_races'] for r in coverage)
        if not checks['all_listed_races_available']:
            raise ValueError('Some listed races lack cached results; do not register a partial day.')
        incoming = pd.concat(frames, ignore_index=True)
        if incoming.duplicated(['race_id', 'horse_id']).any():
            raise ValueError('Duplicate keys across selected days.')
        incoming.to_parquet(output / 'parsed_results.parquet', index=False)
        report['incoming'] = {'races': int(incoming.race_id.nunique()), 'rows': len(incoming),
            'missing_finish': int(incoming.finish_position.isna().sum()),
            'missing_time': int(incoming.time_seconds.isna().sum()),
            'finish_without_time': int((incoming.finish_position.gt(0) & incoming.time_seconds.isna()).sum())}
        checks['finishers_have_times'] = report['incoming']['finish_without_time'] == 0
        with duckdb.connect(str(copied), read_only=True) as con:
            before = con.execute("select count(*) from race_records where dataset_type='historical'").fetchone()[0]
            existing = con.execute("select * from race_records where dataset_type='historical'").df()
        overlap = incoming.merge(existing[['race_id', 'horse_id']], on=['race_id', 'horse_id'])
        checks['new_keys_only'] = overlap.empty
        if not overlap.empty:
            raise ValueError('Selected dates already exist in source historical data.')
        snapshots = []
        for attempt in (1, 2):
            storage.save_race_records(incoming, f'historical-validation-{attempt}', 'historical', database=copied)
            with duckdb.connect(str(copied), read_only=True) as con:
                snapshots.append(con.execute("select * from race_records where dataset_type='historical' order by race_id,horse_id").df())
        first, second = snapshots
        checks['expected_row_increase'] = len(first) == before + len(incoming)
        checks['no_duplicate_historical_keys'] = not second.duplicated(['race_id', 'horse_id']).any()
        payload = [c for c in storage.RACE_RECORD_COLUMNS if c not in {'collection_run_id', 'collected_at'}]
        pd.testing.assert_frame_equal(first[payload], second[payload])
        checks['reregistration_payload_unchanged'] = True
        old = second.merge(existing[['race_id', 'horse_id']], on=['race_id', 'horse_id']).sort_values(['race_id', 'horse_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(old[storage.RACE_RECORD_COLUMNS], existing.sort_values(['race_id', 'horse_id']).reset_index(drop=True)[storage.RACE_RECORD_COLUMNS])
        checks['existing_historical_rows_unchanged'] = True
        with duckdb.connect(str(source), read_only=True) as con, duckdb.connect(str(copied), read_only=True) as copy_con:
            report['schema_additions'] = {}
            for (table,) in con.execute('show tables').fetchall():
                if table == 'race_records':
                    query = "select * from race_records where dataset_type <> 'historical' order by race_id,horse_id"
                else:
                    query = f'SELECT * FROM "{table}" ORDER BY ALL'
                original_frame = con.execute(query).df()
                copied_frame = copy_con.execute(query).df()
                additions = sorted(set(copied_frame.columns) - set(original_frame.columns))
                if additions:
                    # storage.connect performs this existing, additive migration.
                    if table != 'race_odds' or not set(additions) <= {'fetched_at', 'source_url', 'source_cname', 'source_sha256'}:
                        raise ValueError(f'Unexpected schema additions: {table}: {additions}')
                    if not copied_frame[additions].isna().all().all():
                        raise ValueError('Existing odds acquired unexpected provenance values.')
                    report['schema_additions'][table] = additions
                pd.testing.assert_frame_equal(original_frame, copied_frame[original_frame.columns])
        checks['other_tables_and_upcoming_unchanged'] = True
        report['historical_rows_before'] = before
        report['historical_rows_after'] = len(second)
        if not skip_features:
            verify_features(output, incoming, second, feature_source, report)
        else:
            report['feature_generation'] = 'skipped; this run validates acquisition and DB registration'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        logging.exception('Validation failed')
    finally:
        checks['source_databases_unchanged'] = all(digest(Path(p)) == h for p, h in originals.items())
        report['source_sha256'] = originals
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        report['status'] = 'ok' if checks and all(checks.values()) and 'error' not in report else 'incomplete'
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'coverage', 'downloads'}}, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'ok' else 1


def verify_features(output, incoming, history, feature_source, report):
    generators = import_module('src.40_ai_modeling.feature_engineering.generators')
    Context = import_module('src.40_ai_modeling.feature_engineering.context').FeatureContext
    targets = incoming[['race_id', 'horse_id', 'race_date']]
    outputs = {}
    print('DB registration and re-registration checks passed; generating features', flush=True)
    outputs['baseline'] = generators.RecentFormGenerator().transform(targets, history, Context('baseline', '1.1.0'))
    outputs['race_entry'] = generators.RaceEntryGenerator().transform(incoming, history, Context('race_entry', '1.1.0'))
    with duckdb.connect(str(feature_source), read_only=True) as con:
        speeds = con.execute("select race_id,horse_id,race_date,speed_index from performance_features where performance_feature_name='speed_index' and performance_feature_version='1.0.0'").df()
    added_speed = generators.SpeedIndexGenerator().transform(targets, history, Context('speed_index', '1.0.0'))
    added_speed = added_speed.merge(targets, on=['race_id', 'horse_id'], validate='one_to_one')
    speed_history = pd.concat([speeds, added_speed[['race_id', 'horse_id', 'race_date', 'speed_index']]], ignore_index=True)
    outputs['recent_speed'] = generators.RecentSpeedGenerator().transform(targets, speed_history, Context('recent_speed', '1.0.0'))
    report['features'] = {}
    for name, frame in outputs.items():
        report['checks'][f'{name}_all_target_keys'] = len(frame) == len(targets) and set(map(tuple, frame[['race_id', 'horse_id']].values)) == set(map(tuple, targets[['race_id', 'horse_id']].values)) and not frame.duplicated(['race_id', 'horse_id']).any()
        frame.to_parquet(output / f'{name}.parquet', index=False)
        report['features'][name] = {'rows': len(frame), 'columns': len(frame.columns)}
    report['training_input'] = verify_training_input(output)
    report['checks']['training_input_matches_saved_preprocessor'] = True
    report['note'] = 'Feature generation is a smoke check. Unimported intervening dates remain absent from history; this is not a production-ready training dataset.'


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--dates', type=date.fromisoformat, nargs='+', required=True)
    cli.add_argument('--fresh-download', action='store_true', help='Fetch new listings/results; bypass every existing cache')
    cli.add_argument('--skip-features', action='store_true', help='Validate acquisition/registration only')
    cli.add_argument('--simulate-failure', action='store_true', help='Stage one injected failure and verify checkpoint retry')
    args = cli.parse_args()
    if args.simulate_failure and not args.fresh_download:
        cli.error('--simulate-failure requires --fresh-download')
    raise SystemExit(run(args.output, args.dates, fresh_download=args.fresh_download,
                         skip_features=args.skip_features, simulate_failure=args.simulate_failure))
