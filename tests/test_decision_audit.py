"""取得時点と全候補の採否が、日次集約・公開後も追跡できることを検証する。"""
from datetime import date
from hashlib import sha256
from importlib import import_module
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

cal = import_module('src.40_ai_modeling.common.calibration')
betting = import_module('src.40_ai_modeling.betting')
parser = import_module('src.30_scrapers_database_creation.jra_odds_parser')
storage = import_module('src.00_common.storage')


def inputs(tmp_path):
    cal.save_calibrations(tmp_path / 'calibration.json', [
        cal.ProbabilityCalibration('m'),
        cal.ProbabilityCalibration('m', 'class_weight', {'pos_weight': 4.}),
        cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -.5}),
    ])
    frame = pd.DataFrame({'race_id': ['r'] * 6, 'race_date': ['2026-10-18'] * 6,
                          'horse_number': range(1, 7), 'odds': [3.] * 6, 'model_run_id': ['m'] * 6})
    frame = cal.attach_probabilities(frame, [.7, .6, .5, .4, .1, .8], tmp_path, {'model_run_id': 'm'})
    odds = pd.DataFrame({'race_id': ['r'] * 10, 'bet_type': ['place'] * 9 + ['unknown'],
                         'selection_1': [1, 2, 3, 4, 5, 6, 99, 1, 2, 1],
                         'selection_2': [np.nan] * 10, 'selection_3': [np.nan] * 10,
                         'odds_min': [3., 3., 3., 3., 20., 1., 4., 0., np.nan, 3.],
                         'fetched_at': ['2026-10-17T01:00:00+00:00'] * 10})
    return frame, odds


def test_all_candidates_and_reasons_match_actual_recommendations(tmp_path):
    frame, odds = inputs(tmp_path)
    cal.save_comparison_outputs(frame, tmp_path / 'run', odds)
    ids = set()
    for method in cal.METHODS:
        root = tmp_path / 'run/calibration' / method
        decisions = pd.read_csv(root / 'decisions.csv', dtype={'selection': str})
        assert len(decisions) == len(odds)
        assert decisions.candidate_id.is_unique
        ids.update(decisions.odds_snapshot_id)
        selected = decisions[decisions.selected].sort_values('recommendation_rank')
        actual = pd.read_csv(root / 'recommended_bets.csv', dtype={'selection': str})
        assert selected.selection.tolist() == actual.selection.tolist()
        np.testing.assert_allclose(selected.estimated_probability, actual.estimated_probability)
        assert decisions.fetched_at.eq('2026-10-17T01:00:00+00:00').all()
        assert pd.Timestamp(decisions.decision_at.iloc[0]).tzinfo is not None
    assert len(ids) == 1
    raw = pd.read_csv(tmp_path / 'run/calibration/none/decisions.csv')
    assert {'outside_top3', 'low_probability', 'low_expected_return', 'missing_prediction', 'invalid_odds', 'unsupported_bet_type'} <= set(raw.rejection_reason.dropna())
    invalid = raw[raw.rejection_reason.isin(['invalid_odds', 'missing_prediction'])]
    assert invalid.estimated_probability.isna().all()


def test_raw_exclusions_and_archives_survive_daily_aggregation(tmp_path):
    frame, odds = inputs(tmp_path)
    output = tmp_path / 'race'
    (output / 'fetch/odds').mkdir(parents=True)
    acquired = pd.concat([odds, odds.iloc[[0]].assign(selection_1=18)], ignore_index=True)
    acquired.to_csv(output / 'fetch/odds.csv', index=False)
    (output / 'fetch/odds/raw.html').write_text('<html>withdrawn</html>', encoding='utf-8')
    cal.save_comparison_outputs(frame, output, odds)
    decisions = pd.read_csv(output / 'calibration/none/decisions.csv')
    assert len(decisions) == 11
    assert decisions.iloc[-1].rejection_reason == 'excluded_from_prediction_inputs'
    assert not decisions.iloc[-1].selected
    cal.aggregate_comparison_outputs(tmp_path / 'daily', [output])
    archive = next((tmp_path / 'daily/calibration/archives').glob('*.zip'))
    with ZipFile(archive) as bundle:
        assert {'snapshot.json', 'odds.csv', 'audit_odds.csv', 'none/decisions.csv',
                'acquisition/odds.csv', 'acquisition/html/raw.html'} <= set(bundle.namelist())
        context = json.loads(bundle.read('snapshot.json'))
        assert sha256(bundle.read('odds.csv')).hexdigest() == context['odds_sha256']
        assert sha256(bundle.read('audit_odds.csv')).hexdigest() == context['audit_odds_sha256']
        manifest = json.loads(bundle.read('manifest.json'))
        for name, expected in manifest['sha256'].items():
            assert sha256(bundle.read(name)).hexdigest() == expected


def test_fetch_metadata_is_validated_and_preserved_in_existing_database(tmp_path):
    path = tmp_path / '202606040501_win_place.html'
    path.write_text('<table class="tanpuku"><tbody><tr><td class="num">1</td><td class="odds_tan">7.3</td></tr></tbody></table>', encoding='utf-8')
    legacy = parser.parse_odds_html(path, date(2026, 9, 19))
    assert legacy.fetched_at.isna().all()
    metadata = {'fetched_at': '2026-09-19T00:15:00+00:00', 'source_url': 'https://example.test/odds',
                'source_cname': 'request', 'source_sha256': sha256(path.read_bytes()).hexdigest()}
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    parsed = parser.parse_odds_html(path, date(2026, 9, 19))
    database = tmp_path / 'old.duckdb'
    duckdb = pytest.importorskip('duckdb')
    with duckdb.connect(str(database)) as con:
        con.execute(storage.SCHEMA_SQL)
    storage.save_race_odds(parsed, 'run', database=database)
    saved = storage.load_race_odds(database=database)
    assert saved.fetched_at.eq(metadata['fetched_at']).all()
    assert saved.source_sha256.eq(metadata['source_sha256']).all()
    path.write_text('changed HTML', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        parser.parse_odds_html(path, date(2026, 9, 19))


def test_policy_id_changes_when_threshold_changes(tmp_path, monkeypatch):
    frame, odds = inputs(tmp_path)
    cal.save_comparison_outputs(frame, tmp_path / 'first', odds)
    monkeypatch.setattr(betting, 'MIN_RECOVERY_RATE_PERCENT', 150.)
    cal.save_comparison_outputs(frame, tmp_path / 'second', odds)
    first = json.loads((tmp_path / 'first/calibration/snapshot.json').read_text(encoding='utf-8'))
    second = json.loads((tmp_path / 'second/calibration/snapshot.json').read_text(encoding='utf-8'))
    assert first['odds_snapshot_id'] == second['odds_snapshot_id']
    assert first['policy_id'] != second['policy_id']


def test_empty_odds_keeps_decision_schema(tmp_path):
    frame, odds = inputs(tmp_path)
    cal.save_comparison_outputs(frame, tmp_path / 'empty', odds.iloc[:0])
    decisions = pd.read_csv(tmp_path / 'empty/calibration/none/decisions.csv')
    assert decisions.empty
    assert {'selected', 'rejection_reason', 'policy_id', 'candidate_id'} <= set(decisions)


def test_tie_breaking_matches_recommendations(tmp_path):
    frame, odds = inputs(tmp_path)
    # 単勝と複勝のスコアが同じでも、従来の候補表の並び順と一致させる。
    odds = odds.iloc[:4].copy()
    odds.loc[0, 'bet_type'] = 'win'
    odds.loc[0, 'odds_min'] = 3. / .75 * frame.top3_probability.sum()
    cal.save_comparison_outputs(frame, tmp_path / 'tie', odds)
    for method in cal.METHODS:
        root = tmp_path / 'tie/calibration' / method
        decisions = pd.read_csv(root / 'decisions.csv', dtype={'selection': str})
        actual = pd.read_csv(root / 'recommended_bets.csv', dtype={'selection': str})
        chosen = decisions[decisions.selected].sort_values('recommendation_rank')
        assert list(zip(chosen.bet_type, chosen.selection)) == list(zip(actual.bet_type, actual.selection))
