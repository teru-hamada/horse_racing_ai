"""保存した比較買い目の公式精算・返還・不足状態とHTML表示を検証する。"""
from importlib import import_module
import json
from pathlib import Path

import pandas as pd
import pytest

settlement = import_module('src.50_result_comparison.calibration_comparison')
site = import_module('src.60_publication.static_site')


@pytest.fixture
def inputs(tmp_path):
    public = pd.DataFrame({'race_id': ['r', 'r'], 'horse_id': ['h1', 'h2'],
                           'horse_number': [1, 2], 'race_date': ['2026-10-18'] * 2,
                           'model_run_id': ['m'] * 2})
    root = tmp_path / 'inputs/calibration/2026-10-18'
    for method in settlement.METHODS:
        directory = root / method
        directory.mkdir(parents=True)
        predictions = public.assign(top3_probability=.3, calibration_method=method, calibration_id='cal_' + method)
        predictions.to_csv(directory / 'predictions.csv', index=False)
        bets = pd.DataFrame({'race_id': ['r', 'r'], 'bet_type': ['win', 'win'],
                             'selection': ['1', '2'], 'estimated_probability': [.3, .2],
                             'odds_used': [4., 6.], 'calibration_method': method, 'calibration_id': 'cal_' + method})
        bets.to_csv(directory / 'recommended_bets.csv', index=False)
    actual = public[['race_id', 'horse_id', 'horse_number']].assign(
        finish_position=[1, 2], result_status='finished', frame_number=[1, 2])
    actual.attrs['official_payouts'] = {'win:1': 400}
    return public, actual, tmp_path / 'inputs'


def settle(tmp_path, inputs):
    public, actual, snapshot = inputs
    return settlement.settle_saved_comparisons(snapshot, '2026-10-18', public, {'r': actual}, tmp_path / 'result')


def test_official_payouts_apply_to_saved_tickets_without_reselection(tmp_path, inputs):
    result = settle(tmp_path, inputs)
    for method in settlement.METHODS:
        row = result[method]
        assert row['status'] == 'completed'
        assert row['purchase'] == 200 and row['payout'] == 400
        assert row['recovery_rate'] == 200 and row['hit_bets'] == 1
        saved = pd.read_csv(tmp_path / f'result/{method}/bets_results.csv')
        assert saved.selection.tolist() == [1, 2]
    assert '重み付け補正【検証中】' in site._calibration_return_block(result)
    assert 'sigmoid補正【検証中】' in site._calibration_return_block(result)
    assert '回収率 200.0%' in site._calibration_return_block(result)


def test_refund_is_returned_stake_not_a_hit(tmp_path, inputs):
    public, actual, snapshot = inputs
    actual.loc[1, ['finish_position', 'result_status']] = [float('nan'), 'scratched']
    actual.attrs['official_refunds'] = {'horse_numbers': [2]}
    result = settle(tmp_path, inputs)['class_weight']
    assert result['payout'] == 500 and result['recovery_rate'] == 250
    assert result['hit_bets'] == 1 and result['refund_bets'] == 1


def test_missing_payouts_never_display_zero_roi(tmp_path, inputs):
    inputs[1].attrs['official_payouts'] = {'place:1': 150}
    result = settle(tmp_path, inputs)
    assert result['sigmoid']['status'] == 'pending'
    assert result['sigmoid']['recovery_rate'] is None
    html = site._calibration_return_block(result)
    assert '集計中' in html and '回収率 0.0%' not in html


def test_empty_tickets_are_distinct_from_missing_comparison(tmp_path, inputs):
    path = inputs[2] / 'calibration/2026-10-18/class_weight/recommended_bets.csv'
    pd.read_csv(path).iloc[:0].to_csv(path, index=False)
    missing = inputs[2] / 'calibration/2026-10-18/sigmoid/recommended_bets.csv'
    missing.unlink()
    result = settle(tmp_path, inputs)
    assert result['class_weight']['status'] == 'no_bets'
    assert result['sigmoid']['status'] == 'missing'
    html = site._calibration_return_block(result)
    assert '購入対象なし（見送り）' in html and '比較データなし' in html
    assert '回収率 0.0%' not in html


def test_missing_official_result_never_becomes_loss(tmp_path, inputs):
    public, _, snapshot = inputs
    result = settlement.settle_saved_comparisons(snapshot, '2026-10-18', public, {}, tmp_path / 'result')
    assert result['class_weight']['status'] == 'pending'
    assert result['class_weight']['recovery_rate'] is None


def test_corrupt_method_does_not_hide_other_method(tmp_path, inputs):
    path = inputs[2] / 'calibration/2026-10-18/class_weight/predictions.csv'
    frame = pd.read_csv(path)
    frame['model_run_id'] = 'other'
    frame.to_csv(path, index=False)
    result = settle(tmp_path, inputs)
    assert result['class_weight']['status'] == 'incomplete'
    assert result['sigmoid']['status'] == 'completed'


def test_snapshot_detects_changes_and_late_creation(tmp_path):
    root = tmp_path / 'predictions'
    root.mkdir()
    hashes = settlement.snapshot_inputs(root, '2026-10-18', tmp_path / 'snapshot')
    settlement.validate_snapshot(root, '2026-10-18', hashes)
    path = root / 'calibration/2026-10-18/sigmoid/recommended_bets.csv'
    path.parent.mkdir(parents=True)
    path.write_text('changed')
    with pytest.raises(ValueError, match='変更'):
        settlement.validate_snapshot(root, '2026-10-18', hashes)
