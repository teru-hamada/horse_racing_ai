"""Prospective policy plans must remain raw, frozen and officially settled."""
from importlib import import_module
import json
import shutil

import numpy as np
import pandas as pd
import pytest

policy = import_module('src.40_ai_modeling.common.ticket_selection')
cal = import_module('src.40_ai_modeling.common.calibration')
site = import_module('src.60_publication.static_site')


@pytest.fixture
def experiment(tmp_path):
    cal.save_calibrations(tmp_path / 'calibration.json', [cal.ProbabilityCalibration('m'),
        cal.ProbabilityCalibration('m', 'class_weight', {'pos_weight': 4.}),
        cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -.5})])
    frame = pd.DataFrame({'race_id': ['r'] * 8, 'race_date': ['2026-11-08'] * 8,
                          'horse_id': [f'h{i}' for i in range(1, 9)], 'horse_number': range(1, 9),
                          'frame_number': range(1, 9), 'odds': [2.] * 8, 'model_run_id': ['m'] * 8})
    frame = cal.attach_probabilities(frame, [.9, .5, .4, .3, .3, .2, .1, .05], tmp_path, {'model_run_id': 'm'})
    odds = pd.DataFrame({'race_id': ['r'] * 8, 'bet_type': ['place'] * 8,
                         'selection_1': range(1, 9), 'selection_2': [np.nan] * 8,
                         'selection_3': [np.nan] * 8, 'odds_min': [5., 4., 2.75, 4., 5., 3., 4., 5.]})
    output = tmp_path / 'run'
    cal.save_comparison_outputs(frame, output, odds)
    public = tmp_path / 'public'
    shutil.copytree(output / policy.CATEGORY, public / policy.CATEGORY / '2026-11-08')
    snapshot = tmp_path / 'inputs'
    hashes = policy.snapshot_inputs(public, '2026-11-08', snapshot)
    actual = frame[['race_id', 'horse_number', 'horse_id', 'frame_number']].copy()
    actual['finish_position'] = range(1, 9)
    actual['result_status'] = 'finished'
    actual.attrs['official_payouts'] = {'win:1': 120, 'place:1': 110, 'place:2': 150, 'place:3': 250}
    return frame, output, public, snapshot, hashes, actual


def test_cap_before_top3_and_existing_calibration_plans_unchanged(experiment):
    frame, output, _, _, _, _ = experiment
    main = pd.read_csv(output / 'calibration/none/recommended_bets.csv')
    other = pd.read_csv(output / 'ticket_selection/ev_cap2/recommended_bets.csv')
    assert main.selection.tolist() == [1, 2, 5]
    assert other.selection.tolist() == [2, 5, 4]
    assert other.stake_yen.eq(100).all()
    assert other.calibration_method.eq('none').all()
    raw = pd.read_csv(output / 'ticket_selection/ev_cap2/predictions.csv')
    np.testing.assert_allclose(raw.top3_probability, frame.raw_top3_probability)
    assert (output / 'calibration/class_weight/recommended_bets.csv').exists()
    assert (output / 'calibration/sigmoid/recommended_bets.csv').exists()


def test_boundaries_are_inclusive_and_all_filtered_means_empty_plan():
    candidates = pd.DataFrame({'race_id': ['r'] * 4, 'bet_type': ['place'] * 4,
        'selection': ['1', '2', '3', '4'], 'recovery_rate_percent': [109.9, 110., 200., 200.1],
        'recommendation_score': [109.9, 110., 200., 200.1]})
    _, plan = policy.select_candidates(candidates)
    assert plan.selection.tolist() == ['3', '2']
    _, empty = policy.select_candidates(candidates.iloc[[0, 3]])
    assert empty.empty and 'stake_yen' in empty


def test_settlement_uses_frozen_tickets_and_actual_payouts(tmp_path, experiment):
    frame, _, _, snapshot, _, actual = experiment
    result = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'settled')['ev_cap2']
    assert result['status'] == 'completed'
    assert result['purchase'] == 300 and result['payout'] == 150
    assert result['recovery_rate'] == 50
    saved = pd.read_csv(tmp_path / 'settled/ev_cap2/bets_results.csv')
    assert saved.selection.tolist() == [2, 5, 4]


def test_refunds_are_not_hits_and_missing_payouts_not_losses(tmp_path, experiment):
    frame, _, _, snapshot, _, actual = experiment
    actual.loc[actual.horse_number.eq(2), ['finish_position', 'result_status']] = [np.nan, 'scratched']
    actual.attrs['official_refunds'] = {'horse_numbers': [2]}
    row = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'refund')['ev_cap2']
    assert row['refund_bets'] == 1 and row['hit_bets'] == 0 and row['payout'] == 100
    actual.attrs['official_payouts'] = {'win:1': 120}
    row = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'pending')['ev_cap2']
    assert row['status'] == 'pending' and row['recovery_rate'] is None
    html = site._ticket_selection_return_block({'ev_cap2': row}, None, tmp_path)
    assert '精算中' in html and '回収率 0.0%' not in html


def test_snapshot_protects_definition_and_late_creation(tmp_path, experiment):
    _, _, public, _, hashes, _ = experiment
    policy.validate_snapshot(public, '2026-11-08', hashes)
    (public / 'ticket_selection/2026-11-08/ev_cap2/definition.json').write_text('{}')
    with pytest.raises(ValueError, match='変更'):
        policy.validate_snapshot(public, '2026-11-08', hashes)
    empty = policy.snapshot_inputs(tmp_path / 'missing', '2026-11-08', tmp_path / 'empty')
    late = tmp_path / 'missing/ticket_selection/2026-11-08/status.json'
    late.parent.mkdir(parents=True)
    late.write_text('{}')
    with pytest.raises(ValueError, match='変更'):
        policy.validate_snapshot(tmp_path / 'missing', '2026-11-08', empty)


def test_aggregation_incomplete_is_not_a_zero_purchase_day(tmp_path, experiment):
    _, output, _, _, _, _ = experiment
    policy.aggregate(tmp_path / 'day', [output, tmp_path / 'missing'])
    status = json.loads((tmp_path / 'day/ticket_selection/status.json').read_text())
    assert status['ev_cap2']['status'] == 'incomplete'
    assert status['ev_cap2']['races_available'] == 1
    assert not (tmp_path / 'day/ticket_selection/ev_cap2/recommended_bets.csv').exists()
    policy.aggregate(tmp_path / 'complete', [output])
    assert len(pd.read_csv(tmp_path / 'complete/ticket_selection/ev_cap2/recommended_bets.csv')) == 3


def test_missing_odds_for_a_race_is_unavailable_instead_of_a_skip(tmp_path, experiment):
    frame = experiment[0]
    odds = pd.DataFrame({'race_id': ['other'], 'bet_type': ['place'], 'selection_1': [1],
                         'selection_2': [np.nan], 'selection_3': [np.nan], 'odds_min': [3.]})
    policy.save(frame, tmp_path / 'partial', odds)
    status = json.loads((tmp_path / 'partial/ticket_selection/status.json').read_text(encoding='utf-8'))
    assert status['ev_cap2']['status'] == 'unavailable'
    assert not (tmp_path / 'partial/ticket_selection/ev_cap2/recommended_bets.csv').exists()


def test_empty_plan_missing_plan_and_invalid_definition_are_distinct(tmp_path, experiment):
    frame, _, _, snapshot, _, actual = experiment
    path = snapshot / 'ticket_selection/2026-11-08/ev_cap2/recommended_bets.csv'
    pd.read_csv(path).iloc[:0].to_csv(path, index=False)
    status_path = path.parent.parent / 'status.json'
    status = json.loads(status_path.read_text())
    status['ev_cap2']['recommended_rows'] = 0
    status_path.write_text(json.dumps(status))
    row = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'empty')['ev_cap2']
    assert row['status'] == 'no_bets'
    (path.parent / 'definition.json').write_text('{}')
    row = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'invalid')['ev_cap2']
    assert row['status'] == 'incomplete'
    path.unlink()
    row = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, tmp_path / 'missing')['ev_cap2']
    assert row['status'] == 'missing'


def test_html_aggregates_only_paired_completed_dates_and_refresh_is_idempotent(tmp_path, experiment):
    frame, _, public, snapshot, _, actual = experiment
    out = public / 'ticket_selection/2026-11-08'
    returns = policy.settle(snapshot, '2026-11-08', frame, {'r': actual}, out)
    main = pd.DataFrame({'race_id': ['r'], 'bet_type': ['place'], 'selection': ['1'],
                         'bet_result': ['的中'], 'payout_yen': [110]})
    main.to_csv(public / '2026-11-08_bets_results.csv', index=False)
    status = json.loads((out / 'status.json').read_text())
    html = site._ticket_selection_return_block(returns, status, public)
    assert '回収率 50.0%' in html and '回収率 110.0%' in html and '1開催日' in html
    pending = public / 'ticket_selection/2026-11-09'
    pending.mkdir()
    (pending / 'returns.json').write_text(json.dumps({'ev_cap2': {'status': 'pending'}}))
    assert site._ticket_selection_return_block(returns, status, public) == html
    page = public / '2026-11-08.html'
    page.write_text('<main><details class="race">unchanged bets</details></main>', encoding='utf-8')
    site.refresh_ticket_selection_sections(public.parent)
    # Refresh accepts a site root containing predictions/, as in publication.
    predictions = tmp_path / 'site/predictions'
    shutil.copytree(public, predictions)
    site.refresh_ticket_selection_sections(predictions.parent)
    first = (predictions / page.name).read_bytes()
    site.refresh_ticket_selection_sections(predictions.parent)
    assert (predictions / page.name).read_bytes() == first
    assert b'unchanged bets' in first
    assert first.count(b'ticket-selection-verification') == 1
