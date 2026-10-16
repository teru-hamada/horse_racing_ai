"""購入対象の固定・金額・表示・精算が同じ契約に従うことを検証する。"""
from importlib import import_module

import pandas as pd
import pytest

plan = import_module('src.00_common.purchase_plan')
comparison = import_module('src.50_result_comparison.prediction_comparison')
site = import_module('src.60_publication.static_site')


def candidates():
    return pd.DataFrame([{'race_id': 'r', 'bet_type': 'win', 'selection': str(i),
                          'recommendation_score': 500 - i, 'recovery_rate_percent': 200.}
                         for i in range(1, 5)])


def test_frozen_plan_is_not_reselected_when_scores_change():
    frozen = plan.select_plan(candidates())
    assert frozen.selection.tolist() == ['1', '2', '3']
    frozen['recommendation_score'] = [0., 10000., 1000.]
    displayed = site._displayed_bets(frozen)
    pd.testing.assert_frame_equal(displayed, frozen)
    judged = comparison.compare_recommended_bets(displayed, {'r': {'win:4': 9000}})
    summary = plan.return_summary(judged)
    assert summary['purchase'] == 300
    assert summary['payout'] == 0
    assert summary['recovery_rate'] == 0
    assert len(judged) == 3


def test_stakes_refunds_and_pending_use_the_same_summary():
    frozen = plan.select_plan(candidates())
    frozen['stake_yen'] = [200, 300, 100]
    judged = comparison.compare_recommended_bets(frozen, {'r': {'win:1': 300}},
                                                {'r': {'horse_numbers': [2]}})
    assert judged.payout_yen.tolist() == [600, 300, 0]
    expected = plan.return_summary(judged)
    assert expected['purchase'] == 600
    assert expected['payout'] == 900
    assert expected['recovery_rate'] == 150
    assert expected['hit_bets'] == 1
    assert expected['refund_bets'] == 1
    assert site._bet_return_summary(frozen, {'r': {'win:1': 300}}, {'r': {'horse_numbers': [2]}}) == expected
    pending = comparison.compare_recommended_bets(frozen, {}, {'r': {'horse_numbers': [2]}})
    summary = plan.return_summary(pending)
    assert summary['recovery_rate'] is None
    assert summary['settled_bets'] == 1
    assert summary['payout'] == 300


@pytest.mark.parametrize('column,value', [('stake_yen', 0), ('stake_yen', 150),
                                         ('stake_yen', float('inf')), ('purchase_rank', 2),
                                         ('purchase_plan_version', 2)])
def test_corrupt_plan_is_rejected_instead_of_reselected(column, value):
    frozen = plan.select_plan(candidates())
    frozen[column] = frozen[column].astype(float)
    frozen.loc[0, column] = value
    with pytest.raises(ValueError):
        site._displayed_bets(frozen)


def test_saved_plan_ignores_current_candidate_scores_and_legacy_saved_order(tmp_path):
    frozen = plan.select_plan(candidates())
    frozen.to_csv(tmp_path / 'recommended_bets.csv', index=False)
    candidates().assign(recommendation_score=[0., 0., 0., 9999.]).to_csv(tmp_path / 'bets.csv', index=False)
    assert plan.load_saved_plan(tmp_path).selection.tolist() == ['1', '2', '3']
    (tmp_path / 'recommended_bets.csv').unlink()
    legacy = tmp_path / 'calibration/none'
    legacy.mkdir(parents=True)
    # 旧比較ファイルは既に選定済みなので、スコア順に並べ直さない。
    saved = candidates().iloc[[2, 0]].copy()
    saved.to_csv(legacy / 'recommended_bets.csv', index=False)
    assert plan.load_saved_plan(tmp_path).selection.tolist() == ['3', '1']


def test_no_bets_is_a_saved_empty_plan(tmp_path):
    frozen = plan.select_plan(candidates().iloc[:0])
    frozen.to_csv(tmp_path / 'recommended_bets.csv', index=False)
    saved = plan.load_saved_plan(tmp_path)
    assert saved.empty
    judged = comparison.compare_recommended_bets(saved, {})
    assert plan.return_summary(judged)['recovery_rate'] is None
