"""同一の生出力・オッズで3方式を比較し、公開方式を変えずに保存する。"""
from importlib import import_module
import json

import numpy as np
import pandas as pd
import pytest

cal = import_module('src.40_ai_modeling.common.calibration')


def comparison_input(tmp_path):
    cal.save_calibrations(tmp_path / 'calibration.json', [
        cal.ProbabilityCalibration('m'),
        cal.ProbabilityCalibration('m', 'class_weight', {'pos_weight': 4.}),
        cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -.5}),
    ])
    frame = pd.DataFrame({'race_id': ['r', 'r'], 'race_date': ['2026-10-18'] * 2,
                          'horse_id': ['h1', 'h2'], 'horse_number': [1, 2], 'odds': [3., 4.]})
    odds = pd.DataFrame({'race_id': ['r', 'r'], 'bet_type': ['place', 'place'],
                         'selection_1': [1, 2], 'selection_2': [np.nan] * 2,
                         'selection_3': [np.nan] * 2, 'odds_min': [3., 4.]})
    return cal.attach_probabilities(frame, [.5, .3], tmp_path, {'model_run_id': 'm'}), odds


def test_none_saves_all_three_values_and_independent_bets(tmp_path):
    frame, odds = comparison_input(tmp_path)
    np.testing.assert_array_equal(frame.top3_probability, [.5, .3])
    np.testing.assert_allclose(frame.calibration_class_weight_probability, [.2, .3 / (4 - 3 * .3)])
    assert frame.calibration_sigmoid_probability.iloc[0] == pytest.approx(1 / (1 + np.exp(.5)))
    output = tmp_path / 'run'
    status = cal.save_comparison_outputs(frame, output, odds)
    assert status['none']['recommended_rows'] == 2
    assert status['class_weight']['recommended_rows'] == 0
    assert status['sigmoid']['recommended_rows'] == 1
    for method in cal.METHODS:
        directory = output / 'calibration' / method
        saved = pd.read_csv(directory / 'predictions.csv')
        assert saved.calibration_method.eq(method).all()
        assert (directory / 'predictions.parquet').exists()
        bets = pd.read_csv(directory / 'bets.csv')
        assert set(bets.columns) >= {'calibration_method', 'estimated_probability', 'recovery_rate_percent'}
    # noneを公開に使ったまま、別方式の候補・点数が実際に変わっている。
    assert frame.calibration_method.eq('none').all()
    pd.testing.assert_frame_equal(pd.read_csv(output / 'calibration/odds.csv'), odds, check_dtype=False)


def test_missing_definition_is_not_saved_as_fake_raw_correction(tmp_path):
    frame = pd.DataFrame({'race_id': ['r'], 'odds': [3.]})
    result = cal.attach_probabilities(frame, [.5], tmp_path, {'model_run_id': 'legacy'})
    assert result.calibration_class_weight_status.iloc[0] == 'unavailable'
    assert result.calibration_sigmoid_probability.isna().all()
    output = tmp_path / 'run'
    status = cal.save_comparison_outputs(result, output)
    assert status['sigmoid']['status'] == 'unavailable'
    assert not (output / 'calibration/sigmoid/predictions.csv').exists()


def test_corrupt_comparison_definition_still_stops_prediction(tmp_path):
    cal.save_calibrations(tmp_path / 'calibration.json', [cal.ProbabilityCalibration('other')])
    with pytest.raises(ValueError, match='mismatch'):
        cal.attach_probabilities(pd.DataFrame({'race_id': ['r']}), [.5], tmp_path, {'model_run_id': 'm'})


def test_date_aggregation_keeps_empty_bets_and_all_methods(tmp_path):
    frame, odds = comparison_input(tmp_path)
    directories = [tmp_path / 'race1', tmp_path / 'race2']
    cal.save_comparison_outputs(frame, directories[0], odds)
    cal.save_comparison_outputs(frame.assign(race_id='r2'), directories[1], odds.assign(race_id='r2'))
    output = tmp_path / 'date'
    cal.aggregate_comparison_outputs(output, directories)
    status = json.loads((output / 'calibration/status.json').read_text())
    assert all(row['status'] == 'ok' for row in status.values())
    assert len(pd.read_csv(output / 'calibration/none/predictions.csv')) == 4
    assert pd.read_csv(output / 'calibration/class_weight/bets.csv').empty


def test_incomplete_comparison_does_not_look_like_full_day(tmp_path):
    frame, odds = comparison_input(tmp_path)
    cal.save_comparison_outputs(frame, tmp_path / 'race1', odds)
    cal.aggregate_comparison_outputs(tmp_path / 'date', [tmp_path / 'race1', tmp_path / 'missing'])
    status = json.loads((tmp_path / 'date/calibration/status.json').read_text())
    assert status['none']['status'] == 'incomplete'
    assert not (tmp_path / 'date/calibration/none/predictions.csv').exists()
