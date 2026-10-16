"""確率補正の互換性、来歴、誤設定の拒否と予想経路の接続を検証する。"""
from importlib import import_module
import json

import numpy as np
import pandas as pd
import pytest

cal = import_module('src.40_ai_modeling.common.calibration')
service = import_module('src.40_ai_modeling.calibration_service')


def test_none_preserves_endpoints_and_does_not_mutate():
    raw = np.array([0., .2, 1.])
    result = cal.ProbabilityCalibration('m').apply(raw)
    np.testing.assert_array_equal(result, raw)
    result[0] = .5
    assert raw[0] == 0


def test_weight_correction_recovers_known_event_frequency():
    # 真の確率20%に正例重み4を与えると、最適な重み付き出力は50%になる。
    correction = cal.ProbabilityCalibration('m', 'class_weight', {'pos_weight': 4.})
    assert correction.apply([.5])[0] == pytest.approx(.2)
    assert np.isfinite(correction.apply([0, 1])).all()


@pytest.mark.parametrize('method,params', [
    ('invalid', {}), ('class_weight', {'pos_weight': 0}),
    ('class_weight', {'pos_weight': float('nan')}),
    ('sigmoid', {'slope': -1., 'intercept': 0.}),
    ('sigmoid', {'slope': 1., 'intercept': float('inf')}),
    ('none', {'unexpected': 1}),
])
def test_invalid_definitions_are_rejected(method, params):
    with pytest.raises(ValueError):
        cal.ProbabilityCalibration('m', method, params)


@pytest.mark.parametrize('raw', [[float('nan')], [float('inf')], [-.1], [1.1]])
def test_none_also_rejects_invalid_raw_probabilities(raw):
    with pytest.raises(ValueError, match='Invalid raw probability'):
        cal.ProbabilityCalibration('m').apply(raw)


def test_definition_round_trip_and_tampering(tmp_path):
    definition = cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1.1, 'intercept': -1.2})
    path = tmp_path / 'calibration.json'
    definition.save(path)
    loaded = cal.ProbabilityCalibration.load(path)
    assert loaded.calibration_id == definition.calibration_id
    np.testing.assert_array_equal(loaded.apply([.2, .5]), definition.apply([.2, .5]))
    payload = json.loads(path.read_text())
    payload['parameters']['intercept'] = -2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='content ID'):
        cal.ProbabilityCalibration.load(path)


def test_legacy_defaults_to_none_but_requested_correction_fails(tmp_path):
    metrics = {'model_run_id': 'm'}
    assert cal.load_calibration(tmp_path, metrics).method == 'none'
    with pytest.raises(ValueError, match='unavailable'):
        cal.load_calibration(tmp_path, metrics, 'class_weight')
    with pytest.raises(FileNotFoundError):
        cal.load_calibration(tmp_path, {**metrics, 'calibration_file': 'calibration.json'})


@pytest.mark.parametrize('field,value', [('model_run_id', 'other'), ('task_name', 'win'), ('target_column', 'other')])
def test_identity_mismatch_rejected_even_in_none_mode(tmp_path, field, value):
    cal.ProbabilityCalibration('m').save(tmp_path / 'calibration.json')
    metrics = {'model_run_id': 'm', field: value}
    with pytest.raises(ValueError, match='mismatch'):
        cal.load_calibration(tmp_path, metrics)


def test_explicit_method_and_trace_columns(tmp_path):
    cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -1.}).save(tmp_path / 'calibration.json')
    metrics = {'model_run_id': 'm', 'pos_weight': 4}
    frame = pd.DataFrame({'race_id': ['r'], 'horse_id': ['h']})
    default = cal.attach_probabilities(frame, [.5], tmp_path, metrics)
    corrected = cal.attach_probabilities(frame, [.5], tmp_path, metrics, 'class_weight')
    learned = cal.attach_probabilities(frame, [.5], tmp_path, metrics, 'sigmoid')
    assert default.top3_probability.iloc[0] == .5
    assert corrected.top3_probability.iloc[0] == pytest.approx(.2)
    assert learned.top3_probability.iloc[0] == pytest.approx(1 / (1 + np.e))
    assert learned.raw_top3_probability.iloc[0] == .5
    assert learned.calibration_method.iloc[0] == 'sigmoid'
    assert 'top3_probability' not in frame


def calibration_data():
    return pd.DataFrame({'race_id': [f'r{i}' for i in range(8)], 'horse_id': ['h'] * 8,
                         'race_date': pd.date_range('2026-02-01', periods=8),
                         'model_run_id': ['m'] * 8,
                         'raw_top3_probability': [.1, .2, .3, .4, .5, .6, .7, .8],
                         'actual_top3': [0, 0, 0, 1, 0, 1, 1, 1]})


def write_metrics(tmp_path):
    metrics = {'model_run_id': 'm', 'pos_weight': 4,
               'date_ranges': {'train': ['2025-01-01', '2025-12-01'],
                               'validation': ['2025-12-02', '2026-01-31']}}
    (tmp_path / 'metrics.json').write_text(json.dumps(metrics))


def test_calibration_training_records_period_and_held_out_scope(tmp_path):
    write_metrics(tmp_path)
    result = service.create_calibration(tmp_path, 'sigmoid', predictions=calibration_data(), evaluation_start='2026-03-01')
    assert result.parameters['slope'] > 0
    assert result.provenance['rows'] == 8
    assert result.provenance['end_date'] == '2026-02-08'
    assert result.provenance['metrics_scope'].startswith('calibration_fit_only')


@pytest.mark.parametrize('change', ['evaluation_overlap', 'training_overlap', 'wrong_model', 'duplicate', 'bad_label'])
def test_calibration_training_rejects_leakage_and_invalid_data(tmp_path, change):
    write_metrics(tmp_path)
    data = calibration_data()
    boundary = '2026-03-01'
    if change == 'evaluation_overlap': boundary = '2026-02-08'
    if change == 'training_overlap': data.loc[0, 'race_date'] = pd.Timestamp('2026-01-31')
    if change == 'wrong_model': data.loc[0, 'model_run_id'] = 'other'
    if change == 'duplicate': data = pd.concat([data, data.iloc[:1]])
    if change == 'bad_label': data.loc[0, 'actual_top3'] = 2
    with pytest.raises(ValueError):
        service.create_calibration(tmp_path, 'sigmoid', predictions=data, evaluation_start=boundary)


def test_prediction_period_and_double_application_rejected(tmp_path):
    definition = cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -1.},
                                             {'end_date': '2026-02-28'})
    definition.save(tmp_path / 'calibration.json')
    metrics = {'model_run_id': 'm'}
    frame = pd.DataFrame({'race_date': ['2026-02-28']})
    with pytest.raises(ValueError, match='overlaps'):
        cal.attach_probabilities(frame, [.5], tmp_path, metrics, 'sigmoid')
    corrected = cal.attach_probabilities(frame.assign(race_date='2026-03-01'), [.5], tmp_path, metrics, 'sigmoid')
    with pytest.raises(ValueError, match='double'):
        cal.attach_probabilities(corrected, [.5], tmp_path, metrics, 'sigmoid')


@pytest.mark.parametrize('method,expected', [('none', .5), ('class_weight', .2), ('sigmoid', 1 / (1 + np.e))])
def test_all_prediction_routes_share_correction_and_saved_columns(tmp_path, monkeypatch, method, expected):
    """特徴量とlogitだけ固定し、実際の予想・順位付け・ファイル保存を通して検証する。"""
    from types import SimpleNamespace
    import torch
    module = import_module('src.40_ai_modeling.tasks.top3.model')
    metrics = {'model_run_id': 'm', 'pos_weight': 4}
    cal.ProbabilityCalibration('m', 'sigmoid', {'slope': 1., 'intercept': -1.}).save(tmp_path / 'calibration.json')
    class Preprocessor:
        def transform(self, frame):
            return np.zeros((len(frame), 1), dtype='float32')
    monkeypatch.setattr(module, 'load_model_bundle', lambda path: (lambda x: torch.zeros(len(x)), Preprocessor(), metrics))
    monkeypatch.setattr(module, '_prepare_saved_model_features', lambda frame, metrics: frame)
    monkeypatch.setattr(module, 'build_top3_prediction_features', lambda history, target, **kw: target.copy())
    monkeypatch.setattr(module, 'PATHS', SimpleNamespace(predictions=tmp_path / 'predictions'))
    monkeypatch.setattr(module, 'save_prediction_run', lambda record: None)
    monkeypatch.setattr(module._storage, 'load_race_odds', lambda *args, **kw: pd.DataFrame())
    row = {'race_id': 'r', 'race_date': '2026-03-01', 'horse_id': 'h', 'horse_number': 1,
           'frame_number': 1, 'horse_name': 'horse', 'jockey_name': 'jockey', 'course_name': '東京',
           'race_number': 1, 'race_name': 'race', 'odds': 3., 'popularity': 1, 'finish_position': 1}
    upcoming = pd.DataFrame([row])
    history = pd.DataFrame([{**row, 'race_id': 'old', 'race_date': '2026-02-01'}])
    inferred, _ = module.infer_race(history, upcoming, 'r', tmp_path, calibration_method=method)
    single, single_path = module.predict_race(history, upcoming, 'r', tmp_path, calibration_method=method)
    daily, daily_path = module.predict_race_date(history, upcoming, '2026-03-01', tmp_path, calibration_method=method)
    historical, hist_path, _ = module.predict_historical_race(pd.concat([history, upcoming]), 'r', tmp_path, calibration_method=method)
    for frame in [inferred, single, daily, historical,
                  pd.read_parquet(single_path), pd.read_parquet(daily_path), pd.read_parquet(hist_path)]:
        assert frame.raw_top3_probability.iloc[0] == .5
        assert frame.top3_probability.iloc[0] == pytest.approx(expected)
        assert frame.calibration_method.iloc[0] == method
        assert frame.expected_value_index.iloc[0] == pytest.approx(expected * 3)


def test_daily_and_date_workflows_forward_explicit_calibration(tmp_path, monkeypatch):
    from datetime import date
    daily = import_module('src.10_workflows.daily_racing')
    by_date = import_module('src.10_workflows.date_prediction')
    monkeypatch.setattr(daily, 'check_meeting_day', lambda *args: True)
    monkeypatch.setattr(by_date.DateCollector, 'race_ids_for_date', lambda *args, **kw: ['202605040611'])
    calls = []
    def predict(bundle, output, race_date, race_id, *, calibration_method='none'):
        output.mkdir(parents=True, exist_ok=True)
        calls.append(calibration_method)
        return {'status': 'incomplete', 'prediction_status': 'skipped'}
    monkeypatch.setattr(by_date, 'run_prediction', predict)
    result = daily.run_daily(date(2026, 10, 18), tmp_path, tmp_path, tmp_path / 'run', False,
                             calibration_method='class_weight')
    assert calls == ['class_weight']
    assert result['calibration_method'] == 'class_weight'
    saved = json.loads((tmp_path / 'run/prediction/date_report.json').read_text(encoding='utf-8'))
    assert saved['calibration_method'] == 'class_weight'
