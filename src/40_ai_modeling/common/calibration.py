"""モデルに結び付いた確率補正。予想中に学習せず、不整合時は停止する。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

METHODS = ('none', 'class_weight', 'sigmoid')
EPSILON = 1e-7


class CalibrationUnavailable(ValueError):
    """定義が未作成。破損・ID不一致とは分け、比較用の未計算として記録できる。"""


@dataclass(frozen=True)
class ProbabilityCalibration:
    """一つのモデル・目的変数専用の補正定義。係数と来歴をJSONで保持する。"""

    model_run_id: str
    method: str = 'none'
    parameters: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    task_name: str = 'top3'
    target_column: str = 'target_top3'
    format_version: int = 1

    def __post_init__(self):
        if type(self.format_version) is not int or self.format_version != 1 or self.method not in METHODS:
            raise ValueError('Unsupported calibration format or method')
        if not isinstance(self.model_run_id, str) or not self.model_run_id:
            raise ValueError('Calibration requires model_run_id')
        if not isinstance(self.task_name, str) or not self.task_name or not isinstance(self.target_column, str) or not self.target_column:
            raise ValueError('Calibration requires task and target')
        if not isinstance(self.parameters, dict) or not isinstance(self.provenance, dict):
            raise ValueError('Invalid calibration metadata')
        required = {'none': set(), 'class_weight': {'pos_weight'},
                    'sigmoid': {'slope', 'intercept'}}[self.method]
        if set(self.parameters) != required:
            raise ValueError('Invalid calibration parameters')
        for value in self.parameters.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Calibration parameters must be finite numbers')
        if self.method == 'class_weight' and self.parameters['pos_weight'] <= 0:
            raise ValueError('pos_weight must be positive')
        if self.method == 'sigmoid' and self.parameters['slope'] <= 0:
            raise ValueError('Calibration slope must be positive to preserve ranking')

    @property
    def calibration_id(self) -> str:
        """内容が同じなら同じID。係数・モデル・来歴の変更を追跡できる。"""
        import hashlib
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, allow_nan=False)
        return 'cal_' + hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]

    def validate_identity(self, model_run_id, task_name='top3', target_column='target_top3'):
        if (self.model_run_id, self.task_name, self.target_column) != (model_run_id, task_name, target_column):
            raise ValueError('Calibration model/task/target mismatch')

    def apply(self, probabilities):
        """生確率だけを受ける。既に補正した値へ二重適用しない。"""
        values = np.asarray(probabilities, dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('Invalid raw probability: expected finite values in [0, 1]')
        if self.method == 'none':
            return values.copy()  # 端点も変えず、旧方式との完全な比較基準にする。
        logits = logit(np.clip(values, EPSILON, 1 - EPSILON))
        if self.method == 'class_weight':
            return expit(logits - np.log(self.parameters['pos_weight']))
        return expit(self.parameters['slope'] * logits + self.parameters['intercept'])

    def save(self, path: Path):
        path = Path(path)
        payload = {**asdict(self), 'calibration_id': self.calibration_id}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('Invalid calibration definition')
        identity = payload.pop('calibration_id', None)
        try:
            result = cls(**payload)
        except TypeError as exc:
            raise ValueError('Invalid calibration definition') from exc
        if identity != result.calibration_id:
            raise ValueError('Calibration content ID mismatch')
        return result


def save_calibrations(path: Path, definitions):
    """複数方式を同じファイルへ保存する。方式を上書きして他方式を失わない。"""
    definitions = list(definitions)
    if not definitions or len({d.method for d in definitions}) != len(definitions):
        raise ValueError('Calibration collection requires unique methods')
    first = definitions[0]
    for definition in definitions:
        definition.validate_identity(first.model_run_id, first.task_name, first.target_column)
    payload = {'format_version': 2, 'calibrations': {
        d.method: {**asdict(d), 'calibration_id': d.calibration_id} for d in definitions}}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_calibrations(path: Path):
    """旧単一定義と新しい複数定義の両方を検証して読み込む。"""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Invalid calibration definition')
    if payload.get('format_version') != 2:
        definition = ProbabilityCalibration.load(path)
        return {definition.method: definition}
    if set(payload) != {'format_version', 'calibrations'} or not isinstance(payload['calibrations'], dict) or not payload['calibrations']:
        raise ValueError('Invalid calibration collection')
    definitions = {}
    for method, item in payload['calibrations'].items():
        item = dict(item)
        identity = item.pop('calibration_id', None)
        definition = ProbabilityCalibration(**item)
        if method != definition.method or identity != definition.calibration_id:
            raise ValueError('Calibration collection content ID mismatch')
        definitions[method] = definition
    return definitions


def load_calibration(model_dir: Path, metrics: dict, method: str = 'none') -> ProbabilityCalibration:
    """既定は常にnone。指定方式の不足をnoneへ黙ってフォールバックしない。"""
    if method not in METHODS:
        raise ValueError(f'Unknown calibration method: {method}')
    model_id = metrics['model_run_id']
    task = metrics.get('task_name', 'top3')
    target = metrics.get('target_column', 'target_top3')
    if metrics.get('calibration_file') not in (None, 'calibration.json'):
        raise ValueError('Unsupported model calibration file')
    path = Path(model_dir) / 'calibration.json'
    definitions = read_calibrations(path) if path.exists() else {}
    if metrics.get('calibration_file') and not definitions:
        raise FileNotFoundError('Required calibration.json is missing')
    for definition in definitions.values():
        definition.validate_identity(model_id, task, target)
    if method == 'none':
        return ProbabilityCalibration(model_id, task_name=task, target_column=target)
    if method in definitions:
        return definitions[method]
    if method == 'class_weight' and 'pos_weight' in metrics:
        return ProbabilityCalibration(model_id, method, {'pos_weight': metrics['pos_weight']},
                                      task_name=task, target_column=target)
    raise CalibrationUnavailable(f'Calibration {method} is unavailable for model {model_id}')


def validate_prediction_period(frame, calibration):
    """校正学習期間以前への適用を防止する。並行保存でも同じ制約を使う。"""
    if calibration.method == 'sigmoid' and calibration.provenance.get('end_date'):
        import pandas as pd
        dates = pd.to_datetime(frame['race_date'], errors='raise').dt.normalize()
        if dates.isna().any() or (dates <= pd.Timestamp(calibration.provenance['end_date'])).any():
            raise ValueError('Prediction period overlaps calibration fit period')
        if calibration.provenance.get('evaluation_start') and (
            dates < pd.Timestamp(calibration.provenance['evaluation_start'])
        ).any():
            raise ValueError('Prediction precedes calibration evaluation start')


def attach_probabilities(frame, raw_probabilities, model_dir, metrics, method='none'):
    """全予想経路の共通入口。生確率と補正方式も行単位で保存する。"""
    calibration = load_calibration(model_dir, metrics, method)
    if {'raw_top3_probability', 'calibration_method', 'calibration_id'} & set(frame.columns):
        raise ValueError('Probabilities already attached; refusing double calibration')
    validate_prediction_period(frame, calibration)
    result = frame.copy()
    result['raw_top3_probability'] = np.asarray(raw_probabilities, dtype=float)
    result['top3_probability'] = calibration.apply(raw_probabilities)
    result['calibration_method'] = calibration.method
    result['calibration_id'] = calibration.calibration_id
    # 公開用方式と比較用方式を分離する。同じ生出力から一回ずつ補正し、
    # 生→重み補正→sigmoidという連鎖や、未作成方式の生確率への代替は行わない。
    for candidate in METHODS:
        prefix = f'calibration_{candidate}'
        try:
            definition = load_calibration(model_dir, metrics, candidate)
        except CalibrationUnavailable as exc:
            result[prefix + '_probability'] = np.nan
            result[prefix + '_status'] = 'unavailable'
            result[prefix + '_id'] = ''
            result[prefix + '_reason'] = str(exc)
            continue
        try:
            validate_prediction_period(frame, definition)
        except ValueError as exc:
            if candidate == method:
                raise
            result[prefix + '_probability'] = np.nan
            result[prefix + '_status'] = 'not_applicable'
            result[prefix + '_id'] = definition.calibration_id
            result[prefix + '_reason'] = str(exc)
            continue
        result[prefix + '_probability'] = definition.apply(raw_probabilities)
        result[prefix + '_status'] = 'ok'
        result[prefix + '_id'] = definition.calibration_id
        result[prefix + '_reason'] = ''
    return result


def comparison_predictions(predictions):
    """横持ち確率を方式別の予想表に展開する。公開方式は変更しない。"""
    import pandas as pd
    for method in METHODS:
        prefix = f'calibration_{method}'
        if prefix + '_status' not in predictions or not predictions[prefix + '_status'].eq('ok').all():
            continue
        frame = predictions.copy()
        frame['top3_probability'] = frame[prefix + '_probability']
        frame['calibration_method'] = method
        frame['calibration_id'] = frame[prefix + '_id']
        frame['expected_value_index'] = frame.top3_probability * pd.to_numeric(frame.odds, errors='coerce')
        frame['prediction_rank'] = frame.groupby('race_id', sort=False).top3_probability.rank(method='first', ascending=False).astype(int)
        if 'predicted_top3' in frame:
            frame['predicted_top3'] = frame.prediction_rank.le(3)
            frame['top3_hit'] = frame.predicted_top3 & frame.actual_top3
        yield method, frame.sort_values(['race_id', 'prediction_rank'], kind='stable').reset_index(drop=True)


def save_comparison_outputs(predictions, output: Path, odds=None):
    """方式別の予想と買い目を同じ取得オッズで保存する。追加通信はしない。"""
    from importlib import import_module
    betting = import_module('src.40_ai_modeling.betting')
    status = {}
    for method in METHODS:
        prefix = f'calibration_{method}'
        if prefix + '_status' in predictions:
            status[method] = {'status': predictions[prefix + '_status'].iloc[0],
                              'calibration_id': predictions[prefix + '_id'].iloc[0],
                              'reason': predictions[prefix + '_reason'].iloc[0]}
    for method, frame in comparison_predictions(predictions):
        directory = Path(output) / 'calibration' / method
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / 'predictions.csv', index=False, encoding='utf-8-sig')
        frame.to_parquet(directory / 'predictions.parquet', index=False)
        if odds is not None:
            # 候補を再計算するので、公開方式から除外された買い目も方式ごとに評価できる。
            bets = betting.calculate_bet_recommendations(frame, odds, best_only=False)
            bets['calibration_method'] = method
            bets['calibration_id'] = frame.calibration_id.iloc[0]
            bets.to_csv(directory / 'bets.csv', index=False, encoding='utf-8-sig')
            selected = bets.sort_values(['recommendation_score', 'recovery_rate_percent'], ascending=False,
                                        kind='stable').groupby('race_id', sort=False).head(3)
            selected.to_csv(directory / 'recommended_bets.csv', index=False, encoding='utf-8-sig')
            status[method]['bet_rows'] = len(bets)
            status[method]['recommended_rows'] = len(selected)
            status[method]['odds_status'] = 'ok' if not odds.empty else 'missing'
        else:
            status[method]['betting_status'] = 'odds_not_supplied'
    root = Path(output) / 'calibration'
    root.mkdir(parents=True, exist_ok=True)
    if odds is not None:
        odds.to_csv(root / 'odds.csv', index=False, encoding='utf-8-sig')
    (root / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


def aggregate_comparison_outputs(output: Path, race_directories):
    """一日の成功レースの比較成果物を集約。方式の一部欠落をゼロ買いと混同しない。"""
    import pandas as pd
    directories = list(race_directories)
    if not directories or not any((d / 'calibration/status.json').exists() for d in directories):
        return  # 旧成果物や比較出力のないテスト入力にも対応する。
    status = {}
    for method in METHODS:
        sources = [d / 'calibration' / method for d in directories]
        complete = all((d / 'predictions.csv').exists() and (d / 'bets.csv').exists() for d in sources)
        status[method] = {'status': 'ok' if complete else 'incomplete',
                          'races_available': sum((d / 'predictions.csv').exists() for d in sources),
                          'races_expected': len(directories)}
        if not complete:
            continue
        target = Path(output) / 'calibration' / method
        target.mkdir(parents=True, exist_ok=True)
        for name in ('predictions.csv', 'bets.csv', 'recommended_bets.csv'):
            frames = [pd.read_csv(d / name, dtype={'race_id': str, 'horse_id': str, 'selection': str}) for d in sources]
            pd.concat(frames, ignore_index=True).to_csv(target / name, index=False, encoding='utf-8-sig')
    target = Path(output) / 'calibration'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
