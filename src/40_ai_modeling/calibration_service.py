"""日付を持つ学習外予測から補正定義を作成するCLI。元モデルは変更しない。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .common.calibration import ProbabilityCalibration, read_calibrations, save_calibrations


def create_calibration(model_dir: Path, method: str, *, predictions=None, evaluation_start=None):
    """校正期間が学習後・評価前であることを確認する。予測元は同じモデルに限定。"""
    metrics = json.loads((Path(model_dir) / 'metrics.json').read_text(encoding='utf-8'))
    if metrics.get('task_name', 'top3') != 'top3' or metrics.get('target_column', 'target_top3') != 'target_top3':
        raise ValueError('Calibration service requires top3/target_top3 model')
    model_id = metrics['model_run_id']
    if method == 'none':
        return ProbabilityCalibration(model_id)
    if method == 'class_weight':
        if 'pos_weight' not in metrics:
            raise ValueError('Model has no saved pos_weight; do not guess from current DB')
        return ProbabilityCalibration(model_id, method, {'pos_weight': metrics['pos_weight']})
    if method != 'sigmoid' or predictions is None or evaluation_start is None:
        raise ValueError('sigmoid requires predictions and evaluation_start')
    data = predictions.copy()
    required = {'race_id', 'horse_id', 'race_date', 'model_run_id', 'raw_top3_probability', 'actual_top3'}
    if not required <= set(data):
        raise ValueError(f'Calibration data missing columns: {sorted(required - set(data))}')
    if data.empty or data.duplicated(['race_id', 'horse_id']).any() or data[list(required)].isna().any().any():
        raise ValueError('Calibration data is empty, missing values or has duplicate keys')
    if not data.model_run_id.eq(model_id).all():
        raise ValueError('Calibration prediction model mismatch')
    dates = pd.to_datetime(data.race_date, errors='raise').dt.normalize()
    boundary = pd.Timestamp(evaluation_start).normalize()
    # 検証損失によるモデル選択に使用した期間も除き、学習外の予測を校正へ使う。
    ranges = metrics['date_ranges']
    model_end = max(pd.Timestamp(ranges[k][1]).normalize() for k in ('train', 'validation'))
    if dates.isna().any() or pd.isna(boundary) or dates.min() <= model_end or dates.max() >= boundary:
        raise ValueError('Calibration period overlaps model training/validation or evaluation')
    y = pd.to_numeric(data.actual_top3, errors='raise').to_numpy()
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Calibration labels must include both binary classes')
    p = data.raw_top3_probability.to_numpy(dtype=float)
    ProbabilityCalibration(model_id).apply(p)  # 校正なしでも確率の範囲を検査する。
    x = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(x, y)
    provenance = {'rows': len(data), 'start_date': str(dates.min().date()),
                  'end_date': str(dates.max().date()), 'evaluation_start': str(boundary.date()),
                  'created_at': datetime.now(timezone.utc).isoformat(),
                  'metrics_scope': 'calibration_fit_only_not_independent_evaluation',
                  'data_sha256': hashlib.sha256(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()).hexdigest()}
    calibrated = model.predict_proba(x)[:, 1]
    provenance['fit_metrics'] = {'log_loss': float(log_loss(y, calibrated)),
                                 'brier': float(brier_score_loss(y, calibrated))}
    return ProbabilityCalibration(model_id, method,
                                  {'slope': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0])}, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--method', choices=['none', 'class_weight', 'sigmoid'], required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--evaluation-start')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-existing', type=Path, help='既存の補正定義を新しい出力へ引き継ぐ')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('補正定義の出力先には新しいファイルを指定してください。')
    data = pd.read_parquet(args.predictions) if args.predictions and args.predictions.suffix == '.parquet' else (
        pd.read_csv(args.predictions, dtype={'race_id': str, 'horse_id': str}) if args.predictions else None)
    calibration = create_calibration(args.model_dir, args.method, predictions=data, evaluation_start=args.evaluation_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    definitions = read_calibrations(args.include_existing) if args.include_existing else {}
    definitions[calibration.method] = calibration
    save_calibrations(args.output, definitions.values())
    print(calibration.calibration_id)


if __name__ == '__main__':
    main()
