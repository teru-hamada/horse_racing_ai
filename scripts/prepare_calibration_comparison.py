"""既存モデルの検証済み学習外予測から、3方式の比較定義を準備する。"""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_probability_calibration import read_inputs

cal = import_module('src.40_ai_modeling.common.calibration')
service = import_module('src.40_ai_modeling.calibration_service')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/racing.duckdb')
    parser.add_argument('--published', type=Path, default=ROOT / 'docs/predictions')
    parser.add_argument('--bundle-output', type=Path)
    args = parser.parse_args()
    destination = args.model_dir / 'calibration.json'
    if destination.exists():
        parser.error('既存の補正定義は上書きしません。内容を確認してください。')
    # 学習期間・行数と、保存済みテスト予測の全キー・正解の一致を検査する。
    metrics, old, recent, _, count, positives, weight = read_inputs(args.model_dir, args.database, args.published)
    input_frame = old[['race_id', 'horse_id', 'race_date', 'actual_top3']].copy()
    input_frame['raw_top3_probability'] = old.predicted_probability.to_numpy()
    input_frame['model_run_id'] = metrics['model_run_id']
    sigmoid = service.create_calibration(args.model_dir, 'sigmoid', predictions=input_frame,
                                         evaluation_start=recent.day.min())
    weighted = cal.ProbabilityCalibration(
        metrics['model_run_id'], 'class_weight', {'pos_weight': weight},
        {'weight_source': 'reconstructed_from_historical_labels_with_saved_period_and_matching_row_count',
         'train_rows': count, 'train_positives': positives,
         'train_period': metrics['date_ranges']['train'],
         'historical_prediction_labels_verified': len(old)})
    cal.save_calibrations(destination, [cal.ProbabilityCalibration(metrics['model_run_id']), weighted, sigmoid])
    print(f'Created {destination}; public/default method remains none')
    if args.bundle_output:
        manifest = import_module('src.40_ai_modeling.prediction_bundle').export_bundle(args.bundle_output)
        print(f'Bundle format {manifest["format"]}: {manifest["model_id"]}')


if __name__ == '__main__':
    main()
