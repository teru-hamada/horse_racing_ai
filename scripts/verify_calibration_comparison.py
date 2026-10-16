"""配布バンドルを一時展開し、保存済み一レースで3方式の比較保存を確認する。"""
from importlib import import_module
import json
from pathlib import Path
import sys
import tempfile

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    bundle = import_module('src.40_ai_modeling.prediction_bundle')
    model = import_module('src.40_ai_modeling.tasks.top3.model')
    calibration = import_module('src.40_ai_modeling.common.calibration')
    stored = pd.read_csv(ROOT / 'docs/predictions/2026-09-21.csv', dtype={'race_id': str, 'horse_id': str})
    race_id = stored.race_id.iloc[0]
    card = stored[stored.race_id.eq(race_id)].copy()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory) / 'runtime'
        manifest = bundle.unpack_bundle(ROOT / 'prediction_bundle', work)
        with duckdb.connect(str(work / 'racing.duckdb'), read_only=True) as con:
            history = con.execute('SELECT * FROM race_records WHERE race_date < ?', [card.race_date.iloc[0]]).df()
        with duckdb.connect(str(ROOT / 'data/racing.duckdb'), read_only=True) as con:
            odds = con.execute('SELECT * FROM race_odds WHERE race_id=?', [race_id]).df()
        predictions, _ = model.infer_race(history, card, race_id, work / 'model',
                                          feature_database=work / 'features.duckdb')
        assert predictions.calibration_method.eq('none').all()
        assert all(predictions[f'calibration_{method}_status'].eq('ok').all() for method in calibration.METHODS)
        np.testing.assert_array_equal(predictions.top3_probability, predictions.raw_top3_probability)
        output = ROOT / 'data/analysis/calibration_comparison_smoke'
        status = calibration.save_comparison_outputs(predictions, output, odds)
        assert all(status[m]['status'] == 'ok' for m in calibration.METHODS)
        summary = {'bundle_format': manifest['format'], 'race_id': race_id, 'horses': len(predictions),
                   'public_method': 'none', 'methods': status}
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
