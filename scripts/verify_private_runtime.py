"""Verify restored databases and run the actual saved model without external race data."""
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(root=ROOT):
    import duckdb
    import torch

    bundle = import_module('src.40_ai_modeling.prediction_bundle')
    loader = import_module('src.40_ai_modeling.tasks.top3.model')
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory) / 'runtime'
        manifest = bundle.unpack_bundle(root / 'prediction_bundle', work)
        if manifest['python_version'] != f'{sys.version_info.major}.{sys.version_info.minor}':
            raise ValueError('Python version mismatch')
        for package, expected in manifest['versions'].items():
            if version(package).split('+')[0] != expected.split('+')[0]:
                raise ValueError(f'Package version mismatch: {package}')
        with duckdb.connect(str(root / 'data/racing.duckdb'), read_only=True) as con:
            if bundle.latest_model(con)['model_id'] != manifest['model_id']:
                raise ValueError('Registered model differs from private bundle')
        with duckdb.connect(str(root / 'data/features.duckdb'), read_only=True) as con:
            con.execute('SELECT count(*) FROM performance_features').fetchone()
        model, preprocessor, metrics = loader.load_model_bundle(work / 'model')
        if metrics['model_run_id'] != manifest['model_id']:
            raise ValueError('Loaded model ID mismatch')
        first_layer = next(layer for layer in model.modules() if isinstance(layer, torch.nn.Linear))
        with torch.inference_mode():
            result = model(torch.zeros((1, first_layer.in_features)))
        if not torch.isfinite(result).all():
            raise ValueError('CPU model inference returned a non-finite value')
        if not hasattr(preprocessor, 'transform'):
            raise ValueError('Invalid saved preprocessor')
    print('Private runtime verified: hashes, databases, package versions, model and CPU forward pass.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    main(parser.parse_args().root.resolve())
