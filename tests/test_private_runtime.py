"""Private runtime must fail closed on missing files, corruption, or unsafe ZIPs."""
import json
import zipfile

import pytest

from scripts import private_runtime as runtime


def make_snapshot(root):
    bundle = root / 'prediction_bundle'
    bundle.mkdir(parents=True)
    (root / 'data').mkdir()
    for name in ('racing.duckdb', 'features.duckdb'):
        (root / 'data' / name).write_bytes(b'fixture database')
    (bundle / 'requirements.txt').write_bytes(b'numpy==1.0\n')
    (bundle / '.python-version').write_text('3.12\n')
    with zipfile.ZipFile(bundle / 'runtime.zip', 'w') as archive:
        archive.writestr('model/model.pt', b'fixture model')
    manifest = {
        'python_version': '3.12', 'files': {'model/model.pt': 'unused'},
        'archive_sha256': runtime.digest(bundle / 'runtime.zip'),
        'requirements_sha256': runtime.digest(bundle / 'requirements.txt'),
    }
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return bundle, manifest


def test_copy_only_runtime_files_and_preserve_bytes(tmp_path):
    source = tmp_path / 'source'
    bundle, _ = make_snapshot(source)
    (source / 'secret.txt').write_text('must not copy')
    destination = tmp_path / 'destination'
    runtime.copy_runtime(source, destination)
    assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()} == set(runtime.FILES)
    for name in runtime.FILES:
        assert (source / name).read_bytes() == (destination / name).read_bytes()


@pytest.mark.parametrize('corruption', ['missing_database', 'zip', 'requirements', 'python_version', 'traversal'])
def test_invalid_snapshot_is_rejected_before_destination_is_written(tmp_path, corruption):
    source = tmp_path / 'source'
    bundle, manifest = make_snapshot(source)
    if corruption == 'missing_database':
        (source / 'data/racing.duckdb').unlink()
    elif corruption == 'zip':
        (bundle / 'runtime.zip').write_bytes(b'corrupt')
    elif corruption == 'requirements':
        (bundle / 'requirements.txt').write_text('unexpected-package')
    elif corruption == 'python_version':
        (bundle / '.python-version').write_text('3.11')
    else:
        with zipfile.ZipFile(bundle / 'runtime.zip', 'w') as archive:
            archive.writestr('../outside', b'unsafe')
        manifest['files'] = {'../outside': 'unused'}
        manifest['archive_sha256'] = runtime.digest(bundle / 'runtime.zip')
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
    destination = tmp_path / 'destination'
    with pytest.raises(ValueError):
        runtime.copy_runtime(source, destination)
    assert not destination.exists()
