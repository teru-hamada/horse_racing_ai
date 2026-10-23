"""Compatibility, rollback, crash recovery and shared writer locking."""
from importlib import import_module
import json
from pathlib import Path
import threading
import zipfile

import pytest

from scripts import private_runtime as legacy, private_snapshot as split, runtime_transfer as transfer

locking = import_module('src.00_common.runtime_lock')


def make_source(root, payload=b'database'):
    (root / 'data').mkdir(parents=True)
    bundle = root / 'prediction_bundle'
    bundle.mkdir()
    for name in ('racing.duckdb', 'features.duckdb'):
        (root / 'data' / name).write_bytes(payload)
    (bundle / 'requirements.txt').write_text('numpy==1.0\n')
    (bundle / '.python-version').write_text('3.12\n')
    with zipfile.ZipFile(bundle / 'runtime.zip', 'w') as archive:
        archive.writestr('model/model.pt', b'model')
    manifest = {'python_version': '3.12', 'files': {'model/model.pt': 'unused'},
                'archive_sha256': legacy.digest(bundle / 'runtime.zip'),
                'requirements_sha256': legacy.digest(bundle / 'requirements.txt')}
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize('packed', [False, True])
def test_activation_accepts_both_formats_and_keeps_previous_backup(tmp_path, packed):
    source = make_source(tmp_path / 'source', b'new')
    target = make_source(tmp_path / 'target', b'old')
    if packed:
        split.prepare(source, tmp_path / 'packed')
        source = tmp_path / 'packed'
    result = transfer.activate(source, target, expected_hashes=transfer.hashes(target))
    assert (target / 'data/racing.duckdb').read_bytes() == b'new'
    assert (Path(result['backup']) / 'data/racing.duckdb').read_bytes() == b'old'
    assert not (target / '.private-runtime/activation-pending.json').exists()


def test_write_failure_rolls_back_all_live_files(tmp_path, monkeypatch):
    source = make_source(tmp_path / 'source', b'new')
    target = make_source(tmp_path / 'target', b'old')
    before = transfer.hashes(target)
    original = Path.replace

    def fail_on_database(path, destination):
        if Path(destination) == target / 'data/racing.duckdb':
            raise OSError('injected disk error')
        return original(path, destination)

    monkeypatch.setattr(Path, 'replace', fail_on_database)
    with pytest.raises(OSError, match='injected'):
        transfer.activate(source, target)
    assert transfer.hashes(target) == before
    assert not (target / '.private-runtime/activation-pending.json').exists()


def test_stale_local_fingerprint_rejects_activation(tmp_path):
    source = make_source(tmp_path / 'source', b'new')
    target = make_source(tmp_path / 'target', b'old')
    expected = transfer.hashes(target)
    (target / 'data/racing.duckdb').write_bytes(b'local update')
    with pytest.raises(ValueError, match='変更'):
        transfer.activate(source, target, expected_hashes=expected)
    assert (target / 'data/racing.duckdb').read_bytes() == b'local update'


@pytest.mark.parametrize('packed', [False, True])
def test_metadata_only_supports_new_and_old_packed_snapshots(tmp_path, packed):
    source = make_source(tmp_path / 'source')
    if packed:
        split.prepare(source, tmp_path / 'packed')
        source = tmp_path / 'packed'
        # Older split snapshots did not have standalone metadata copies.
        for name in transfer.METADATA:
            (source / name).unlink()
    output = tmp_path / 'audit'
    transfer.metadata(source, output)
    assert (output / 'prediction_bundle/.python-version').read_text().strip() == '3.12'
    assert not (output / 'data').exists()


def test_crash_marker_blocks_writes_until_recovery(tmp_path):
    root = make_source(tmp_path / 'target', b'old')
    backup_id = 'a' * 32
    backup = root / '.private-runtime/backups' / backup_id
    (backup / 'data').mkdir(parents=True)
    (backup / 'data/racing.duckdb').write_bytes(b'old')
    (backup / 'activation.json').write_text(json.dumps({'targets': ['data/racing.duckdb'],
                                                      'existed': ['data/racing.duckdb']}))
    (root / '.private-runtime/activation-pending.json').write_text(json.dumps({'backup': backup_id}))
    (root / 'data/racing.duckdb').write_bytes(b'interrupted update')
    with pytest.raises(RuntimeError, match='中断'):
        with locking.runtime_lock(root):
            pass
    transfer.recover(root)
    assert (root / 'data/racing.duckdb').read_bytes() == b'old'
    with locking.runtime_lock(root):
        pass


def test_lock_is_reentrant_but_excludes_other_threads(tmp_path):
    errors = []

    def compete():
        try:
            with locking.runtime_lock(tmp_path):
                errors.append('unexpected entry')
        except RuntimeError:
            errors.append('blocked')

    with locking.runtime_lock(tmp_path):
        with locking.runtime_lock(tmp_path):
            worker = threading.Thread(target=compete)
            worker.start()
            worker.join(timeout=5)
    assert errors == ['blocked']
    with locking.runtime_lock(tmp_path):
        pass


def test_database_connection_holds_sync_lock_until_close(tmp_path):
    import duckdb
    connection = locking.connect_locked(tmp_path, tmp_path / 'db.duckdb', duckdb.connect)
    outcomes = []

    def compete():
        try:
            with locking.runtime_lock(tmp_path):
                outcomes.append('entered')
        except RuntimeError:
            outcomes.append('blocked')

    worker = threading.Thread(target=compete)
    worker.start()
    worker.join(timeout=5)
    assert outcomes == ['blocked']
    connection.close()
    worker = threading.Thread(target=compete)
    worker.start()
    worker.join(timeout=5)
    assert outcomes == ['blocked', 'entered']


def test_lock_excludes_an_independent_process(tmp_path):
    import subprocess
    import sys
    code = (
        "from importlib import import_module; import sys; "
        "lock=import_module('src.00_common.runtime_lock').runtime_lock; "
        "\nwith lock(sys.argv[1]):\n print('locked',flush=True); sys.stdin.readline()\n"
    )
    child = subprocess.Popen([sys.executable, '-u', '-c', code, str(tmp_path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8')
    try:
        assert child.stdout.readline().strip() == 'locked'
        with pytest.raises(RuntimeError, match='別のプロセス'):
            with locking.runtime_lock(tmp_path):
                pass
    finally:
        child.communicate('\n', timeout=10)
    with locking.runtime_lock(tmp_path):
        pass


@pytest.mark.parametrize('mismatch', [False, True])
def test_model_artifacts_are_verified_and_installed_with_databases(tmp_path, mismatch):
    source = make_source(tmp_path / 'source', b'new')
    target = make_source(tmp_path / 'target', b'old')
    bundle = source / 'prediction_bundle'
    payload_id = 'wrong' if mismatch else 'fixture_model'
    contents = {'model/model.pt': b'weights', 'model/preprocessor.joblib': b'preprocessor',
                'model/metrics.json': json.dumps({'model_run_id': payload_id}).encode(),
                'model/manifest.json': json.dumps({'model_run_id': payload_id}).encode()}
    import hashlib
    with zipfile.ZipFile(bundle / 'runtime.zip', 'w') as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest.update(model_id='fixture_model', archive_sha256=legacy.digest(bundle / 'runtime.zip'),
                    files={name: hashlib.sha256(content).hexdigest() for name, content in contents.items()})
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    before = transfer.hashes(target)
    if mismatch:
        with pytest.raises(ValueError, match='model ID'):
            transfer.activate(source, target)
        assert transfer.hashes(target) == before
    else:
        transfer.activate(source, target)
        assert (target / 'models/top3/fixture_model/model.pt').read_bytes() == b'weights'
        assert (target / 'data/racing.duckdb').read_bytes() == b'new'
