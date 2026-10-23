"""Private snapshot transfers must preserve bytes and reject incomplete generations."""
import json
import os
from pathlib import Path
import zipfile

import pytest

from scripts import private_snapshot as snapshot
from scripts import private_runtime


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'data').mkdir(parents=True)
    bundle = root / 'prediction_bundle'
    bundle.mkdir()
    for name in ('racing.duckdb', 'features.duckdb'):
        (root / 'data' / name).write_bytes(os.urandom(8192))
    (bundle / 'requirements.txt').write_text('numpy==1.0\n')
    (bundle / '.python-version').write_text('3.12\n')
    with zipfile.ZipFile(bundle / 'runtime.zip', 'w') as archive:
        archive.writestr('model/model.pt', b'fixture model')
    manifest = {'python_version': '3.12', 'files': {'model/model.pt': 'fixture'},
                'archive_sha256': snapshot.digest(bundle / 'runtime.zip'),
                'requirements_sha256': snapshot.digest(bundle / 'requirements.txt')}
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return root


def test_multipart_round_trip_is_identical_and_only_copies_allowlisted_files(source, tmp_path):
    (source / 'secret.txt').write_text('never publish')
    packed = tmp_path / 'packed'
    manifest = snapshot.prepare(source, packed, part_bytes=4096)
    assert len(manifest['parts']) > 1
    assert all(item['bytes'] <= 4096 for item in manifest['parts'])
    restored = tmp_path / 'restored'
    snapshot.restore(packed, restored)
    assert {p.relative_to(restored).as_posix() for p in restored.rglob('*') if p.is_file()} == set(snapshot.FILES)
    assert all(snapshot.digest(source / name) == snapshot.digest(restored / name) for name in snapshot.FILES)


@pytest.mark.parametrize('damage', ['missing_part', 'corrupt_part', 'manifest_hash', 'unsafe_path', 'reordered_parts'])
def test_damaged_snapshot_never_publishes_restore_or_changes_previous(source, tmp_path, damage):
    packed = tmp_path / 'packed'
    manifest = snapshot.prepare(source, packed, part_bytes=4096)
    previous = tmp_path / 'previous'
    snapshot.restore(packed, previous)
    previous_hashes = {name: snapshot.digest(previous / name) for name in snapshot.FILES}
    part = packed / manifest['parts'][-1]['path']
    if damage == 'missing_part':
        part.unlink()
    elif damage == 'corrupt_part':
        block = bytearray(part.read_bytes())
        block[0] ^= 1
        part.write_bytes(block)
    elif damage == 'manifest_hash':
        manifest['files']['data/racing.duckdb']['sha256'] = '0' * 64
    elif damage == 'unsafe_path':
        manifest['parts'][0]['path'] = '../outside'
    else:
        manifest['parts'].reverse()
    (packed / snapshot.MANIFEST).write_text(json.dumps(manifest))
    destination = tmp_path / 'failed_restore'
    with pytest.raises(ValueError):
        snapshot.restore(packed, destination)
    assert not destination.exists()
    assert all(snapshot.digest(previous / name) == h for name, h in previous_hashes.items())


def test_existing_generation_is_never_overwritten(source, tmp_path):
    packed = tmp_path / 'packed'
    snapshot.prepare(source, packed)
    old_manifest = (packed / snapshot.MANIFEST).read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        snapshot.prepare(source, packed)
    assert (packed / snapshot.MANIFEST).read_bytes() == old_manifest
    with pytest.raises(ValueError, match='new directory'):
        snapshot.restore(packed, source)


def test_database_larger_than_git_limit_can_round_trip_without_large_git_objects(source, tmp_path):
    large = source / 'data/features.duckdb'
    with large.open('wb') as stream:
        stream.truncate(snapshot.GIT_LIMIT + 1)
    with pytest.raises(ValueError, match='size limit'):
        private_runtime.validate(source)
    packed = tmp_path / 'packed'
    snapshot.prepare(source, packed)
    assert all(p.stat().st_size < snapshot.GIT_LIMIT for p in packed.rglob('*') if p.is_file())
    restored = tmp_path / 'restored'
    snapshot.restore(packed, restored)
    assert (restored / 'data/features.duckdb').stat().st_size == snapshot.GIT_LIMIT + 1
    assert snapshot.digest(large) == snapshot.digest(restored / 'data/features.duckdb')


def test_open_database_wal_prevents_snapshot(source, tmp_path):
    Path(str(source / 'data/racing.duckdb') + '.wal').write_bytes(b'pending write')
    destination = tmp_path / 'packed'
    with pytest.raises(ValueError, match='database writer'):
        snapshot.prepare(source, destination)
    assert not destination.exists()


def test_source_change_during_pack_is_rejected(source, tmp_path, monkeypatch):
    original = zipfile.ZipFile.write

    def change_after_write(archive, filename, *args, **kwargs):
        result = original(archive, filename, *args, **kwargs)
        if Path(filename).name == 'racing.duckdb':
            Path(filename).write_bytes(b'concurrent update')
        return result

    monkeypatch.setattr(zipfile.ZipFile, 'write', change_after_write)
    destination = tmp_path / 'packed'
    with pytest.raises(ValueError, match='Source changed'):
        snapshot.prepare(source, destination)
    assert not destination.exists()
