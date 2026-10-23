"""Pack private runtime files into bounded Git objects and restore to a new directory.

This does not activate a runtime or overwrite the application's live databases.
Only the Python standard library is required, including on Actions before pip.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.private_runtime import FILES, digest, validate

GIT_LIMIT = 100 * 1024 * 1024
PART_BYTES = 64 * 1024 * 1024
MANIFEST = 'snapshot.json'


def inspect_manifest(source: Path) -> dict:
    path = source / MANIFEST
    if path.is_symlink() or not path.is_file() or path.stat().st_size >= GIT_LIMIT:
        raise ValueError('Missing or unsafe snapshot manifest')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format') != 1 or set(manifest.get('files', {})) != set(FILES):
        raise ValueError('Unsupported snapshot format or file list')
    parts = manifest.get('parts', [])
    if not parts or not 0 < manifest.get('part_bytes', 0) < GIT_LIMIT:
        raise ValueError('Invalid part list or part size')
    for index, item in enumerate(parts, 1):
        if item['path'] != f'parts/snapshot-{index:05d}.part':
            raise ValueError('Invalid or unordered snapshot part path')
        if not 0 < item['bytes'] <= manifest['part_bytes']:
            raise ValueError('Part exceeds declared size bound')
        if index < len(parts) and item['bytes'] != manifest['part_bytes']:
            raise ValueError('Short non-final snapshot part')
    if sum(item['bytes'] for item in parts) != manifest['archive_bytes']:
        raise ValueError('Archive size differs from part sizes')
    for item in manifest['files'].values():
        if not isinstance(item.get('bytes'), int) or item['bytes'] < 0:
            raise ValueError('Invalid restored file size')
    return manifest


def prepare(source: Path, destination: Path, *, part_bytes=PART_BYTES) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('Snapshot destination already exists; use a new directory')
    if not 0 < part_bytes < GIT_LIMIT:
        raise ValueError('Each part must be smaller than 100 MiB')
    # The transfer objects, rather than the restored databases, obey the Git limit.
    validate(source, enforce_git_limit=False)
    for name in ('data/racing.duckdb', 'data/features.duckdb'):
        if Path(str(source / name) + '.wal').exists():
            raise ValueError('Close the database writer before taking a snapshot')
    files = {name: {'bytes': (source / name).stat().st_size,
                    'sha256': digest(source / name)} for name in FILES}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.snapshot-stage-') as temporary:
        temporary = Path(temporary)
        stage = temporary / 'snapshot'
        (stage / 'parts').mkdir(parents=True)
        archive_path = temporary / 'payload.zip'
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in FILES:
                archive.write(source / name, name)
        # Detect a concurrent change rather than publishing inconsistent data.
        if any(digest(source / name) != item['sha256'] for name, item in files.items()):
            raise ValueError('Source changed during snapshot preparation')
        parts = []
        with archive_path.open('rb') as stream:
            index = 1
            while block := stream.read(part_bytes):
                name = f'parts/snapshot-{index:05d}.part'
                (stage / name).write_bytes(block)
                parts.append({'path': name, 'bytes': len(block),
                              'sha256': hashlib.sha256(block).hexdigest()})
                index += 1
        manifest = {'format': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'part_bytes': part_bytes, 'archive_bytes': archive_path.stat().st_size,
                    'archive_sha256': digest(archive_path), 'parts': parts, 'files': files}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        # Audits can read the pinned environment without downloading the databases.
        for name in ('prediction_bundle/requirements.txt', 'prediction_bundle/.python-version'):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
            if digest(target) != files[name]['sha256']:
                raise ValueError('Environment metadata changed during snapshot preparation')
        (stage / '.gitattributes').write_text('*.part -text -diff\n', encoding='utf-8')
        inspect_manifest(stage)
        if any(path.stat().st_size >= GIT_LIMIT for path in stage.rglob('*') if path.is_file()):
            raise ValueError('Snapshot includes a Git object at or above 100 MiB')
        # Publish only a completed directory; an older snapshot is never overwritten.
        stage.rename(destination)
    return manifest


def restore(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination == ROOT:
        raise ValueError('Restore requires a new directory; live databases are not overwritten')
    manifest = inspect_manifest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.restore-stage-') as temporary:
        temporary = Path(temporary)
        archive_path = temporary / 'payload.zip'
        archive_hash = hashlib.sha256()
        with archive_path.open('wb') as archive:
            for item in manifest['parts']:
                path = source / item['path']
                if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes']:
                    raise ValueError('Missing or invalid snapshot part')
                part_hash = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        part_hash.update(block)
                        archive_hash.update(block)
                        archive.write(block)
                if part_hash.hexdigest() != item['sha256']:
                    raise ValueError('Snapshot part hash mismatch')
        if archive_hash.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Snapshot archive hash mismatch')
        stage = temporary / 'restored'
        stage.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if len(archive.namelist()) != len(FILES) or set(archive.namelist()) != set(FILES):
                raise ValueError('Unexpected archive member list')
            for name in FILES:
                item = manifest['files'][name]
                if archive.getinfo(name).file_size != item['bytes']:
                    raise ValueError('Restored file size differs from manifest')
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as original, target.open('wb') as output:
                    shutil.copyfileobj(original, output)
                if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                    raise ValueError('Restored file hash mismatch')
        validate(stage, enforce_git_limit=False)
        stage.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'restore'):
        sub = commands.add_parser(command)
        sub.add_argument('--source', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        if command == 'prepare':
            sub.add_argument('--part-mib', type=int, default=64)
    args = parser.parse_args()
    private_root = (ROOT / '.private-runtime').resolve()
    output = args.output.resolve()
    if output != private_root and private_root not in output.parents:
        parser.error('Output must be inside ignored .private-runtime')
    manifest = (prepare(args.source, output, part_bytes=args.part_mib * 1024 * 1024)
                if args.command == 'prepare' else restore(args.source, output))
    print(json.dumps({'archive_bytes': manifest['archive_bytes'],
                      'parts': len(manifest['parts']), 'file_count': len(manifest['files'])}))


if __name__ == '__main__':
    main()
