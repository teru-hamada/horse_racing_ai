"""Copy an allowlisted private runtime without publishing database/model files.

Uses only the standard library so restore can run before pip installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'prediction_bundle/runtime.zip', 'prediction_bundle/manifest.json',
    'prediction_bundle/requirements.txt', 'prediction_bundle/.python-version',
    'data/racing.duckdb', 'data/features.duckdb',
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate(source: Path) -> None:
    for name in FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe private runtime file: {name}')
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f'File exceeds the GitHub Git file size limit: {name}')
    bundle = source / 'prediction_bundle'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    for filename, key in (('runtime.zip', 'archive_sha256'), ('requirements.txt', 'requirements_sha256')):
        if digest(bundle / filename) != manifest[key]:
            raise ValueError(f'Bundle hash mismatch: {filename}')
    if (bundle / '.python-version').read_text(encoding='utf-8').strip() != manifest['python_version']:
        raise ValueError('Python version differs from bundle manifest')
    # Check member names before any model deserialization occurs downstream.
    with zipfile.ZipFile(bundle / 'runtime.zip') as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']):
            raise ValueError('Unexpected bundle archive members')
        for name in names:
            if name.startswith('/') or '\\' in name or '..' in name.split('/'):
                raise ValueError('Unsafe bundle archive member')


def copy_runtime(source: Path, destination: Path) -> None:
    validate(source)
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    validate(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare', help='Prepare an ignored copy for a private repository')
    prepare.add_argument('--output', type=Path, default=ROOT / '.private-runtime')
    restore = subparsers.add_parser('restore', help='Restore a checked-out private snapshot into the application')
    restore.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        destination = args.output.resolve()
        # Keep generated private data under the already ignored workspace directory.
        private_root = (ROOT / '.private-runtime').resolve()
        if destination != private_root and private_root not in destination.parents:
            raise SystemExit('Output must be inside .private-runtime to avoid accidental publication.')
        if destination.exists():
            raise SystemExit('Output already exists; use a new subdirectory under .private-runtime.')
        copy_runtime(ROOT, destination)
        (destination / '.gitattributes').write_text('* -text\n', encoding='utf-8')
        (destination / 'README.md').write_text(
            '# Private prediction runtime\n\nKeep this repository PRIVATE. No Actions or Pages required.\n',
            encoding='utf-8')
        print('Prepared private runtime copy. Original files were preserved.')
    else:
        copy_runtime(args.source.resolve(), ROOT)
        print('Restored and verified private runtime files.')


if __name__ == '__main__':
    main()
