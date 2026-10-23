"""Shared legacy/split restore and transactional activation, stdlib only."""
from importlib import import_module
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import zipfile

from scripts import private_runtime as legacy
from scripts import private_snapshot as split

runtime_lock = import_module('src.00_common.runtime_lock').runtime_lock
METADATA = ('prediction_bundle/requirements.txt', 'prediction_bundle/.python-version')


def materialize(source: Path, destination: Path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('復元先には新しいフォルダーを指定してください。')
    if (source / split.MANIFEST).exists():
        split.restore(source, destination)
    else:
        legacy.copy_runtime(source, destination)


def metadata(source: Path, destination: Path):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (source / split.MANIFEST).exists():
        manifest = split.inspect_manifest(source)
        if not all((source / name).is_file() for name in METADATA):
            with tempfile.TemporaryDirectory() as temporary:
                restored = Path(temporary) / 'runtime'
                materialize(source, restored)
                return metadata(restored, destination)
        for name in METADATA:
            if (source / name).is_symlink() or legacy.digest(source / name) != manifest['files'][name]['sha256']:
                raise ValueError('環境設定ファイルのハッシュが一致しません。')
    else:
        manifest = json.loads((source / 'prediction_bundle/manifest.json').read_text(encoding='utf-8'))
        if legacy.digest(source / METADATA[0]) != manifest['requirements_sha256']:
            raise ValueError('依存ライブラリ設定のハッシュが一致しません。')
        if (source / METADATA[1]).read_text().strip() != manifest['python_version']:
            raise ValueError('Python設定が一致しません。')
    for name in METADATA:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)


def activate(source: Path, root: Path, *, expected_hashes=None):
    """Verify before touching live files, back up, and roll back on write failure."""
    root = Path(root).resolve()
    private = root / '.private-runtime'
    with runtime_lock(root):
        private.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private, prefix='activate-') as temporary:
            staged = Path(temporary) / 'runtime'
            materialize(source, staged)
            bundle = json.loads((staged / 'prediction_bundle/manifest.json').read_text(encoding='utf-8'))
            model_id = bundle.get('model_id')
            targets = list(legacy.FILES)
            if model_id is not None:
                if not re.fullmatch(r'[A-Za-z0-9_-]+', model_id):
                    raise ValueError('Invalid model ID')
                with zipfile.ZipFile(staged / 'prediction_bundle/runtime.zip') as archive:
                    for name in archive.namelist():
                        if not name.startswith('model/'):
                            continue
                        if len(Path(name).parts) != 2:
                            raise ValueError('Invalid model archive member')
                        target = f'models/top3/{model_id}/{Path(name).name}'
                        content = archive.read(name)
                        import hashlib
                        if hashlib.sha256(content).hexdigest() != bundle['files'][name]:
                            raise ValueError('Model payload hash mismatch')
                        (staged / target).parent.mkdir(parents=True, exist_ok=True)
                        (staged / target).write_bytes(content)
                        targets.append(target)
                model_root = staged / f'models/top3/{model_id}'
                for name in ('model.pt', 'preprocessor.joblib', 'metrics.json', 'manifest.json'):
                    if not (model_root / name).is_file():
                        raise ValueError('Required model artifact is missing')
                for name in ('metrics.json', 'manifest.json'):
                    if json.loads((model_root / name).read_text(encoding='utf-8')).get('model_run_id') != model_id:
                        raise ValueError('Model artifacts and bundle model ID differ')
            if expected_hashes is not None:
                current = hashes(root)
                if current != expected_hashes:
                    raise ValueError('ローカルデータが変更されています。再確認してから復元してください。')
            for name in ('data/racing.duckdb', 'data/features.duckdb'):
                if Path(str(root / name) + '.wal').exists():
                    raise ValueError('DB更新処理を終了してから復元してください。')
            backup = private / 'backups' / uuid.uuid4().hex
            existed = []
            for name in targets:
                target = root / name
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError('Symlink target cannot be replaced')
                if target.exists():
                    old = backup / name
                    old.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, old)
                    existed.append(name)
            # Backup metadata survives for manual recovery even after a process interruption.
            backup.mkdir(parents=True, exist_ok=True)
            journal = backup / 'activation.json'
            state = {'status': 'installing', 'targets': targets, 'existed': existed}
            journal.write_text(json.dumps(state, indent=2), encoding='utf-8')
            pending = private / 'activation-pending.json'
            pending_stage = Path(temporary) / 'pending.json'
            pending_stage.write_text(json.dumps({'backup': backup.name}), encoding='utf-8')
            pending_stage.replace(pending)
            installed = []
            try:
                for name in targets:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (staged / name).replace(target)
                    installed.append(name)
            except BaseException:
                for name in reversed(installed):
                    if name in existed:
                        shutil.copy2(backup / name, root / name)
                    else:
                        (root / name).unlink()
                state['status'] = 'rolled_back'
                journal.write_text(json.dumps(state, indent=2), encoding='utf-8')
                pending.unlink()
                raise
            state['status'] = 'completed'
            journal.write_text(json.dumps(state, indent=2), encoding='utf-8')
            pending.unlink()
            return {'backup': str(backup), 'model_id': model_id, 'files': len(targets)}


def hashes(root):
    return {name: legacy.digest(Path(root) / name) if (Path(root) / name).is_file() else None
            for name in legacy.FILES}


def recover(root):
    root = Path(root).resolve()
    private = root / '.private-runtime'
    with runtime_lock(root, allow_recovery=True):
        pending = private / 'activation-pending.json'
        backup_id = json.loads(pending.read_text())['backup']
        if not re.fullmatch(r'[0-9a-f]{32}', backup_id):
            raise ValueError('Invalid recovery backup')
        backup = private / 'backups' / backup_id
        state = json.loads((backup / 'activation.json').read_text())
        if not set(state['existed']) <= set(state['targets']):
            raise ValueError('Invalid recovery targets')
        for name in state['targets']:
            if name not in legacy.FILES and not re.fullmatch(r'models/top3/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', name):
                raise ValueError('Invalid recovery path')
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError('Unsafe recovery target')
            if name in state['existed'] and not (backup / name).is_file():
                raise ValueError('Missing recovery backup file')
        for name in state['targets']:
            target = root / name
            if name in state['existed']:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / name, target)
            elif target.exists():
                target.unlink()
        state['status'] = 'recovered'
        (backup / 'activation.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        pending.unlink()
        return {'backup': str(backup)}
