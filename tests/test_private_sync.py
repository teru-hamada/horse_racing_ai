"""Publishing is private-only, checks the base commit, and never pushes main."""
from importlib import import_module
import io
import json
import shutil

import pytest

from tests.test_runtime_transfer import make_source

sync = import_module('src.10_workflows.private_sync')
get_paths = import_module('src.00_common.config').get_paths


def test_public_repository_is_rejected_before_any_data_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, '_git', lambda *a, **k: 'password=fixture-token')
    monkeypatch.setattr(sync.urllib.request, 'urlopen', lambda *a, **k:
                        io.StringIO(json.dumps({'private': False})))
    with pytest.raises(ValueError, match='非公開'):
        sync.repository_info('example/public', root=tmp_path)


def test_changed_remote_base_rejects_save_before_preparing_data(tmp_path, monkeypatch):
    paths = get_paths(tmp_path)
    monkeypatch.setattr(sync, 'repository_info', lambda *a, **k: {'commit': 'b' * 40})
    monkeypatch.setattr(sync, '_git', lambda *a, **k: pytest.fail('No Git write should occur'))
    with pytest.raises(ValueError, match='更新'):
        sync.save_snapshot('example/private', 'a' * 40, paths=paths)


def test_save_only_pushes_new_candidate_branch_without_mutating_source(tmp_path, monkeypatch):
    root = make_source(tmp_path / 'app', b'current DB')
    paths = get_paths(root)
    before = sync.runtime_transfer.hashes(root)
    monkeypatch.setattr(sync, 'repository_info', lambda *a, **k: {'commit': 'a' * 40})
    modeling = import_module('src.40_ai_modeling.prediction_bundle')
    monkeypatch.setattr(modeling, 'export_bundle', lambda output, **k:
                        shutil.copytree(root / 'prediction_bundle', output))
    commands = []

    def git(args, **kwargs):
        commands.append(args)
        if args[:2] == ['config', 'user.name']:
            return 'Test'
        if args[:2] == ['config', 'user.email']:
            return 'test@example.invalid'
        if args[:2] == ['rev-parse', 'HEAD']:
            return 'c' * 40
        return ''

    monkeypatch.setattr(sync, '_git', git)
    result = sync.save_snapshot('example/private', 'a' * 40, paths=paths)
    pushes = [args for args in commands if args[0] == 'push']
    assert len(pushes) == 1
    assert pushes[0][2].startswith('HEAD:refs/heads/snapshots/gui-')
    assert result['commit'] == 'c' * 40
    assert sync.runtime_transfer.hashes(root) == before


def test_wrong_downloaded_commit_never_activates(tmp_path, monkeypatch):
    root = make_source(tmp_path / 'app', b'current DB')
    paths = get_paths(root)
    before = sync.runtime_transfer.hashes(root)
    monkeypatch.setattr(sync, 'repository_info', lambda *a, **k: {'commit': 'a' * 40})
    monkeypatch.setattr(sync, '_git', lambda args, **kwargs: 'b' * 40 if args[0] == 'rev-parse' else '')
    with pytest.raises(ValueError, match='SHA'):
        sync.restore_snapshot('example/private', 'a' * 40, paths=paths, expected_hashes=before)
    assert sync.runtime_transfer.hashes(root) == before
