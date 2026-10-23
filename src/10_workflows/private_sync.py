"""GUI/CLI private snapshot sync. Publishing creates a candidate, never promotes it."""
from dataclasses import replace
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request
import uuid

from scripts import private_snapshot, runtime_transfer

runtime_lock = import_module('src.00_common.runtime_lock').runtime_lock


def _git(args, *, cwd, input=None):
    result = subprocess.run(['git', *args], cwd=cwd, input=input,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'},
        text=True, encoding='utf-8', errors='replace', capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError('Git処理に失敗しました。GitHubへの認証と接続を確認してください。')
    return result.stdout.strip()


def repository_info(repository, *, root):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('リポジトリは owner/name で指定してください。')
    credential = _git(['credential', 'fill'], cwd=root, input='protocol=https\nhost=github.com\n\n')
    fields = dict(line.split('=', 1) for line in credential.splitlines() if '=' in line)
    if not fields.get('password'):
        raise RuntimeError('GitHubの認証情報がありません。Git Credential Manager等で認証してください。')

    def read(path):
        request = urllib.request.Request('https://api.github.com/' + path, headers={
            'Authorization': 'Bearer ' + fields['password'], 'Accept': 'application/vnd.github+json',
            'User-Agent': 'horse-racing-private-sync', 'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    repo = read(f'repos/{repository}')
    if repo['private'] is not True:
        raise ValueError('非公開リポジトリだけを利用できます。')
    commit = read(f'repos/{repository}/commits/{repo["default_branch"]}')['sha']
    return {'repository': repository, 'branch': repo['default_branch'], 'commit': commit}


def save_snapshot(repository, expected_base, *, paths, check_idle=lambda: None):
    root = paths.root
    if not re.fullmatch(r'[0-9a-fA-F]{40}', expected_base):
        raise ValueError('先に接続確認を実行してください。')
    with runtime_lock(root):
        check_idle()
        info = repository_info(repository, root=root)
        if info['commit'] != expected_base:
            raise ValueError('保存先が更新されています。接続確認をやり直してください。')
        private = root / '.private-runtime'
        private.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private, prefix='save-') as temporary:
            temporary = Path(temporary)
            candidate = temporary / 'candidate'
            (candidate / 'data').mkdir(parents=True)
            source_hashes = runtime_transfer.hashes(root)
            for name, source in (('racing.duckdb', paths.database), ('features.duckdb', paths.feature_database)):
                if Path(str(source) + '.wal').exists():
                    raise ValueError('DB更新処理を終了してから保存してください。')
                shutil.copy2(source, candidate / 'data' / name)
            modeling = import_module('src.40_ai_modeling.prediction_bundle')
            modeling.export_bundle(candidate / 'prediction_bundle', paths=replace(
                paths, database=candidate / 'data/racing.duckdb', feature_database=candidate / 'data/features.duckdb'))
            if runtime_transfer.hashes(root) != source_hashes:
                raise ValueError('保存中にローカルデータが更新されました。再実行してください。')
            repository_path = temporary / 'repository'
            manifest = private_snapshot.prepare(candidate, repository_path)
            branch = 'snapshots/gui-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
            _git(['init', '-b', branch], cwd=repository_path)
            for key in ('user.name', 'user.email'):
                _git(['config', key, _git(['config', key], cwd=root)], cwd=repository_path)
            (repository_path / 'README.md').write_text(
                '# Private runtime candidate\n\nCreated by GUI. Base commit: ' + expected_base +
                '\nAdoption by the prediction workflow is a separate operation.\n', encoding='utf-8')
            _git(['add', '--', 'snapshot.json', 'parts', 'prediction_bundle', '.gitattributes', 'README.md'], cwd=repository_path)
            _git(['commit', '-m', 'Save split private runtime candidate'], cwd=repository_path)
            if repository_info(repository, root=root)['commit'] != expected_base:
                raise ValueError('保存先が更新されています。接続確認をやり直してください。')
            _git(['remote', 'add', 'origin', f'https://github.com/{repository}.git'], cwd=repository_path)
            _git(['push', 'origin', f'HEAD:refs/heads/{branch}'], cwd=repository_path)
            commit = _git(['rev-parse', 'HEAD'], cwd=repository_path)
            return {'commit': commit, 'branch': branch, 'parts': len(manifest['parts']),
                    'bytes': manifest['archive_bytes'], 'url': f'https://github.com/{repository}/tree/{branch}'}


def restore_snapshot(repository, commit, *, paths, expected_hashes, check_idle=lambda: None):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
        raise ValueError('取得するコミットSHAを40文字で指定してください。')
    with runtime_lock(paths.root):
        check_idle()
        repository_info(repository, root=paths.root)
        private = paths.root / '.private-runtime'
        private.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private, prefix='download-') as temporary:
            download = Path(temporary)
            _git(['init'], cwd=download)
            _git(['remote', 'add', 'origin', f'https://github.com/{repository}.git'], cwd=download)
            _git(['fetch', '--depth', '1', 'origin', commit], cwd=download)
            _git(['checkout', '--detach', 'FETCH_HEAD'], cwd=download)
            if _git(['rev-parse', 'HEAD'], cwd=download).lower() != commit.lower():
                raise ValueError('取得したコミットSHAが一致しません。')
            result = runtime_transfer.activate(download, paths.root, expected_hashes=expected_hashes)
            result['commit'] = commit
            return result
