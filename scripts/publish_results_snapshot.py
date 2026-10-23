"""Publish a verified result-only candidate with optimistic Git concurrency."""
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import private_runtime, private_snapshot, runtime_transfer

STATE = 'result_update_state.json'
REPORT = 'result_update_report.json'


def git(repository, *args):
    result = subprocess.run(['git', *args], cwd=repository, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=240)
    if result.returncode:
        raise RuntimeError(f'Git operation failed: {args[0]} (exit {result.returncode})')
    return result.stdout.strip()


def stage_candidate(repository, candidate, expected_commit, report):
    repository, candidate = Path(repository).resolve(), Path(candidate).resolve()
    if not re.fullmatch(r'[0-9a-fA-F]{40}', expected_commit) or git(repository, 'rev-parse', 'HEAD') != expected_commit:
        raise ValueError('Private checkout no longer matches the expected base')
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        runtime_transfer.materialize(repository, temporary / 'before')
        runtime_transfer.materialize(candidate, temporary / 'after')
        for name in private_runtime.FILES:
            if name != 'data/racing.duckdb' and private_runtime.digest(temporary / 'before' / name) != private_runtime.digest(temporary / 'after' / name):
                raise ValueError('A result-only update cannot change model/bundle/features')
    manifest = private_snapshot.inspect_manifest(candidate)
    old_parts = []
    if (repository / private_snapshot.MANIFEST).exists():
        old_parts = [part['path'] for part in private_snapshot.inspect_manifest(repository)['parts']]
    tracked = set(git(repository, 'ls-files').splitlines())
    new_names = [private_snapshot.MANIFEST, '.gitattributes', STATE, *runtime_transfer.METADATA,
                 *(part['path'] for part in manifest['parts'])]
    touched = set(new_names + old_parts + list(private_runtime.FILES) + [REPORT])
    # Every mutated path is generated from a fixed allowlist or validated part manifest.
    for name in touched:
        if not (repository / name).resolve().is_relative_to(repository):
            raise ValueError('Unsafe private repository target')
    for name in list(private_runtime.FILES) + old_parts:
        path = repository / name
        if name not in new_names and path.is_file():
            path.unlink()
    for name in new_names:
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate / name, target)
    shutil.copy2(report, repository / REPORT)
    paths = sorted(name for name in touched if name in tracked or (repository / name).is_file())
    if any((repository / name).is_file() and (repository / name).stat().st_size >= private_snapshot.GIT_LIMIT for name in paths):
        raise ValueError('Git file size limit exceeded')
    git(repository, 'add', '-A', '--', *paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    # This publisher is intentionally run by the private repository's own token.
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'))
    if event.get('repository', {}).get('private') is not True:
        raise SystemExit('Result publishing requires a private repository event')
    remote = git(args.repository, 'ls-remote', 'origin', 'refs/heads/main').split()[0]
    if remote != args.expected_commit:
        raise SystemExit('Private main changed; retry from the latest snapshot')
    stage_candidate(args.repository, args.candidate, args.expected_commit, args.report)
    git(args.repository, 'config', 'user.name', 'github-actions[bot]')
    git(args.repository, 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git(args.repository, 'commit', '-m', 'Update verified historical race results')
    # Never force push: a simultaneous GUI/manual update must cause a conflict.
    git(args.repository, 'push', 'origin', 'HEAD:refs/heads/main')
    commit = git(args.repository, 'rev-parse', 'HEAD')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write(f'saved_sha={commit}\n')
    print('Saved verified result DB:', commit)


if __name__ == '__main__':
    main()
