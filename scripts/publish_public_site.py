"""Commit validated site data on a clean publisher with no private runtime access."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

try:
    from .validate_public_site import validate_site
except ImportError:
    from validate_public_site import validate_site


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def stage_site(source: Path, root: Path, baseline: str, branch: str) -> str:
    validate_site(source)
    # Refuse to replace predictions settled/updated by another run or a manual edit.
    if git(root, 'diff', '--name-only', baseline, f'origin/{branch}', '--', 'docs'):
        raise ValueError('Public files changed after prediction started; rerun instead of overwriting them.')
    git(root, 'merge', '--ff-only', f'origin/{branch}')
    for path in source.rglob('*'):
        if path.is_file():
            target = root / 'docs' / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    validate_site(root / 'docs')
    git(root, 'config', 'user.name', 'github-actions[bot]')
    git(root, 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git(root, 'add', '--', 'docs')
    if git(root, 'diff', '--cached', '--name-only'):
        git(root, 'commit', '-m', 'Publish daily predictions and result comparisons')
    return git(root, 'rev-parse', 'HEAD')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    branch = os.environ['DEFAULT_BRANCH']
    git(root, 'fetch', 'origin', branch)
    commit = stage_site(args.source, root, os.environ['GITHUB_SHA'], branch)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write(f'commit={commit}\n')
    print('Public site validated and prepared for publication.')
