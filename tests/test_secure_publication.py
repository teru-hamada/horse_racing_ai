import pytest

from scripts.publish_public_site import git, stage_site
from scripts.validate_public_site import validate_repository


def repository(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'Test')
    git(root, 'config', 'user.email', 'test@example.invalid')
    (root / 'docs').mkdir()
    (root / 'docs/index.html').write_text('<html>original</html>')
    git(root, 'add', '.')
    git(root, 'commit', '-m', 'initial')
    base = git(root, 'rev-parse', 'HEAD')
    git(root, 'update-ref', 'refs/remotes/origin/main', base)
    source = tmp_path / 'artifact'
    source.mkdir()
    (source / 'index.html').write_text('<html>new prediction</html>')
    return root, source, base


def test_publish_changes_only_docs(tmp_path):
    root, source, base = repository(tmp_path)
    commit = stage_site(source, root, base, 'main')
    assert commit != base
    assert git(root, 'diff', '--name-only', base, commit) == 'docs/index.html'
    assert (root / 'docs/index.html').read_text() == '<html>new prediction</html>'


def test_stale_prediction_cannot_overwrite_updated_public_data(tmp_path):
    root, source, base = repository(tmp_path)
    (root / 'docs/index.html').write_text('<html>settled results</html>')
    git(root, 'add', 'docs')
    git(root, 'commit', '-m', 'settle results')
    latest = git(root, 'rev-parse', 'HEAD')
    git(root, 'update-ref', 'refs/remotes/origin/main', latest)
    with pytest.raises(ValueError, match='changed after prediction'):
        stage_site(source, root, base, 'main')
    assert git(root, 'rev-parse', 'HEAD') == latest
    assert 'settled results' in (root / 'docs/index.html').read_text()


def test_private_artifact_is_rejected_before_repository_changes(tmp_path):
    root, source, base = repository(tmp_path)
    (source / 'racing.duckdb').write_bytes(b'private')
    with pytest.raises(ValueError):
        stage_site(source, root, base, 'main')
    assert git(root, 'rev-parse', 'HEAD') == base
    assert not git(root, 'status', '--porcelain')


def test_repository_guard_rejects_private_data_added_to_git(tmp_path):
    root, _, _ = repository(tmp_path)
    (root / 'data').mkdir()
    (root / 'data/racing.duckdb').write_bytes(b'private database')
    git(root, 'add', 'data')
    with pytest.raises(ValueError, match='Private file tracked'):
        validate_repository(root)


def test_repository_guard_rejects_credential_added_to_code(tmp_path):
    root, _, _ = repository(tmp_path)
    (root / 'settings.py').write_text('token = "' + 'ghp_' + 'a' * 36 + '"')
    git(root, 'add', 'settings.py')
    with pytest.raises(ValueError, match='Possible credential tracked'):
        validate_repository(root)
