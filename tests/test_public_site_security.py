import zipfile

import pytest

from scripts.validate_public_site import validate_site


def site(tmp_path):
    (tmp_path / 'index.html').write_text('<html>Public prediction</html>', encoding='utf-8')
    return tmp_path


def test_public_predictions_and_audit_source_are_allowed(tmp_path):
    root = site(tmp_path)
    (root / '.nojekyll').touch()
    with zipfile.ZipFile(root / 'audit.zip', 'w') as archive:
        archive.writestr('source/betting.py', 'def score(): return 1\n')
        archive.writestr('predictions.csv', 'probability\n0.4\n')
    assert validate_site(root) == 2


@pytest.mark.parametrize('filename', ['racing.duckdb', 'model.pt', 'preprocessor.joblib',
                                     '.env', '.python-version', 'manifest.json', 'runtime.zip'])
def test_private_files_are_rejected(tmp_path, filename):
    root = site(tmp_path)
    (root / filename).write_text('private', encoding='utf-8')
    with pytest.raises(ValueError):
        validate_site(root)


@pytest.mark.parametrize('member', ['model/preprocessor.joblib', '../outside.csv',
                                   '.git/config', 'nested.zip'])
def test_private_or_unsafe_files_inside_zip_are_rejected(tmp_path, member):
    root = site(tmp_path)
    with zipfile.ZipFile(root / 'audit.zip', 'w') as archive:
        archive.writestr(member, 'private')
    with pytest.raises(ValueError):
        validate_site(root)


@pytest.mark.parametrize('in_archive', [False, True])
def test_credentials_are_rejected_without_printing_secret(tmp_path, in_archive):
    root = site(tmp_path)
    secret = 'ghp_' + 'a' * 36
    if in_archive:
        with zipfile.ZipFile(root / 'audit.zip', 'w') as archive:
            archive.writestr('settings.json', secret)
    else:
        (root / 'settings.json').write_text(secret)
    with pytest.raises(ValueError) as error:
        validate_site(root)
    assert secret not in str(error.value)


def test_database_disguised_as_csv_is_rejected(tmp_path):
    root = site(tmp_path)
    (root / 'data.csv').write_bytes(b'12345678DUCK' + b'0' * 50)
    with pytest.raises(ValueError):
        validate_site(root)


@pytest.mark.parametrize('html', ['<script>alert(1)</script>', '<img onerror="alert(1)">',
                                '<a href="java&#x09;script:alert(1)">link</a>',
                                '<iframe src="https://example.com"></iframe>',
                                '<meta http-equiv="refresh" content="0;url=https://example.com">'])
def test_injected_html_code_is_rejected(tmp_path, html):
    root = site(tmp_path)
    (root / 'index.html').write_text(html, encoding='utf-8')
    with pytest.raises(ValueError):
        validate_site(root)


def test_unapproved_javascript_asset_is_rejected(tmp_path):
    root = site(tmp_path)
    (root / 'job-status.js').write_text('alert(1)', encoding='utf-8')
    with pytest.raises(ValueError):
        validate_site(root)
