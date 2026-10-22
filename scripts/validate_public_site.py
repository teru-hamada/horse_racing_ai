"""Reject private data, unsafe paths and credentials before public artifact/upload.

Only standard-library code runs in the publishing job. ZIP audit files are
inspected without extracting them; downloaded artifact code is never executed.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

PUBLIC_EXTENSIONS = {'.html', '.csv', '.json', '.js', '.md', '.zip'}
AUDIT_EXTENSIONS = {'.html', '.csv', '.json', '.py', '.md', '.txt'}
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
SECRETS = re.compile(
    rb'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|'
    rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b|'
    rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b|'
    rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'
)


class PublicHTML(HTMLParser):
    def __init__(self, name: str):
        super().__init__(convert_charrefs=True)
        self.name = name
        self.in_script = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag in {'iframe', 'object', 'embed', 'form', 'base'}:
            raise ValueError(f'Active embedded content in public page: {self.name}')
        if tag == 'meta' and (attrs.get('http-equiv') or '').lower() == 'refresh':
            raise ValueError(f'Automatic redirect in public page: {self.name}')
        for key, value in attributes:
            if key.startswith('on'):
                raise ValueError(f'Inline event handler in public page: {self.name}')
            if key in {'href', 'src', 'action', 'formaction'} and value:
                normalized = re.sub(r'[\x00-\x20]', '', value).lower()
                if normalized.startswith(('javascript:', 'vbscript:', 'data:')):
                    raise ValueError(f'Unsafe URL in public page: {self.name}')
        if tag == 'script':
            if self.name != 'job-status.html' or attrs.get('src') != 'job-status.js':
                raise ValueError(f'Unapproved script in public page: {self.name}')
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False

    def handle_data(self, data):
        if self.in_script and data.strip():
            raise ValueError(f'Inline script in public page: {self.name}')


def check_name(name: str, extensions: set[str]) -> None:
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith('/') or '\\' in name or ':' in name
            or '..' in parts or any(part.startswith('.') for part in parts)):
        raise ValueError(f'Unsafe public path: {name}')
    if PurePosixPath(name).suffix.lower() not in extensions:
        raise ValueError(f'Forbidden public file type: {name}')
    if extensions == PUBLIC_EXTENSIONS and PurePosixPath(name).name.lower() in {'manifest.json', 'requirements.txt', 'runtime.zip'}:
        # These bundle filenames are never needed on the published site.
        raise ValueError(f'Private runtime file in public site: {name}')


def scan(content: bytes, name: str) -> None:
    if SECRETS.search(content):
        # Never print matching contents, only the affected filename.
        raise ValueError(f'Possible credential in public file: {name}')
    if content[:16].startswith((b'\x80\x04', b'\x80\x05')) or content[:4] == b'PAR1':
        raise ValueError(f'Binary model/data disguised as text: {name}')
    if content[8:12] == b'DUCK' or content.startswith(b'SQLite format 3'):
        raise ValueError(f'Database disguised as text: {name}')
    try:
        content.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError(f'Non-text content in public file: {name}') from None
    if b'\x00' in content:
        raise ValueError(f'Binary content in public text file: {name}')


def validate_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 10000 or len({item.filename for item in members}) != len(members):
            raise ValueError(f'Invalid audit archive members: {path.name}')
        if sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
            raise ValueError(f'Audit archive is too large: {path.name}')
        for item in members:
            check_name(item.filename.rstrip('/'), AUDIT_EXTENSIONS if not item.is_dir() else {''})
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or item.flag_bits & 1 or item.file_size > MAX_FILE_BYTES:
                raise ValueError(f'Unsafe audit member: {item.filename}')
            if not item.is_dir():
                scan(archive.read(item), f'{path.name}/{item.filename}')


def validate_site(root: Path) -> int:
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Public site must be a regular directory')
    count = 0
    for path in root.rglob('*'):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f'Public symlink is forbidden: {name}')
        if path.is_dir():
            if any(part.startswith('.') for part in path.relative_to(root).parts):
                raise ValueError(f'Hidden directory in public site: {name}')
            continue
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'Invalid public file: {name}')
        if name == '.nojekyll':
            if path.read_bytes().strip():
                raise ValueError('.nojekyll must be empty')
            continue
        check_name(name, PUBLIC_EXTENSIONS)
        if path.suffix.lower() == '.zip':
            validate_archive(path)
        else:
            content = path.read_bytes()
            scan(content, name)
            if path.suffix.lower() == '.html':
                PublicHTML(name).feed(content.decode('utf-8-sig'))
            if path.suffix.lower() == '.js':
                trusted = Path(__file__).resolve().parents[1] / 'src/60_publication/job_status.js'
                if name != 'job-status.js' or content.replace(b'\r\n', b'\n') != trusted.read_bytes().replace(b'\r\n', b'\n'):
                    raise ValueError(f'Unapproved JavaScript asset: {name}')
        count += 1
    if not (root / 'index.html').is_file():
        raise ValueError('Public site index.html is missing')
    return count


def validate_repository(root: Path) -> None:
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode('utf-8').split('\0')
    private_extensions = {'.duckdb', '.pt', '.joblib', '.pkl', '.pickle', '.pem', '.key', '.sqlite', '.sqlite3', '.parquet'}
    private_bundle = {'runtime.zip', 'manifest.json', 'requirements.txt', '.python-version'}
    for name in filter(None, names):
        path = root / name
        parts = PurePosixPath(name).parts
        if (path.suffix.lower() in private_extensions or parts[0] in {'.private-runtime', 'models'}
                or path.name == '.env' or path.name.endswith('.env')
                or name == '.streamlit/secrets.toml'
                or (parts[0] == 'prediction_bundle' and path.name in private_bundle)):
            raise ValueError(f'Private file tracked in public repository: {name}')
        if path.is_symlink():
            raise ValueError(f'Tracked symlink is forbidden: {name}')
        if path.is_file() and path.suffix.lower() != '.zip' and SECRETS.search(path.read_bytes()):
            raise ValueError(f'Possible credential tracked in public repository: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path, nargs='?', default=Path('docs'))
    parser.add_argument('--check-repository', action='store_true')
    args = parser.parse_args()
    if args.check_repository:
        validate_repository(Path(__file__).resolve().parents[1])
    print(f'Public file safety passed: {validate_site(args.site)} files checked.')
