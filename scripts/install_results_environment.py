"""Install only ingestion dependencies from the private pinned environment."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, required=True)
    args = parser.parse_args()
    wanted = {'numpy', 'pandas', 'requests', 'beautifulsoup4', 'lxml', 'duckdb'}
    pins = {}
    for line in args.requirements.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_-]+)==([A-Za-z0-9.+_-]+)', line.strip())
        if match and match[1] in wanted:
            pins[match[1]] = match[0]
    if set(pins) != wanted:
        raise SystemExit('Missing pinned ingestion dependencies')
    floors = []
    for line in (Path(__file__).resolve().parents[1] / 'requirements-security.txt').read_text().splitlines():
        if any(line.startswith(name + '>=') for name in ('certifi', 'urllib3', 'pip', 'setuptools')):
            floors.append(line)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *pins.values(), *floors], check=True)


if __name__ == '__main__':
    main()
