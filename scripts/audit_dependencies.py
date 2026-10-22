"""Audit the complete installed environment, including the CPU PyTorch build.

The official +cpu build shares upstream vulnerability version numbering but is
not published on PyPI. Normalize only that suffix so PyTorch is audited too.
Fail if any installed distribution cannot be audited instead of silently skipping it.
"""
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def requirement_lines(versions: dict[str, str]) -> str:
    lines = []
    for name, version in sorted(versions.items()):
        if name.lower() == 'torch' and version.endswith('+cpu'):
            version = version.removesuffix('+cpu')
        lines.append(f'{name}=={version}\n')
    return ''.join(lines)


def check_coverage(report: dict) -> int:
    skipped = [item['name'] for item in report['dependencies'] if item.get('skip_reason')]
    if skipped:
        raise ValueError('Dependencies could not be audited: ' + ', '.join(skipped))
    return sum(len(item.get('vulns', [])) for item in report['dependencies'])


def main() -> int:
    versions = {dist.metadata['Name']: metadata.version(dist.metadata['Name'])
                for dist in metadata.distributions() if dist.metadata.get('Name')}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        requirements, report_path = root / 'installed.txt', root / 'report.json'
        requirements.write_text(requirement_lines(versions), encoding='utf-8')
        result = subprocess.run([
            sys.executable, '-m', 'pip_audit', '--no-deps', '--disable-pip',
            '--progress-spinner', 'off', '--format', 'json', '--output', str(report_path),
            '-r', str(requirements)], check=False)
        if not report_path.exists():
            raise RuntimeError('Dependency audit failed to produce a report')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        count = check_coverage(report)
        for item in report['dependencies']:
            for advisory in item.get('vulns', []):
                print(f'{item["name"]} {item["version"]}: {advisory["id"]}; fixes: {advisory["fix_versions"]}')
        if result.returncode or count:
            return 1
        print(f'All {len(report["dependencies"])} installed dependencies audited; no known vulnerabilities.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
