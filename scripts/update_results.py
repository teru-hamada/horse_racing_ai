"""Build a result-only private snapshot candidate; never push or select a model."""
import argparse
from datetime import date
from importlib import import_module
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat)
    parser.add_argument('--end', type=date.fromisoformat)
    parser.add_argument('--max-days', type=int, default=7)
    args = parser.parse_args()
    result = import_module('src.10_workflows.result_update').update(
        args.source, args.output, args.work, start=args.start, end=args.end, max_days=args.max_days)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write(f'changed={str(result["changed"]).lower()}\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'days'}, ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
