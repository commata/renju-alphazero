"""Read-only stage-1 report. Run on SER5; no engine, torch or credentials needed.

    .venv-ser5/bin/python scripts/ser5/admin_inspect.py --db data/renju_web.db

Writes JSON to stdout, diagnostics to stderr; missing/wrong schema is exit 1. This
report describes the machine where it ran, not a remote SER5 or a newer database.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from webadmin.inspect import checkout_commit, inspect_database  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT / 'data/renju_web.db')
    parser.add_argument('--session-ttl', type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        report = inspect_database(args.db, ttl=args.session_ttl)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f'admin inspection failed: {exc}', file=sys.stderr)
        return 1
    report['checkout_commit'] = checkout_commit(ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
