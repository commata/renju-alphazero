"""Online backup of the web game database (deploy/README.md §6): a consistent copy while the server
writes (SQLite backup API, WAL-safe), then delete all but the newest ``--keep`` copies.

    .venv-ser5/bin/python deploy/backup_db.py --db data/renju_web.db --dir data/backups --keep 14
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weblog.backup import snapshot_database  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=Path, default=Path('data/renju_web.db'))
    parser.add_argument('--dir', type=Path, default=Path('data/backups'))
    parser.add_argument('--keep', type=int, default=14)
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f'no database at {args.db}', file=sys.stderr)
        return 1
    args.dir.mkdir(parents=True, exist_ok=True)
    target = args.dir / f'{args.db.stem}-{datetime.now():%Y%m%d-%H%M%S}.db'
    snapshot_database(args.db, target)
    backups = sorted(args.dir.glob(f'{args.db.stem}-*.db'))
    for old in backups[:-args.keep] if args.keep > 0 else []:
        old.unlink()
    print(f'backup {target} ({target.stat().st_size:,} bytes), kept {min(len(backups), args.keep)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
