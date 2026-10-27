"""Run the read-only admin foundation on loopback port 8010 (deploy/README-admin.md)."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import signal
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from webadmin.exports import ExportManager  # noqa: E402
from webadmin.auth import Auth, hash_password  # noqa: E402
from webadmin.inspect import inspect_database  # noqa: E402
from webadmin.server import AdminServer  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-password', action='store_true', help='prompt securely and print a scrypt hash')
    parser.add_argument('--db', type=Path, default=ROOT / 'data/renju_web.db')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--export-dir', type=Path, default=ROOT / 'data/admin_exports')
    parser.add_argument('--export-keep', type=int, default=5)
    args = parser.parse_args(argv)
    if args.hash_password:
        first = getpass.getpass('Admin password: ')
        if first != getpass.getpass('Confirm password: '):
            print('passwords do not match', file=sys.stderr)
            return 1
        try:
            print(hash_password(first))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    try:
        auth = Auth(os.environ.get('ADMIN_USER', ''), os.environ.get('ADMIN_PASSWORD_HASH', ''))
        origin = os.environ.get('ADMIN_ORIGIN', '')
        if not origin:
            raise ValueError('ADMIN_ORIGIN is required')
        inspect_database(args.db)  # fail closed; never initialize or migrate the production DB
        server = AdminServer(('127.0.0.1', args.port), db=args.db, origin=origin,
                             auth=auth, assets=ROOT / 'web/admin',
                             exports=ExportManager(args.db, args.export_dir, repo=ROOT,
                                                   log_root=ROOT / 'logs/web_play', keep=args.export_keep))
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'admin startup failed: {exc}', file=sys.stderr)
        return 1

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f'Renju admin listening on loopback port {args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
