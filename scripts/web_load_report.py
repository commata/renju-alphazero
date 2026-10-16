"""Table of B3 load runs (``scripts/web_load_test.py`` outputs), one row per run (docs/web-ser5.md §9.11).

    python scripts/web_load_report.py runs/b3/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def cell(block, key):
    value = (block or {}).get(key)
    return '-' if value is None else f'{value:g}' if isinstance(value, float) else str(value)


def main(paths) -> int:
    print('| run | sessions | AI moves | wait median / p95 / max (s) | server queue p95 | engine p95 / max | '
          '>27.5 s | watchdog | emergency | helper failed | 503 | errors |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for path in sorted(paths):
        run = json.loads(Path(path).read_text(encoding='utf-8'))
        wait, server = run['client_wait_seconds'], run.get('server', {})
        engine = server.get('engine_seconds', {})
        over = next((v for k, v in server.items() if k.startswith('engine_over_')), '-')
        events = run['events']
        print(f"| {run.get('label') or Path(path).stem} | {run['sessions']} | {wait.get('n', 0)} | "
              f"{cell(wait, 'median')} / {cell(wait, 'p95')} / {cell(wait, 'max')} | "
              f"{cell(server.get('queue_seconds'), 'p95')} | {cell(engine, 'p95')} / {cell(engine, 'max')} | {over} | "
              f"{server.get('watchdog_moves', '-')} | {server.get('emergency_moves', '-')} | "
              f"{server.get('attack_helper_failed_moves', '-')} | {events['full_503']} | "
              f"{events['errors'] + events['ai_errors'] + len(run['failures'])} |")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
