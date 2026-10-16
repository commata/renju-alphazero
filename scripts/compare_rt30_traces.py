"""Side by side the ``--trace`` rows of ``run_rt30_divergence.py`` runs on the same case (docs/web-ser5.md §9.10).

For every own move: the move, its route, deadline events and the random-state digest before the
search, per run; then the first ply where the moves or the random states part, and the root visits
at the case ply. Used for the A3-v3 ``rt30_no_deadline`` move (``docs/rt30-eval/a3v3-no-deadline-v1.json``).

    python scripts/compare_rt30_traces.py runs/rt30/nd_tb1.json runs/rt30/nd_v3_free.json runs/rt30/nd_v3_rt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def label(path: Path) -> str:
    return path.stem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args(argv)
    runs = {label(p): json.loads(p.read_text(encoding='utf-8')) for p in args.runs}
    keys = sorted({r['key'] for run in runs.values() for r in run['rows']})
    for key in keys:
        rows = {name: next((r for r in run['rows'] if r['key'] == key), None) for name, run in runs.items()}
        rows = {name: r for name, r in rows.items() if r is not None}
        if any('trace' not in r for r in rows.values()):
            print(f'{key}: run without --trace'); continue
        names = list(rows)
        first = next(iter(rows.values()))
        print(f"## {key} (TB-Engine-v1 recorded {first['tb1_move']})\n")
        print('| ply | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
        traces = {n: {t['ply']: t for t in rows[n]['trace']} for n in names}
        plies = sorted({p for t in traces.values() for p in t})
        part_move = part_rng = None
        for ply in plies:
            cells = []
            for n in names:
                t = traces[n].get(ply)
                if t is None:
                    cells.append('-'); continue
                events = '/'.join(t['timeouts'] + [f'skip:{s}' for s in t['skipped']]) or '-'
                cells.append(f"{t['move']} {t['route']} {t['seconds']:.1f}s {events} rng {t['rng_before'][:8]}")
            print(f'| {ply} | ' + ' | '.join(cells) + ' |')
            present = [traces[n][ply] for n in names if ply in traces[n]]
            if part_move is None and len({tuple(t['move']) for t in present}) > 1:
                part_move = ply
            if part_rng is None and len({t['rng_before'] for t in present}) > 1:
                part_rng = ply
        print(f'\nfirst different move: {part_move}; first different random state before a search: {part_rng}')
        for n in names:
            visits = sorted(rows[n].get('root_visits', []), key=lambda v: -v[1])[:4]
            print(f"- {n}: case move {rows[n]['move']}, top root visits {visits}")
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
