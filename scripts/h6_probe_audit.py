"""Which probe / evaluation positions drop which RenjuNet games (H6-1 exclusion audit, docs/h6-value.md).

No proofs, a few minutes: for every game, the probe positions it reaches (D4, side to move included).
Reports per probe how many games reach it, and per ``--probe-game-min-ply`` threshold how many games a
label build would drop whole. Counts only (no positions from the games); safe to commit.

    python scripts/h6_probe_audit.py --games ..\\renju-alphazero\\data\\external\\renjunet\\games.jsonl.gz \\
        --workers 12 --output docs/h6/h6_probe_audit.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_labels import canonical_hash, default_probe_files, probe_positions  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402

THRESHOLDS = (0, 4, 5, 6, 7, 8, 9, 10, 12, 16)
_PROBES: dict = {}


def _init(probes):
    _PROBES.update(probes)


def _hits(chunk):
    """Per game: the probe hashes any prefix reaches."""
    return [sorted({h for ply in range(len(moves) + 1) if (h := canonical_hash(moves[:ply])) in _PROBES})
            for moves in chunk]


def audit(games: list[dict], probes: dict, *, workers: int = 1, chunk: int = 200) -> dict:
    moves = [g['moves'] for g in games]
    chunks = [moves[i:i + chunk] for i in range(0, len(moves), chunk)]
    progress = Progress('probe audit', len(moves), 'games')
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(probes,)) as pool:
            parts = run_chunks(pool, _hits, chunks, progress, len)
    else:
        _init(probes)
        parts = []
        for c in chunks:
            parts.append(_hits(c))
            progress.update(len(c))
    hits = [h for part in parts for h in part]
    by_probe = {}
    for game_hits in hits:
        for h in game_hits:
            by_probe[h] = by_probe.get(h, 0) + 1
    by_split = {}
    for g, game_hits in zip(games, hits):
        s = by_split.setdefault(g['split'], {'games': 0, 'reach_any_probe': 0})
        s['games'] += 1
        s['reach_any_probe'] += bool(game_hits)
    return {
        'games': len(games), 'probe_positions': len(probes),
        'games_dropped_by_min_ply': {str(n): sum(any(probes[h]['ply'] >= n for h in gh) for gh in hits)
                                     for n in THRESHOLDS},
        'by_split': by_split,
        'probes_reached': sorted(({**probes[h], 'hash': h, 'games': n} for h, n in by_probe.items()),
                                 key=lambda r: (-r['games'], r['ply'])),
        'probes_by_ply': {str(p): sum(v['ply'] == p for v in probes.values())
                          for p in sorted({v['ply'] for v in probes.values()})},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, required=True, help='H2 games.jsonl.gz')
    parser.add_argument('--probe-files', nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_h6_labels import read_games
    probe_files = default_probe_files(ROOT) if args.probe_files is None else args.probe_files
    result = {'format': 'h6-probe-audit-v1', **audit(read_games(args.games), probe_positions(probe_files, ROOT),
                                                     workers=args.workers)}
    print(json.dumps({k: v for k, v in result.items() if k != 'probes_reached'}, indent=1), flush=True)
    print(json.dumps(result['probes_reached'][:25], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
