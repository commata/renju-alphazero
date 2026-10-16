"""H6 builder --fast check on real games: the same rows and summary in every mode, and each mode's speed.

Runs ``build_h6_labels.build`` once per mode on the same deterministic game sample (the pilot's hash
sample, settings as the v3 labels), compares the rows (as JSON lines, by SHA256) and the summary
(including ``prove_losses_budget_cuts`` and ``proof_by_distance_to_end``) with the first mode, and
reports positions per second per mode. Exit code 1 on any difference. Compare machines by the absolute
positions/s of the same sample, not by the speed-up.

    python -u scripts/check_h6_fast.py --games data/external/renjunet/games.jsonl.gz --modes off fast2 fast3 \\
        --sample-every 500 --prove-losses-last-n 8 --workers 12 --report runs/h6/h6_fast_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h6_fast  # noqa: E402
from hybrid.h6_labels import default_probe_files, probe_positions  # noqa: E402
from scripts.build_h6_labels import _git_commit, build, read_games  # noqa: E402


def rows_digest(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row.to_dict(), separators=(',', ':'), sort_keys=True) + '\n').encode())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, required=True, help='H2 games.jsonl.gz')
    parser.add_argument('--modes', nargs='+', choices=h6_fast.MODES, default=list(h6_fast.MODES))
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--sample-every', type=int, default=500, help='1-in-N hash game sample (as the pilot)')
    parser.add_argument('--limit', type=int, help='first N games of the file, before sampling')
    parser.add_argument('--from-ply', type=int, default=5)
    parser.add_argument('--prove-losses-last-n', type=int, default=0, help='0 = every position (as the pilot)')
    parser.add_argument('--no-prove-losses', dest='prove_losses', action='store_false')
    parser.add_argument('--probe-game-min-ply', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--report', type=Path, required=True, help='under runs/ (counts and timings only)')
    args = parser.parse_args(argv)
    for mode in args.modes:  # fail before the long runs (e.g. fast3 not built)
        try:
            h6_fast.install(mode)
        except RuntimeError as error:
            parser.error(str(error))
    h6_fast.install('off')
    games = read_games(args.games)[:args.limit]
    probes = probe_positions(default_probe_files(ROOT), ROOT)
    runs, reference = {}, None
    for mode in args.modes:
        print(f'[check] --fast {mode}', flush=True)
        started = perf_counter()
        rows, summary = build(games, splits=set(args.splits), from_ply=args.from_ply, outcome_weight=0.1,
                              proof_weight=1.0, prove_losses=args.prove_losses, vct1=False, probes=probes,
                              workers=args.workers, probe_game_min_ply=args.probe_game_min_ply,
                              prove_losses_last_n=args.prove_losses_last_n, sample_every=args.sample_every,
                              fast=mode)
        seconds = perf_counter() - started
        result = {'seconds': round(seconds, 2), 'rows': len(rows), 'rows_sha256': rows_digest(rows),
                  'positions_per_second': round(len(rows) / seconds, 2),
                  'budget_cuts': summary['prove_losses_budget_cuts'], 'acceleration': h6_fast.provenance(mode)}
        if reference is None:
            reference = (mode, result['rows_sha256'], summary)
        result['same_as_' + reference[0]] = (result['rows_sha256'], summary) == reference[1:]
        runs[mode] = result
        print(f'[check] --fast {mode}: {seconds / 60:.1f} min, {len(rows):,} rows, '
              f'{result["positions_per_second"]:,} positions/s, same as {reference[0]}: '
              f'{result["same_as_" + reference[0]]}', flush=True)
    base = runs[args.modes[0]]['seconds']
    for result in runs.values():
        result['speedup_vs_' + args.modes[0]] = round(base / result['seconds'], 2)
    same = all(r['same_as_' + args.modes[0]] for r in runs.values())
    payload = {'format': 'h6-fast-check-v1', 'git_commit': _git_commit(), 'games': str(args.games),
               'machine': {'platform': platform.platform(), 'python': platform.python_version(),
                           'processor': platform.processor(), 'cpu_count': os.cpu_count()},
               'settings': {k: getattr(args, k) for k in ('modes', 'splits', 'sample_every', 'limit', 'from_ply',
                                                          'prove_losses', 'prove_losses_last_n',
                                                          'probe_game_min_ply', 'workers')},
               'sample': {'games_kept': reference[2]['games_kept'], 'rows': runs[args.modes[0]]['rows']},
               'same': same, 'runs': runs}
    print(json.dumps({'same': same, **{m: {k: r[k] for k in ('seconds', 'positions_per_second', 'budget_cuts')}
                                       for m, r in runs.items()}}, indent=1), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
