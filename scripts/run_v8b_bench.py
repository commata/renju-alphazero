"""V8-B (own VCT1 attack) benchmark on the E3-A records: win recall against time (docs/web-ser5.md §3.8).

V8-B runs before the tree and uses no randomness and no policy, so the records are the ground truth
of TB-Engine-v1's own V8-B:

    positive       a recorded own_vct move: TB1 proved a VCT1 win there (the winning move is recorded)
    hard negative  V8-B ran for more than --neg-seconds without a win (route not own_vct)

Each position runs only the RT30 V8-B (``rt30.engine._own_vct_attack``, fast windows, TB1 budgets)
with a deadline of --cap seconds (none: --cap 0), for every --order / --scheduler combination given.
Reported per combination: recall (a WIN proven; any proven win counts, ``same_move`` says whether it
is the recorded one), seconds on positives and negatives, timeouts. A WIN is a proof, so a found win
is never wrong; the cost of a variant shows on the negatives (time spent finding nothing).

    python scripts/run_v8b_bench.py --cap 4 --order frozen threes_first --scheduler fair widen `
        --workers 1 --output runs/rt30/v8b_cap4.json 2>&1 | Tee-Object runs/rt30/v8b_cap4.log
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid.progress import Progress, run_chunks  # noqa: E402

E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
DEFAULT_JSONL = [E3 / 'puct_policy_vct2_50k_8415.jsonl', E3 / 'puct_policy_vct2_50k_8416.jsonl']
TB1_ATTACK = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 200_000}


def load_sets(paths, neg_seconds: float, neg_limit: int | None):
    positives, negatives = [], []
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle):
                record = json.loads(line)
                for move in record['v8_moves']:
                    attack = move['attack']
                    row = {'key': f"{Path(path).stem}/{line_no}/{move['ply']}", 'moves': record['moves'][:move['ply']],
                           'recorded_attack_seconds': attack['seconds'], 'recorded_candidates': attack['candidates'],
                           'recorded_rank': attack['rank']}
                    if move['route'] == 'own_vct':
                        positives.append({**row, 'kind': 'positive', 'win': record['moves'][move['ply']]})
                    elif attack['seconds'] > neg_seconds and attack['status'] != 'WIN':
                        negatives.append({**row, 'kind': 'negative', 'win': None})
    negatives.sort(key=lambda r: -r['recorded_attack_seconds'])
    return positives, negatives[:neg_limit] if neg_limit else negatives


def run_one(task):
    from analysis.mcts_v8 import SearchDiagnostics
    from renju import Game
    from rt30 import fast
    from rt30.deadline import Deadline
    from rt30.engine import _own_vct_attack
    from search.mcts_v5 import _RootContext

    fast.install_all()
    row, combos, cap = task['row'], task['combos'], task['cap']
    game = Game()
    for move in row['moves']:
        game.play(*move)
    out = {'key': row['key'], 'kind': row['kind'], 'win': row['win'], 'recorded_rank': row['recorded_rank'],
           'recorded_attack_seconds': row['recorded_attack_seconds'], 'runs': {}}
    for order, scheduler in combos:
        diag = SearchDiagnostics()
        context = _RootContext(game.legal_moves(), diag)
        started = perf_counter()
        move, timed_out = _own_vct_attack(game, context, diag, **TB1_ATTACK,
                                          deadline=Deadline.after(cap) if cap else None,
                                          order=order, scheduler=scheduler)
        out['runs'][f'{order}/{scheduler}'] = {
            'seconds': round(perf_counter() - started, 4), 'won': move is not None,
            'move': list(move) if move else None, 'same_move': bool(move) and list(move) == row['win'],
            'timed_out': timed_out, 'nodes': diag.v8_attack_nodes, 'calls': diag.v8_attack_calls,
            'rank': diag.v8_attack_rank, 'candidates': diag.v8_attack_candidates}
    return out


def _dist(values):
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {'n': len(values), 'median': round(statistics.median(values), 3), 'p95': round(pick(0.95), 3),
            'max': round(values[-1], 3), 'sum': round(sum(values), 1)}


def summarize(rows, combos):
    summary = {'positives': sum(r['kind'] == 'positive' for r in rows),
               'negatives': sum(r['kind'] == 'negative' for r in rows), 'combos': {}}
    for order, scheduler in combos:
        name = f'{order}/{scheduler}'
        pos = [r['runs'][name] for r in rows if r['kind'] == 'positive']
        neg = [r['runs'][name] for r in rows if r['kind'] == 'negative']
        summary['combos'][name] = {
            'recall': f"{sum(x['won'] for x in pos)}/{len(pos)}", 'won': sum(x['won'] for x in pos),
            'same_move': sum(x['same_move'] for x in pos),
            'positive_seconds': _dist([x['seconds'] for x in pos]),
            'positive_seconds_when_won': _dist([x['seconds'] for x in pos if x['won']]),
            'negative_seconds': _dist([x['seconds'] for x in neg]),
            'timeouts': {'positive': sum(x['timed_out'] for x in pos), 'negative': sum(x['timed_out'] for x in neg)},
            'false_wins_on_negatives': sum(x['won'] for x in neg),  # proofs: a win the records missed, not an error
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jsonl', type=Path, nargs='+', default=DEFAULT_JSONL)
    parser.add_argument('--cap', type=float, default=4.0, help='V8-B deadline in seconds (0: none)')
    parser.add_argument('--order', nargs='+', default=['frozen'], choices=('frozen', 'threes_first'))
    parser.add_argument('--scheduler', nargs='+', default=['fair'], choices=('fair', 'widen', 'widen3'))
    parser.add_argument('--neg-seconds', type=float, default=5.0, help='hard negatives: recorded V8-B seconds above this')
    parser.add_argument('--neg-limit', type=int, help='at most this many hard negatives (the slowest first)')
    parser.add_argument('--positives-only', action='store_true')
    parser.add_argument('--limit', type=int, help='at most this many positions of each set (smoke runs)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    positives, negatives = load_sets(args.jsonl, args.neg_seconds, args.neg_limit)
    if args.positives_only:
        negatives = []
    if args.limit:
        positives, negatives = positives[:args.limit], negatives[:args.limit]
    combos = list(itertools.product(args.order, args.scheduler))
    rows_in = positives + negatives
    print(f'positives={len(positives)} negatives={len(negatives)} combos={len(combos)} cap={args.cap} '
          f'workers={args.workers}', flush=True)
    progress = Progress('v8b-bench', len(rows_in), 'positions')
    tasks = [{'row': row, 'combos': combos, 'cap': args.cap} for row in rows_in]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = run_chunks(pool, run_one, tasks, progress, lambda t: 1, unit_name='positions')
    summary = summarize(rows, combos)
    report = {'format': 'v8b-bench-v1', 'cap': args.cap, 'combos': [list(c) for c in combos],
              'neg_seconds': args.neg_seconds, 'workers': args.workers, 'jsonl': [str(p) for p in args.jsonl],
              'machine': {'platform': platform.platform(), 'cpu_count': os.cpu_count()},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
