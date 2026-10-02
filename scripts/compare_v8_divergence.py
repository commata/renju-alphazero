"""Where two paired V8 benchmark runs part ways (docs/mcts-v8-teacher.md §12.7).

Two runs of ``run_mcts_v8_benchmark.py`` with the same ``--seed`` and ``--opponent``
play the same openings with the same random streams, so their games stay identical
until one engine decision differs. Comparing whole-game scores or per-move time
distributions mixes that one decision with everything that happens after it. This
script looks only at the first divergence of each paired game:

- which side moved there (a V8 decision, or the opponent, which would mean the
  runs are not really paired), the route, V7's move and the V8-C / V8-A diagnostics
  (statuses, rank, nodes, seconds, switch reason) of both runs;
- the VCT1 status of both played moves with a fresh, much larger solver budget
  (``--node-limit`` per VCF, ``--node-budget`` per check), so "veto kept a losing
  move" or "aggressive found a safer move" can be checked instead of inferred;
- the game results and lengths after the divergence.

Identical games are counted, with the V8 thinking time of both runs (timing noise).

    python scripts/compare_v8_divergence.py --a runs/v8_h1/full_vs_b.json \\
        --b runs/v8_h1/veto_vs_b.json --output runs/v8_h1/divergence_full_veto.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.mcts_v8 import _BudgetedSolver  # noqa: E402
from renju import Game  # noqa: E402


def _position(moves, ply) -> Game:
    game = Game()
    for move in moves[:ply]:
        game.play(*move)
    return game


def vct1_check(game, move, node_limit, node_budget) -> dict:
    solver = _BudgetedSolver(node_limit=node_limit, call_limit=10**9, node_budget=node_budget)
    started = perf_counter()
    status = solver.status_after(game, tuple(move))
    return {'status': status, 'exhausted': solver.exhausted, 'nodes': solver.nodes_used,
            'seconds': round(perf_counter() - started, 2)}


def _decision(record) -> dict:
    """The parts of a runner move record that explain the decision."""
    if record is None:
        return None
    module = record['root'] if record['route'] == 'tree' else record['vct']
    return {'route': record['route'], 'v7_move': record['v7_move'], 'changed': record['changed'],
            'seconds': record['seconds'], 'checked': module['checked'],
            'rank': record['root'].get('rank') if record['route'] == 'tree' else None,
            'switch': record['root'].get('switch') if record['route'] == 'tree' else None,
            'nodes': module['nodes'], 'calls': module['calls'], 'exhausted': module['exhausted']}


def first_divergence(a_game, b_game) -> int | None:
    for ply, (x, y) in enumerate(zip(a_game['moves'], b_game['moves'])):
        if x != y:
            return ply
    if len(a_game['moves']) != len(b_game['moves']):
        return min(len(a_game['moves']), len(b_game['moves']))
    return None


def compare(a, b, *, node_limit, node_budget, check=True) -> dict:
    if (a['seed'], a.get('opponent', 'v7')) != (b['seed'], b.get('opponent', 'v7')):
        raise ValueError('runs must use the same --seed and --opponent')
    b_games = {(g['pair'], g['v8_color']): g for g in b['games']}
    identical, rows = [], []
    for a_game in sorted(a['games'], key=lambda g: (g['pair'], g['v8_color'])):
        key = (a_game['pair'], a_game['v8_color'])
        b_game = b_games.get(key)
        if b_game is None:
            continue
        ply = first_divergence(a_game, b_game)
        if ply is None:
            identical.append({'pair': key[0], 'v8_color': key[1], 'result': a_game['result'],
                              'a_v8_seconds': round(sum(m['seconds'] for m in a_game['v8_moves']), 2),
                              'b_v8_seconds': round(sum(m['seconds'] for m in b_game['v8_moves']), 2)})
            continue
        a_rec = next((m for m in a_game['v8_moves'] if m['ply'] == ply), None)
        b_rec = next((m for m in b_game['v8_moves'] if m['ply'] == ply), None)
        row = {'pair': key[0], 'v8_color': key[1], 'ply': ply,
               'by': 'v8' if a_rec is not None and b_rec is not None else 'opponent',
               'a': {'move': a_game['moves'][ply] if ply < len(a_game['moves']) else None,
                     'result': a_game['result'], 'length': a_game['length'], 'decision': _decision(a_rec)},
               'b': {'move': b_game['moves'][ply] if ply < len(b_game['moves']) else None,
                     'result': b_game['result'], 'length': b_game['length'], 'decision': _decision(b_rec)}}
        if check and row['by'] == 'v8':
            game = _position(a_game['moves'], ply)
            for side in ('a', 'b'):
                row[side]['vct1_offline'] = vct1_check(game, row[side]['move'], node_limit, node_budget)
        rows.append(row)
        print(f"pair {key[0]} {key[1]} ply {ply} by {row['by']}: "
              f"a {row['a']['move']} {row['a'].get('vct1_offline', {}).get('status')} -> {row['a']['result']} | "
              f"b {row['b']['move']} {row['b'].get('vct1_offline', {}).get('status')} -> {row['b']['result']}",
              flush=True)
    return {'identical_games': len(identical), 'divergent_games': len(rows),
            'identical_v8_seconds': {'a': round(sum(r['a_v8_seconds'] for r in identical), 2),
                                     'b': round(sum(r['b_v8_seconds'] for r in identical), 2)},
            'identical': identical, 'divergences': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--a', type=Path, required=True, help='benchmark JSON (e.g. arm full)')
    parser.add_argument('--b', type=Path, required=True, help='paired benchmark JSON (same seed and opponent)')
    parser.add_argument('--node-limit', type=int, default=200_000, help='per-VCF node limit of the offline check')
    parser.add_argument('--node-budget', type=int, default=2_000_000, help='total VCF nodes per offline check')
    parser.add_argument('--no-check', action='store_true', help='structure only, no offline VCT1 checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    a = json.loads(args.a.read_text(encoding='utf-8'))
    b = json.loads(args.b.read_text(encoding='utf-8'))
    try:
        result = compare(a, b, node_limit=args.node_limit, node_budget=args.node_budget, check=not args.no_check)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {'format': 'mcts-v8-divergence-v1', 'a': str(args.a), 'a_arm': a['arm'], 'b': str(args.b),
               'b_arm': b['arm'], 'opponent': a.get('opponent', 'v7'), 'seed': a['seed'],
               'a_git_commit': a.get('git_commit'), 'b_git_commit': b.get('git_commit'),
               'node_limit': args.node_limit, 'node_budget': args.node_budget, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(json.dumps({k: payload[k] for k in ('a_arm', 'b_arm', 'identical_games', 'divergent_games',
                                              'identical_v8_seconds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
