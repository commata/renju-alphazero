"""Post-hoc analysis of V8-4 benchmark runs (docs/mcts-v8-teacher.md §11.11, §11.12).

Two analyses, both written to one JSON file so every claim in the design doc can be
re-checked from the stored positions:

1. matched counterfactual (``--baseline``): at every V8-B move (route ``own_vct``) of
   the main run, take the move the paired baseline game actually played in the same
   position (same opening, colour and seed, identical history up to that ply) and
   classify it as a VCT1 attack (WIN / REFUTED / UNKNOWN). Unlike the runner's
   ``--counterfactual`` record, which re-runs V7 with an independent random stream,
   this uses the baseline's real move.
2. loss analysis: for every lost game of the main run, the VCF and VCT1 status of
   V8's last ``--loss-plies`` moves (latest first), to find the first V8 move proven
   to lose and the route that chose it.

Every check uses a fresh ``_BudgetedSolver`` (per-VCF node limit ``--node-limit``,
total ``--node-budget``); a budget cut is reported as UNKNOWN with ``exhausted``.

    python scripts/analyze_v8_benchmark.py --run runs/v8_4/pilot_full.json \\
        --baseline runs/v8_4/pilot_a_only.json --output runs/v8_4/pilot_analysis.json
"""
from __future__ import annotations

import argparse
import hashlib
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


def _history_sha(moves, ply) -> str:
    return hashlib.sha256(json.dumps(moves[:ply], separators=(',', ':')).encode()).hexdigest()[:16]


def _check(game, move, depth_fn, args) -> dict:
    solver = _BudgetedSolver(node_limit=args.node_limit, call_limit=10**9, node_budget=args.node_budget)
    started = perf_counter()
    status = depth_fn(solver, game, tuple(move))
    return {'status': status, 'exhausted': solver.exhausted, 'nodes': solver.nodes_used,
            'calls': solver.vcf_calls, 'seconds': round(perf_counter() - started, 2)}


def _vcf(solver, game, move):
    return solver._bounded(game, move, None, None, lambda g: solver.after_move(g, 0)[0])


def _vct1(solver, game, move):
    return solver.status_after(game, move)


def _attack(solver, game, move):
    return solver.attack_status(game, move)


def matched_counterfactual(run, baseline, args) -> list[dict]:
    paired = {(g['pair'], g['v8_color']): g for g in baseline['games']}
    rows = []
    for game_record in run['games']:
        base = paired.get((game_record['pair'], game_record['v8_color']))
        for move in game_record['v8_moves']:
            if move['route'] != 'own_vct':
                continue
            ply = move['ply']
            moves = game_record['moves']
            row = {'pair': game_record['pair'], 'v8_color': game_record['v8_color'], 'ply': ply,
                   'history_sha16': _history_sha(moves, ply), 'v8_move': moves[ply],
                   'run_result': game_record['result']}
            if base is None or base['moves'][:ply] != moves[:ply] or len(base['moves']) <= ply:
                row['baseline'] = None  # not the same position in the baseline game
            else:
                baseline_move = base['moves'][ply]
                row['baseline'] = {'move': baseline_move, 'same_as_v8': baseline_move == moves[ply],
                                   'result': base['result'],
                                   'attack': _check(_position(moves, ply), baseline_move, _attack, args)}
            recorded = move.get('counterfactual')
            if recorded:
                row['recorded_counterfactual'] = recorded
            rows.append(row)
            print(f"cf pair {row['pair']} {row['v8_color']} ply {ply}: v8 {row['v8_move']} "
                  f"baseline {row['baseline'] and row['baseline']['move']} "
                  f"{row['baseline'] and row['baseline']['attack']}", flush=True)
    return rows


def loss_analysis(run, args) -> list[dict]:
    rows = []
    for game_record in run['games']:
        if game_record['result'] != 'loss':
            continue
        moves = game_record['moves']
        by_ply = {m['ply']: m for m in game_record['v8_moves']}
        checked = []
        for ply in sorted(by_ply, reverse=True)[:args.loss_plies]:
            game = _position(moves, ply)
            vcf = _check(game, moves[ply], _vcf, args)
            vct1 = _check(game, moves[ply], _vct1, args) if vcf['status'] == 'SAFE' else None
            record = by_ply[ply]
            checked.append({'ply': ply, 'history_sha16': _history_sha(moves, ply), 'move': moves[ply],
                            'route': record['route'], 'changed': record['changed'],
                            'vct_exhausted': record['vct']['exhausted'],
                            'attack_exhausted': record['attack']['exhausted'],
                            'vcf': vcf, 'vct1': vct1})
            print(f"loss pair {game_record['pair']} {game_record['v8_color']} ply {ply} {record['route']} "
                  f"VCF {vcf['status']} VCT1 {vct1 and vct1['status']}", flush=True)
        first_loss = None
        for item in reversed(checked):  # earliest first
            final = item['vct1']['status'] if item['vct1'] else item['vcf']['status']
            if final == 'UNSAFE':
                first_loss = item['ply']
                break
        rows.append({'pair': game_record['pair'], 'v8_color': game_record['v8_color'],
                     'length': game_record['length'], 'first_proven_losing_v8_ply': first_loss,
                     'checked': checked})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True, help='benchmark JSON to analyse (usually arm full)')
    parser.add_argument('--baseline', type=Path, help='paired benchmark JSON (same seed), e.g. arm a_only')
    parser.add_argument('--loss-plies', type=int, default=8, help="V8's last N moves checked in each loss")
    parser.add_argument('--node-limit', type=int, default=100_000, help='per-VCF node limit')
    parser.add_argument('--node-budget', type=int, default=2_000_000, help='total VCF nodes per check')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run = json.loads(args.run.read_text(encoding='utf-8'))
    payload = {'format': 'mcts-v8-benchmark-analysis-v1', 'run': str(args.run),
               'run_arm': run['arm'], 'opponent': run.get('opponent', 'v7'), 'run_git_commit': run.get('git_commit'), 'seed': run['seed'],
               'node_limit': args.node_limit, 'node_budget': args.node_budget}
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
        if baseline['seed'] != run['seed']:
            parser.error('baseline must use the same --seed (paired openings)')
        if baseline.get('opponent', 'v7') != run.get('opponent', 'v7'):
            parser.error('baseline must use the same --opponent')
        payload['baseline'] = str(args.baseline)
        payload['baseline_arm'] = baseline['arm']
        rows = matched_counterfactual(run, baseline, args)
        statuses = [r['baseline']['attack']['status'] for r in rows if r['baseline']]
        payload['matched_counterfactual'] = {
            'v8_b_moves': len(rows),
            'baseline_same_position': len(statuses),
            'baseline_same_move': sum(r['baseline']['same_as_v8'] for r in rows if r['baseline']),
            'baseline_move_attack_status': {s: statuses.count(s) for s in ('WIN', 'REFUTED', 'UNKNOWN')},
            'rows': rows,
        }
    payload['losses'] = loss_analysis(run, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.get('matched_counterfactual', {}).items() if k != 'rows'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
