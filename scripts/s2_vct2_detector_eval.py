"""S2-3: evaluate the selective VCT2 detector on fixed sets (docs/mcts-v8-teacher.md §12.23).

Sets (fixed before measuring):
    A  core        the two CONFIRMED witness positions: the decisive move (proven lost at
                   depth 2) must be PROVEN_LOSS; policy alternatives use their full-solver
                   truth (s2_witness_alternatives.json): a move with no loss within depth 2
                   must not be PROVEN_LOSS
    B  exploratory the three TENTATIVE witness positions (decisive move), reported apart
    C  stress      P92: the 129 moves proven lost at depth 2 (recall) and the 4 UNKNOWN moves
                   (the detector must not claim more than the full solver could verify)
    D  control     tree moves V8 played in won H5 puct_policy games (evaluation only, never
                   training): intervention rate and cost

Every depth-2 PROVEN_LOSS is re-checked with ``verify_witness`` (full solver along the
witness threat); a False is a soundness violation. Items are written to ``--jsonl`` as they
finish (resume) and computed in ``--workers`` processes.

Gate (fixed): A's decisive moves (the moves V8 played) all detected, no soundness violation, D cost median <= 2 s and p95 <= 10 s.
The D cost is reported twice: the whole check and the selective stage alone (the depth 0-1
stage is the class V8-C already checks on the tree route, so the selective stage is what
the detector adds there); the gate is evaluated on both (``D_cost_ok_total`` /
``D_cost_ok_added``). ``--selective-node-budget`` changes the depth-2 budget for a sweep.

    python scripts/s2_vct2_detector_eval.py --sets A B C D --workers 8 \\
        --jsonl runs/s2/detector.jsonl --output runs/s2/detector_eval.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from random import Random
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.selective_vct import DEFAULT_BUDGET, PROVEN_LOSS, classify, verify_witness  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402

RESULTS = ROOT / 'docs' / 'mcts-v8-results'
FORMAT = 's2-vct2-detector-eval-v1'


def _game(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def build_items(sets, control: int, control_seed: int) -> list[dict]:
    load = lambda name: json.loads((RESULTS / name).read_text(encoding='utf-8'))  # noqa: E731
    items = []
    if {'A', 'B'} & set(sets):
        witness = load('s2_witness_positions.json')['positions']
        alternatives = {(p['seed'], p['pair']): p['checks'] for p in load('s2_witness_alternatives.json')['positions']}
        for pos in witness:
            name = 'A' if pos['causal_status'] == 'CONFIRMED' else 'B'
            if name not in sets:
                continue
            base = f"{pos['seed']}-p{pos['pair']}-{pos['v8_color']}"
            checks = alternatives.get((pos['seed'], pos['pair']))
            if checks is None:  # B: only the decisive move has a known truth
                checks = [{'move': pos['decisive_move_1idx'], 'status': 'PROVEN_LOSS', 'played': True}]
            for check in checks:
                move = check['move']
                items.append({'key': f"{name}/{base}/{move[0]},{move[1]}", 'set': name, 'moves': pos['moves'],
                              'move': [move[0] - 1, move[1] - 1],
                              'expect': 'PROVEN_LOSS' if check['status'] == 'PROVEN_LOSS' else 'NOT_PROVEN_LOSS',
                              'meta': {'played': check.get('played'), 'policy_rank': check.get('policy_rank')}})
    if 'C' in sets:
        probes = load('probe_web_v8_loss_20261007.json')
        history = probes['moves'][:probes['probes']['P92']['plies_played']]
        for row in load('s2_p92_joined.json')['moves']:  # policy order: the likeliest moves first
            move = row['move']
            items.append({'key': f"C/P92/{move[0]},{move[1]}", 'set': 'C', 'moves': history,
                          'move': [move[0] - 1, move[1] - 1],
                          'expect': 'PROVEN_LOSS' if row['status'] == 'PROVEN_LOSS' else None,
                          'meta': {'truth': row['status'], 'policy_rank': row['policy_rank']}})
    if 'D' in sets:
        pool = []
        for seed in (8401, 8402):
            result = load(f'h5_policy_{seed}.json')
            for game in result['games']:
                if game['result'] != 'win':
                    continue
                for record in game['v8_moves']:
                    if record['route'] == 'tree':
                        pool.append((seed, game['pair'], game['v8_color'], record['ply'], game['moves']))
        for seed, pair, color, ply, moves in Random(control_seed).sample(pool, min(control, len(pool))):
            items.append({'key': f"D/{seed}-p{pair}-{color}/{ply}", 'set': 'D', 'moves': moves[:ply],
                          'move': moves[ply], 'expect': None, 'meta': {'ply': ply}})
    return items


def evaluate(task) -> dict:
    item, budget, verify = task
    game = _game(item['moves'])
    move = tuple(item['move'])
    result = classify(game, move, budget)
    verified = None
    if verify and result['status'] == PROVEN_LOSS:
        started = perf_counter()
        verified = verify_witness(game, move, result)
        result['verify_seconds'] = round(perf_counter() - started, 3)
    return {'key': item['key'], 'set': item['set'], 'move': [move[0] + 1, move[1] + 1], 'expect': item['expect'],
            'meta': item['meta'], 'result': result, 'verified': verified, 'budget': budget}


def _dist(values):
    values = sorted(values)
    if not values:
        return None
    return {'n': len(values), 'median': round(median(values), 3),
            'p95': values[min(len(values) - 1, int(0.95 * len(values)))], 'max': values[-1]}


def _cost_ok(dist) -> bool:
    return bool(dist) and dist['median'] <= 2 and dist['p95'] <= 10


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r['set'] for r in rows}):
        part = [r for r in rows if r['set'] == name]
        status = {}
        for r in part:
            key = r['result']['status'] if r['result']['status'] != 'UNKNOWN' else f"UNKNOWN:{r['result'].get('cause')}"
            status[key] = status.get(key, 0) + 1
        positives = [r for r in part if r['expect'] == 'PROVEN_LOSS']
        negatives = [r for r in part if r['expect'] == 'NOT_PROVEN_LOSS']
        out[name] = {
            'items': len(part), 'status': status,
            'recall': (f"{sum(r['result']['status'] == PROVEN_LOSS for r in positives)}/{len(positives)}"
                       if positives else None),
            'flagged_expected_not_lost': sum(r['result']['status'] == PROVEN_LOSS for r in negatives),
            'flagged': sum(r['result']['status'] == PROVEN_LOSS for r in part),
            'verify': {str(v): sum(r['verified'] is v for r in part if r['result']['status'] == PROVEN_LOSS)
                       for v in (True, False, None)},
            'seconds': _dist([r['result']['seconds'] for r in part]),
            'selective_seconds': _dist([r['result'].get('stage_seconds', {}).get('selective', 0.0) for r in part]),
        }
    violations = [r['key'] for r in rows if r['verified'] is False
                  or (r['expect'] == 'NOT_PROVEN_LOSS' and r['result']['status'] == PROVEN_LOSS)]
    decisive = [r for r in rows if r['set'] == 'A' and (r.get('meta') or {}).get('played')]
    if 'A' in out:
        out['A']['decisive_recall'] = f"{sum(r['result']['status'] == PROVEN_LOSS for r in decisive)}/{len(decisive)}"
    # The gate asks for the decisive moves V8 actually played (§12.23); other A positives are reported.
    gate = {'A_all_detected': bool(decisive) and all(r['result']['status'] == PROVEN_LOSS for r in decisive),
            'soundness_violations': violations,
            'D_cost_ok_total': _cost_ok(out.get('D', {}).get('seconds')),
            'D_cost_ok_added': _cost_ok(out.get('D', {}).get('selective_seconds'))}
    gate['pass_total'] = gate['A_all_detected'] and not violations and gate['D_cost_ok_total']
    gate['pass_added'] = gate['A_all_detected'] and not violations and gate['D_cost_ok_added']
    return {'sets': out, 'gate': gate}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sets', nargs='+', default=['A', 'B', 'C', 'D'], choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--control', type=int, default=100, help='D: number of sampled tree moves')
    parser.add_argument('--control-seed', type=int, default=2301)
    parser.add_argument('--limit', type=int, help='smoke tests: first N items only')
    parser.add_argument('--no-verify', action='store_true')
    parser.add_argument('--selective-node-budget', type=int, help='depth-2 node budget (sweep); default '
                        f"{DEFAULT_BUDGET['selective']['node_budget']}")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    items = build_items(args.sets, args.control, args.control_seed)[:args.limit]
    budget = {**DEFAULT_BUDGET}
    if args.selective_node_budget:
        budget['selective'] = {**DEFAULT_BUDGET['selective'], 'node_budget': args.selective_node_budget}
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: recomputed
            if row.get('budget') != budget:
                parser.error(f'{args.jsonl} holds results of another budget; use another --jsonl')
            done[row['key']] = row
    pending = [i for i in items if i['key'] not in done]
    print(f'items={len(items)} done={len(items) - len(pending)} pending={len(pending)} workers={args.workers}',
          flush=True)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(row):
        done[row['key']] = row
        r = row['result']
        print(f"{len(done)}/{len(items)} {row['key']} {r['status']} depth={r.get('depth')} "
              f"{r['seconds']:.1f}s verified={row['verified']}", flush=True)
        if args.jsonl is not None:
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')

    tasks = [(item, budget, not args.no_verify) for item in pending]
    if args.workers == 1:
        for task in tasks:
            keep(evaluate(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(evaluate, t) for t in tasks]):
                keep(future.result())
    rows = [done[i['key']] for i in items]
    summary = summarize(rows)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'budget': budget, 'sets': args.sets,
               'control': args.control, 'control_seed': args.control_seed, 'restricted': args.limit is not None,
               'inputs': {name: file_sha256(RESULTS / name) for name in (
                   's2_witness_positions.json', 's2_witness_alternatives.json', 's2_p92_joined.json',
                   'probe_web_v8_loss_20261007.json', 'h5_policy_8401.json', 'h5_policy_8402.json')},
               'summary': summary, 'items': rows}
    print(json.dumps(summary, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
