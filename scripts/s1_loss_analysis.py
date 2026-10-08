"""S1: VCT2-like horizon signature frequency and loss-cause classification (docs/mcts-v8-teacher.md §12.19).

Analysis only, on finished ``run_mcts_v8_benchmark.py`` result files (``--output`` JSON).

Signature (``VCT2-like horizon signature``): a V8 tree move whose played move V8-C
rated SAFE, followed by V8's next move on route stage4 whose every checked defence is
UNSAFE. It only sees losses that surface on the very next move, so it is a lower bound,
not a VCT2 rate. Reported per game and per opening (the two colours of one opening
pair count once; openings of different seeds are different).

Loss classification (``--classify``): for every lost game of the chosen arms, V8's
moves are walked back from the end. After each V8 move the position is checked with
the bounded ``ThreatSolver`` (``analysis.mcts_v8._BudgetedSolver``) at depth 0, 1, 2
in turn; the walk stops at the first move that is not proven lost within depth 2.
The decisive move is the earliest move of that proven-lost run. Budgets are node and
call counts only (no time cut), so reruns give the same result.

primary (one per loss):
    VCT2_HORIZON          decisive move lost at depth 2 (not at depth <= 1)
    VCT1_LOSS             decisive move lost at depth <= 1
    DEEPER_OR_POSITIONAL  the decisive move was forced (a stage-2 block of a five, or V8's
                          own record shows every checked alternative UNSAFE), so the game
                          was lost earlier, deeper than depth 2
    UNRESOLVED            even V8's last move is not proven lost (budget)
flags (any number): ENGINE_SAID_SAFE, ENGINE_UNKNOWN, BOUNDARY_UNKNOWN, ROOT_UNKNOWN,
    ROOT_BUDGET_EXHAUSTED, STAGE_BUDGET_EXHAUSTED, OPP_OWN_VCT, V8C_SWITCHED, LONG_GAME,
    SIGNATURE
causal_status (§12.21): CONFIRMED when the V8 move before the lost run is SAFE (or the run
    reaches V8's first move), so the decisive move is the first losing transition;
    TENTATIVE when that move is UNKNOWN (the first losing move may be earlier);
    NOT_APPLICABLE for DEEPER_OR_POSITIONAL / UNRESOLVED.

``--reuse-classification`` rebuilds the derived fields and totals from an earlier output
(its walks) without re-running the solver; ``--witness-output`` writes the VCT2 positions
(the position before each decisive move) as the S2 test set.

    python scripts/s1_loss_analysis.py docs/mcts-v8-results/h5_*_8401.json runs/h5/*_8402.json \\
        --classify puct_policy --workers 4 --output runs/s1/losses.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.mcts_v8 import _BudgetedSolver  # noqa: E402
from analysis.threats import SAFE, UNKNOWN, UNSAFE  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402

FORMAT = 's1-loss-analysis-v2'  # v2: causal_status, boundary_status, input_sha256
LONG_GAME = 150


def _status_of(checked, move):
    return dict((tuple(m), s) for m, s in checked).get(tuple(move))


def engine_status(record: dict) -> str | None:
    """V8's own status of the move it played (V8-C on the tree route, V8-A on stage 4/5)."""
    played = record['played']
    if record['route'] == 'tree':
        return _status_of(record['root']['checked'], played)
    if record['route'] in ('stage4', 'stage5'):
        return _status_of(record['vct']['checked'], played)
    return None


def forced_loss(record: dict) -> bool:
    """The move had no surviving alternative: a five block, or every checked move UNSAFE."""
    if record['route'] == 'stage2':
        return True
    checked = record['root']['checked'] + record['vct']['checked']
    return bool(checked) and all(s == UNSAFE for _, s in checked)


def signature_ply(game: dict) -> int | None:
    """Ply of the first tree move (SAFE per V8-C) whose next V8 move is a lost stage-4 defence."""
    moves = game['v8_moves']
    for a, b in zip(moves, moves[1:]):
        if (a['route'] == 'tree' and engine_status(a) == SAFE and b['route'] == 'stage4'
                and b['vct']['checked'] and all(s == UNSAFE for _, s in b['vct']['checked'])):
            return a['ply']
    return None


def with_played(result: dict) -> list[dict]:
    games = result['games']
    for game in games:
        for m in game['v8_moves']:
            if 'played' not in m:
                m['played'] = game['moves'][m['ply']]
    return games


def signature_table(results: list[dict]) -> dict:
    rows = {}
    for result in results:
        row = rows.setdefault(result['arm'], {'games': 0, 'affected_games': 0, 'openings': set(),
                                              'affected_openings': set(), 'hits': []})
        for game in with_played(result):
            opening = (result['seed'], game['pair'])
            row['games'] += 1
            row['openings'].add(opening)
            ply = signature_ply(game)
            if ply is not None:
                row['affected_games'] += 1
                row['affected_openings'].add(opening)
                row['hits'].append({'seed': result['seed'], 'pair': game['pair'], 'v8_color': game['v8_color'],
                                    'ply': ply, 'result': game['result']})
    return {arm: {'games': r['games'], 'affected_games': r['affected_games'],
                  'game_rate': round(r['affected_games'] / r['games'], 4) if r['games'] else None,
                  'unique_openings': len(r['openings']), 'affected_openings': len(r['affected_openings']),
                  'opening_rate': round(len(r['affected_openings']) / len(r['openings']), 4) if r['openings'] else None,
                  'hits': r['hits']}
            for arm, r in rows.items()}


def primary_cause(walk: list[dict]) -> tuple[str, dict | None]:
    """``walk``: V8 moves from the last one backwards, each {'depth': d | None, 'status', 'record'}.

    ``depth`` is the smallest depth (0-2) at which the position after the move is proven lost,
    None when it is not. Returns the primary cause and the decisive entry.
    """
    run = []
    for entry in walk:
        if entry['depth'] is None:
            break
        run.append(entry)
    if not run:
        return 'UNRESOLVED', None
    decisive = run[-1]
    if forced_loss(decisive['record']):
        return 'DEEPER_OR_POSITIONAL', decisive
    return ('VCT2_HORIZON' if decisive['depth'] == 2 else 'VCT1_LOSS'), decisive


def lost_depth_info(moves, ply, budget) -> tuple[int | None, str, dict]:
    """Smallest depth <= 2 at which the side that played ``moves[ply]`` is proven lost after it.

    ``info`` says why a result is UNKNOWN: ``budget_exhausted`` (the move's node or call
    budget ran out) and/or ``vcf_cut_calls`` (single VCF calls cut at ``node_limit``; such a
    call is UNKNOWN even when the total budget is not used up).
    """
    game = Game()
    for move in moves[:ply]:
        game.play(*move)
    solver = _BudgetedSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    depth_found, worst = None, SAFE
    for depth in (0, 1, 2):
        status = solver._bounded(game, tuple(moves[ply]), None, None,
                                 lambda g, d=depth: solver.after_move(g, d)[0])
        if status == UNSAFE:
            depth_found, worst = depth, UNSAFE
            break
        if status == UNKNOWN:
            worst = UNKNOWN
            break  # a deeper search cannot finish inside a budget the shallower one ran out of
    info = {'budget_exhausted': solver.exhausted, 'vcf_cut_calls': solver.vcf_exhausted,
            'vcf_calls': solver.vcf_calls, 'nodes_used': solver.nodes_used}
    return depth_found, worst, info


def _lost_depth(moves, ply, budget) -> tuple[int | None, str]:
    depth, status, _ = lost_depth_info(moves, ply, budget)
    return depth, status


def causal_fields(primary: str, walk: list[dict]) -> dict:
    """Whether the decisive move is the first losing transition (from the walk alone)."""
    boundary = walk[-1]['status'] if walk and walk[-1]['depth'] is None else 'START'
    if primary not in ('VCT2_HORIZON', 'VCT1_LOSS'):
        causal = 'NOT_APPLICABLE'
    else:
        causal = 'TENTATIVE' if boundary == UNKNOWN else 'CONFIRMED'
    return {'boundary_status': boundary, 'causal_status': causal}


def classify_game(args) -> dict:
    game, budget, seed, arm = args
    walk = []
    for record in reversed(game['v8_moves']):
        depth, status = _lost_depth(game['moves'], record['ply'], budget)
        walk.append({'ply': record['ply'], 'depth': depth, 'status': status, 'record': record})
        if depth is None:
            break
    cause, decisive = primary_cause(walk)
    moves = game['v8_moves']
    flags = set()
    if decisive is not None:
        said = engine_status(decisive['record'])
        if said == SAFE:
            flags.add('ENGINE_SAID_SAFE')
        elif said == UNKNOWN:
            flags.add('ENGINE_UNKNOWN')
        if walk[-1]['depth'] is None and walk[-1]['status'] == UNKNOWN:
            flags.add('BOUNDARY_UNKNOWN')  # the move before the lost run is unresolved, not SAFE
    if any(m['route'] == 'tree' and engine_status(m) == UNKNOWN for m in moves):
        flags.add('ROOT_UNKNOWN')
    if any(m['root']['exhausted'] for m in moves):
        flags.add('ROOT_BUDGET_EXHAUSTED')
    if any(m['vct']['exhausted'] for m in moves):
        flags.add('STAGE_BUDGET_EXHAUSTED')
    if game.get('opponent_routes', {}).get('own_vct'):
        flags.add('OPP_OWN_VCT')
    if any(m['root'].get('switch') for m in moves):
        flags.add('V8C_SWITCHED')
    if game['length'] >= LONG_GAME:
        flags.add('LONG_GAME')
    if signature_ply(game) is not None:
        flags.add('SIGNATURE')
    return {
        'arm': arm, 'seed': seed, 'pair': game['pair'], 'v8_color': game['v8_color'], 'length': game['length'],
        'primary': cause, 'flags': sorted(flags),
        'decisive_ply': None if decisive is None else decisive['ply'],
        'decisive_move': None if decisive is None else [decisive['record']['played'][0] + 1,
                                                        decisive['record']['played'][1] + 1],
        'decisive_depth': None if decisive is None else decisive['depth'],
        'decisive_route': None if decisive is None else decisive['record']['route'],
        'walk': [{'ply': e['ply'], 'depth': e['depth'], 'status': e['status']} for e in walk],
        **causal_fields(cause, walk),
    }


def classify(results: list[dict], arms: set[str], budget: dict, workers: int) -> dict:
    jobs = [(g, budget, r['seed'], r['arm']) for r in results if r['arm'] in arms
            for g in with_played(r) if g['result'] == 'loss']
    if workers == 1:
        rows = [classify_game(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(classify_game, jobs))
    return aggregate(rows)


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        entry = out.setdefault(row['arm'], {'losses': 0, 'primary': {}, 'flags': {}, 'games': [],
                                     'vct2_openings': set()})
        entry['losses'] += 1
        entry['primary'][row['primary']] = entry['primary'].get(row['primary'], 0) + 1
        for flag in row['flags']:
            entry['flags'][flag] = entry['flags'].get(flag, 0) + 1
        if row['primary'] == 'VCT2_HORIZON':
            entry['vct2_openings'].add((row['seed'], row['pair']))
        entry['games'].append(row)
    for entry in out.values():
        entry['vct2_horizon_losses'] = entry['primary'].get('VCT2_HORIZON', 0)
        vct2 = [g for g in entry['games'] if g['primary'] == 'VCT2_HORIZON']
        entry['vct2_confirmed'] = sum(g['causal_status'] == 'CONFIRMED' for g in vct2)
        entry['vct2_tentative'] = sum(g['causal_status'] == 'TENTATIVE' for g in vct2)
        entry['vct2_repeated_opening'] = entry['vct2_horizon_losses'] > len(entry['vct2_openings'])
        entry['vct2_openings'] = sorted(entry['vct2_openings'])
    return out


def witness_set(results: list[dict], classification: dict) -> list[dict]:
    """Positions before each VCT2_HORIZON decisive move: V8 to move, the decisive move lost at depth 2."""
    games = {(r['arm'], r['seed'], g['pair'], g['v8_color']): g for r in results for g in with_played(r)}
    out = []
    for arm, entry in sorted(classification.items()):
        for row in entry['games']:
            if row['primary'] != 'VCT2_HORIZON':
                continue
            game = games[(arm, row['seed'], row['pair'], row['v8_color'])]
            record = next(m for m in game['v8_moves'] if m['ply'] == row['decisive_ply'])
            out.append({'arm': arm, 'seed': row['seed'], 'pair': row['pair'], 'v8_color': row['v8_color'],
                        'ply': row['decisive_ply'], 'moves': game['moves'][:row['decisive_ply']],
                        'decisive_move_1idx': row['decisive_move'], 'decisive_route': row['decisive_route'],
                        'engine_status': engine_status(record), 'causal_status': row['causal_status'],
                        'boundary_status': row['boundary_status']})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--classify', nargs='*', default=[], help='arms whose losses are classified')
    parser.add_argument('--reuse-classification', type=Path,
                        help='take the walks from an earlier output instead of re-running the solver')
    parser.add_argument('--node-limit', type=int, default=20_000, help='per VCF call')
    parser.add_argument('--call-limit', type=int, default=50_000, help='VCF calls per checked move')
    parser.add_argument('--node-budget', type=int, default=3_000_000, help='VCF nodes per checked move')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--witness-output', type=Path, help='S2 test set: positions of the VCT2 losses')
    args = parser.parse_args(argv)
    results = [json.loads(p.read_text(encoding='utf-8')) for p in args.results]
    budget = {'node_limit': args.node_limit, 'call_limit': args.call_limit, 'node_budget': args.node_budget}
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'inputs': [str(p) for p in args.results],
               'input_sha256': {str(p): file_sha256(p) for p in args.results},
               'budget': budget, 'signature': signature_table(results)}
    if args.reuse_classification is not None:
        earlier = json.loads(args.reuse_classification.read_text(encoding='utf-8'))
        payload['budget'] = earlier['budget']  # the walks were computed with that budget
        payload['reused_from'] = {'path': str(args.reuse_classification), 'git_commit': earlier.get('git_commit'),
                                  'sha256': file_sha256(args.reuse_classification)}
        rows = [{**row, **causal_fields(row['primary'], row['walk'])}
                for entry in earlier.get('classification', {}).values() for row in entry['games']]
        payload['classification'] = aggregate(rows)
    elif args.classify:
        payload['classification'] = classify(results, set(args.classify), budget, args.workers)
    print(json.dumps({'signature': {a: {k: v for k, v in r.items() if k != 'hits'}
                                    for a, r in payload['signature'].items()},
                      'classification': {a: {k: v for k, v in r.items() if k != 'games'}
                                         for a, r in payload.get('classification', {}).items()}},
                     indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    if args.witness_output is not None:
        witnesses = witness_set(results, payload.get('classification', {}))
        args.witness_output.parent.mkdir(parents=True, exist_ok=True)
        args.witness_output.write_text(json.dumps({'format': 's2-witness-set-v1', 'source': payload['inputs'],
                                                   'positions': witnesses}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
