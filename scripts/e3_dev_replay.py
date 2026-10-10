"""E3 dev replay: the E3 arms on the known failure positions, replacement included (§12.32).

Dev data only (E2 seeds 8413/8414, E1-dev, E1-S). The E3 judgment seeds 8415/8416 are never read.
For each case the arm's engine is run on the position and its whole VCT2 decision is recorded:
did it prove the played move lost, which move did it play instead, and is that move really safe?

Cases (``--cases``, default all):
    e2-tree   E2 games lost right after an opponent ``own_vct2`` WIN whose previous move of ours was a
              tree move: arm E3-A. The game is replayed from its opening with the game's own engine seed,
              calling the agent at every move of ours so its random stream matches the E2 game; the
              recorded moves are played, so the tree at the target move is the one the E2 game had.
              ``consistent`` says whether the tree's move there equals the E2 move.
    e2-stage  the same games when that move came from Stage 4/5: arm E3-S (stage routes use no random,
              so the position is run directly).
    e1-tree   E1-dev DETECT_MISS runs: arm E3-A with the E1 run seed on the E1 position (as e1_evaluate).
    e1s-miss  E1-S ROUTE_COVERAGE_MISS positions: arm E3-S.

Every replacement (and the played move when it was kept) is verified with the selective check at
``VERIFY_BUDGET`` (200k) and the full depth 0-2 class (``e1_build_suite.truth``) at ``--truth-budget``:

    NOT_DETECTED   the arm's check did not prove the played move lost
    RESCUED        it did, switched, and the new move is VCT2_CLEAR in the full class
    LOSS_TO_LOSS   it switched to a move that is proven lost (selective 200k or full class)
    SWITCH_UNRESOLVED  it switched to a move neither proven lost nor VCT2_CLEAR
    KEPT           it proved the move lost and found no alternative (the move stands)

    python scripts/e3_dev_replay.py --workers 6 --jsonl runs/e3/dev_replay.jsonl --output runs/e3/dev_replay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from renju import Game  # noqa: E402
from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, TRUTH_BUDGET, _run_jobs, truth  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, derive_seed, file_sha256  # noqa: E402

RESULTS = ROOT / 'docs' / 'mcts-v8-results'
E2_SOURCES = ('e2/puct_policy_vct2_8413.json', 'e2/puct_policy_vct2_8414.json')
E1_MANIFEST, E1_EVAL, E1S_MANIFEST = 'e1/e1_manifest.json', 'e1/e1_eval.json', 'e1s/e1s_manifest.json'
VERIFY_BUDGET = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 200_000}
# The E1 truth budget: a VCT2_CLEAR proof can need millions of nodes (E1-S (11,9): 6.9M, §12.32).
DEFAULT_TRUTH_BUDGET = dict(TRUTH_BUDGET)
CASES = ('e2-tree', 'e2-stage', 'e1-tree', 'e1s-miss')
OUTCOMES = ('NOT_DETECTED', 'RESCUED', 'LOSS_TO_LOSS', 'SWITCH_UNRESOLVED', 'KEPT')


def build_cases(kinds=CASES) -> list[dict]:
    cases = []
    if {'e2-tree', 'e2-stage'} & set(kinds):
        for name in E2_SOURCES:
            run = json.loads((RESULTS / name).read_text(encoding='utf-8'))
            for game in run['games']:
                wins = [a for a in game.get('opponent_vct2_attack', []) if a['status'] == 'WIN']
                if not wins:
                    continue
                ply = wins[0]['ply'] - 1
                move = next(m for m in game['v8_moves'] if m['ply'] == ply)
                kind = 'e2-tree' if move['route'] == 'tree' else 'e2-stage' if move['route'] in ('stage4', 'stage5') else None
                if kind not in kinds:
                    continue
                cases.append({'key': f"{kind}/{run['seed']}/{game['pair']}/{game['v8_color']}/{ply}", 'kind': kind,
                              'arm': 'e3a' if kind == 'e2-tree' else 'e3s', 'source': name,
                              'history': game['moves'][:ply], 'logged_move': game['moves'][ply],
                              'opening_plies': len(next(o['moves'] for o in run['openings'] if o['pair'] == game['pair'])),
                              'our_plies': [m['ply'] for m in game['v8_moves'] if m['ply'] <= ply],
                              'engine_seed': derive_seed(game['seed'], 'v8'), 'replay': kind == 'e2-tree',
                              'original_known_lost': True})
    if 'e1-tree' in kinds:
        manifest = {p['key']: p for p in json.loads((RESULTS / E1_MANIFEST).read_text(encoding='utf-8'))['positions']}
        for row in json.loads((RESULTS / E1_EVAL).read_text(encoding='utf-8'))['rows']:
            for run in row['runs']:
                if run['outcome'] != 'DETECT_MISS':
                    continue
                position = manifest[row['key']]
                cases.append({'key': f"e1-tree/{row['key']}/{run['seed']}", 'kind': 'e1-tree', 'arm': 'e3a',
                              'source': E1_EVAL, 'history': position['history'], 'logged_move': run['pre'],
                              'engine_seed': run['seed'], 'replay': False, 'original_known_lost': True})
    if 'e1s-miss' in kinds:
        for position in json.loads((RESULTS / E1S_MANIFEST).read_text(encoding='utf-8'))['positions']:
            if position['class'] != 'ROUTE_COVERAGE_MISS':
                continue
            cases.append({'key': f"e1s-miss/{position['key']}", 'kind': 'e1s-miss', 'arm': 'e3s',
                          'source': E1S_MANIFEST, 'history': position['history'], 'logged_move': position['played'],
                          'engine_seed': 42, 'replay': False, 'original_known_lost': True})
    return cases


def verify(history, move, truth_budget) -> dict:
    """Selective 200k status and the full-class truth of ``move`` after ``history``."""
    from analysis.e3_arms import selective_status

    game = Game()
    for m in history:
        game.play(*m)
    started = perf_counter()
    status, nodes = selective_status(game, move, VERIFY_BUDGET)
    full = truth(history, move, truth_budget)
    return {'selective_200k': status, 'selective_nodes': nodes, 'full': full,
            'seconds': round(perf_counter() - started, 2)}


def classify(decision: dict, checks: dict) -> str:
    if not decision['detected']:
        return 'NOT_DETECTED'
    if not decision['switched']:
        return 'KEPT'
    final = checks['final']
    if final['selective_200k'] == 'UNSAFE' or final['full']['status'] == PROVEN_LOSS:
        return 'LOSS_TO_LOSS'
    return 'RESCUED' if final['full']['status'] == CLEAR else 'SWITCH_UNRESOLVED'


def _decision(agent, arm) -> dict:
    d = agent.diagnostics
    if arm == 'e3a':
        checked = [[list(m), s] for m, s in d.v8_vct2_checked]
        return {'route': d.v8_route, 'pre': checked[0][0] if checked else None, 'checked': checked,
                'check_nodes': list(d.v8_vct2_check_nodes), 'detected': bool(checked) and checked[0][1] == 'UNSAFE',
                'switched': d.v8_vct2_switched, 'seconds': round(d.v8_vct2_seconds, 3)}
    info = agent.stage_info
    checked = info.get('checked', [])
    return {'route': d.v8_route, 'pre': checked[0][0] if checked else None, 'checked': checked,
            'detected': bool(checked) and checked[0][1] == 'UNSAFE', 'switched': info.get('switched', False),
            'final_tier': info.get('final_tier'), 'seconds': info.get('seconds')}


def _case_task(task):
    case, checkpoint, truth_budget = task
    from analysis.e3_arms import make_agent
    from scripts.run_mcts_v8_benchmark import load_policy

    policy = load_policy(checkpoint) if checkpoint else _uniform
    agent = make_agent(case['arm'], policy, seed=case['engine_seed'])
    game = Game()
    started = perf_counter()
    divergence = []
    if case['replay']:
        ours = set(case['our_plies'])
        for ply, move in enumerate(case['history']):
            if ply in ours and ply >= case['opening_plies']:
                chosen = agent.select_move(game)
                tree_move = agent.diagnostics.v8_v7_move
                if tuple(chosen) != tuple(move):
                    divergence.append({'ply': ply, 'arm_move': list(chosen), 'tree_move': list(tree_move or ()),
                                       'logged': list(move)})
            game.play(*move)
    else:
        for move in case['history']:
            game.play(*move)
    played = list(agent.select_move(game))
    decision = _decision(agent, case['arm'])
    decision['played'] = played
    decision['consistent'] = decision['pre'] == list(case['logged_move'])
    checks = {'final': verify(case['history'], played, truth_budget)}
    return {'key': case['key'], 'case': {k: v for k, v in case.items() if k not in ('history', 'our_plies')},
            'decision': decision, 'checks': checks, 'outcome': classify(decision, checks),
            'earlier_divergence': divergence, 'seconds': round(perf_counter() - started, 1),
            'budget': {'verify': VERIFY_BUDGET, 'truth': truth_budget}}


def _uniform(game):  # tests / stage-only smoke runs without the H3 checkpoint
    legal = game.legal_moves()
    return {m: 1.0 / len(legal) for m in legal}


def summarize(rows: list[dict]) -> dict:
    by_kind = {}
    for row in rows:
        kind = row['case']['kind']
        by_kind.setdefault(kind, {o: 0 for o in OUTCOMES})[row['outcome']] += 1
    return {'cases': len(rows), 'outcomes': {o: sum(r['outcome'] == o for r in rows) for o in OUTCOMES},
            'by_kind': by_kind, 'inconsistent': [r['key'] for r in rows if not r['decision']['consistent']],
            'unknown_replacements': sum(r['decision']['switched'] and r['decision']['checked'][-1][1] == 'UNKNOWN'
                                        for r in rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--uniform-policy', action='store_true', help='smoke tests only: no H3 checkpoint')
    parser.add_argument('--truth-budget', type=int, default=DEFAULT_TRUTH_BUDGET['node_budget'],
                        help='full-class node budget for verifying the played move (default 10M, the E1 truth budget)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    checkpoint = None
    hashes = None
    if not args.uniform_policy:
        from analysis.s3_vct2_v1 import check_checkpoint
        hashes = check_checkpoint(args.policy_checkpoint)
        checkpoint = args.policy_checkpoint
    truth_budget = {**DEFAULT_TRUTH_BUDGET, 'node_budget': args.truth_budget}
    cases = build_cases(args.cases)[:args.limit]
    done = _run_jobs([(c, checkpoint, truth_budget) for c in cases], _case_task, args.workers, args.jsonl,
                     lambda t: t[0]['key'], {'verify': VERIFY_BUDGET, 'truth': truth_budget})
    rows = [done[c['key']] for c in cases]
    payload = {'format': 'e3-dev-replay-v1', 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
               'restricted': args.limit is not None or args.uniform_policy, 'policy': hashes,
               'inputs': {n: file_sha256(RESULTS / n) for n in (*E2_SOURCES, E1_MANIFEST, E1_EVAL, E1S_MANIFEST)},
               'summary': summarize(rows), 'rows': rows}
    print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
