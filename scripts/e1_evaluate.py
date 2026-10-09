"""E1 evaluator: run the frozen S3-VCT2-v1 on the E1 manifest (docs/mcts-v8-teacher.md §12.26).

For every manifest position and each of its seeds the frozen engine (``analysis.s3_vct2_v1``)
chooses a move. With the same seed the tree equals the builder's run (VCT2 check off), so the
move the VCT2 check examined first (``pre``) must equal the builder's played move; a mismatch
is reported (``inconsistent_runs``). Each run ends in one outcome:

E1-P (a VCT2_CLEAR pool move exists)
    RESCUED           pre is PROVEN_LOSS, the check switched, the final move is VCT2_CLEAR
    TREE_AVOIDED      pre itself is not lost (the tree did not pick the losing move this run)
    DETECT_MISS       pre is PROVEN_LOSS but the 10k check did not prove it
    POOL_MISS         this run's root children hold no VCT2_CLEAR move
    K4_MISS           clear moves are in the pool but none among the first ``vct2_max_children``
    BUDGET_AMBIGUITY  a clear move is among them, but the check stopped at (or left) a move it
                      could not prove lost (UNKNOWN) before a move it identified as no-loss
    SELECTION_ERROR   the check identified a clear move as no-loss (SAFE) and played another
    TRUTH_UNRESOLVED  the truth needed for the verdict is UNKNOWN
    ROUTE_OTHER       the engine moved by a stage / own-attack route (no VCT2 check)
E1-N (every pool move lost): KEPT (no switch) / LOSS_TO_LOSS_SWITCH / TRUTH_UNRESOLVED
E1-C (control): NO_VETO / VETO_ON_LOSS (the tree picked a lost move this run) / FALSE_VETO

UNSOUND counts an engine UNSAFE on a move whose truth is VCT2_CLEAR (must be 0).

    python scripts/e1_evaluate.py --manifest runs/e1/e1_manifest.json --workers 14 \\
        --jsonl runs/e1/eval.jsonl --output runs/e1/e1_eval.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.mcts_v8 import _root_order  # noqa: E402
from analysis.s3_vct2_v1 import NAME, S3_VCT2_V1  # noqa: E402
from renju import Game  # noqa: E402
from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, TRUTH_BUDGET, truth  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256  # noqa: E402

FORMAT = 'e1-eval-v1'
K = S3_VCT2_V1['vct2_max_children']


def _key(move) -> str:
    return f'{move[0]},{move[1]}'


def outcome(cls: str, truths: dict, run: dict, k: int = K) -> str:
    """Outcome of one engine run (``run``: route, pre, played, checked, switched, children)."""
    status = lambda m: (truths.get(_key(m)) or {}).get('status')  # noqa: E731
    engine = {_key(m): s for m, s in run['checked']}
    pre, final = run['pre'], run['played']
    vetoed = bool(run['checked']) and run['checked'][0][1] == 'UNSAFE'
    if cls == 'E1-C':
        if not vetoed:
            return 'NO_VETO'
        return 'VETO_ON_LOSS' if status(pre) == PROVEN_LOSS else 'FALSE_VETO'
    if cls == 'E1-N':
        if status(final) != PROVEN_LOSS:
            return 'TRUTH_UNRESOLVED'
        return 'LOSS_TO_LOSS_SWITCH' if run['switched'] else 'KEPT'
    # E1-P
    if run['route'] != 'tree' and status(final) != CLEAR:
        return 'ROUTE_OTHER'  # a stage / own-attack move: the VCT2 check does not run there
    if status(final) == CLEAR:
        return 'RESCUED' if run['switched'] else 'TREE_AVOIDED'
    if status(pre) != PROVEN_LOSS:
        return 'TREE_AVOIDED' if status(pre) == CLEAR else 'TRUTH_UNRESOLVED'
    if engine.get(_key(pre)) != 'UNSAFE':
        return 'DETECT_MISS'
    if status(final) != PROVEN_LOSS:
        return 'TRUTH_UNRESOLVED'
    clear = [tuple(m) for m, _, _ in run['children'] if status(m) == CLEAR]
    if not clear:
        return 'POOL_MISS'
    children = [SimpleNamespace(move=tuple(m), visits=v, mean_value=q) for m, v, q in run['children']]
    first_k = _root_order(tuple(pre), children)[:k]
    if not any(m in first_k for m in clear):
        return 'K4_MISS'
    if any(engine.get(_key(m)) == 'SAFE' for m in clear):
        return 'SELECTION_ERROR'
    return 'BUDGET_AMBIGUITY'


def _evaluate(task):
    from analysis.s3_vct2_v1 import make_agent
    from scripts.run_mcts_v8_benchmark import load_policy

    position, checkpoint = task
    policy = load_policy(checkpoint)
    game = Game()
    for move in position['history']:
        game.play(*move)
    truths = dict(position['truths'])
    runs = []
    for seed, built in zip(position['e1_seeds'], position['runs']):
        agent = make_agent(policy, seed=seed)
        started = perf_counter()
        played = list(agent.select_move(game))
        seconds = perf_counter() - started
        d = agent.diagnostics
        checked = [[list(m), s] for m, s in d.v8_vct2_checked]
        run = {'seed': seed, 'route': d.v8_route, 'pre': checked[0][0] if checked else played, 'played': played,
               'checked': checked, 'switched': d.v8_vct2_switched,
               'children': [[list(m), v, q] for m, v, q in d.v8_root_visits],
               'vct2_seconds': round(d.v8_vct2_seconds, 4), 'vct2_nodes': d.v8_vct2_nodes,
               'seconds': round(seconds, 3), 'consistent': (checked[0][0] if checked else played) == built['played']}
        for move in {tuple(run['pre']), tuple(played)}:  # truths the verdict needs
            if _key(move) not in truths:
                truths[_key(move)] = {**truth(position['history'], move, TRUTH_BUDGET), 'computed_in_eval': True}
        run['outcome'] = outcome(position['class'], truths, run)
        run['unsound'] = [m for m, s in checked if s == 'UNSAFE' and truths.get(_key(m), {}).get('status') == CLEAR]
        runs.append(run)
    return {'key': position['key'], 'class': position['class'], 'runs': runs,
            'truths_added': {k: v for k, v in truths.items() if k not in position['truths']}}


def _dist(values):
    values = sorted(values)
    if not values:
        return None
    return {'n': len(values), 'p50': round(median(values), 3),
            'p95': round(values[min(len(values) - 1, int(0.95 * len(values)))], 3), 'max': round(values[-1], 3)}


def _ratio(num, den):
    return {'value': round(num / den, 3) if den else None, 'n': f'{num}/{den}'}


def summarize(rows: list[dict], truths_of: dict) -> dict:
    runs = [(r['class'], r['key'], run) for r in rows for run in r['runs']]
    status = lambda key, m: (truths_of[key].get(_key(m)) or {}).get('status')  # noqa: E731
    out = {'outcomes': {}}
    for cls, _, run in runs:
        out['outcomes'].setdefault(cls, {})
        out['outcomes'][cls][run['outcome']] = out['outcomes'][cls].get(run['outcome'], 0) + 1
    p = [(key, run) for cls, key, run in runs if cls == 'E1-P']
    lossy = [(key, run) for cls, key, run in runs if cls in ('E1-P', 'E1-N') and status(key, run['pre']) == PROVEN_LOSS]
    switched = [(key, run) for cls, key, run in runs if run['switched']]
    clear_in = lambda key, run, first: any(  # noqa: E731
        status(key, m) == CLEAR for m in first(key, run))
    order = lambda key, run: _root_order(tuple(run['pre']), [SimpleNamespace(  # noqa: E731
        move=tuple(m), visits=v, mean_value=q) for m, v, q in run['children']])
    out['metrics'] = {
        'rescue_rate': _ratio(sum(status(k, r['played']) == CLEAR for k, r in p), len(p)),
        'veto_recall': _ratio(sum(dict((_key(m), s) for m, s in r['checked']).get(_key(r['pre'])) == 'UNSAFE'
                                  for _, r in lossy), len(lossy)),
        'rescue_pool_coverage': _ratio(sum(clear_in(k, r, order) for k, r in p), len(p)),
        'rescue_k4_coverage': _ratio(sum(clear_in(k, r, lambda kk, rr: order(kk, rr)[:K]) for k, r in p), len(p)),
        'conditional_escape_rate': _ratio(sum(status(k, r['played']) == CLEAR for k, r in switched), len(switched)),
        'unknown_replacement_rate': _ratio(sum(dict((_key(m), s) for m, s in r['checked']).get(_key(r['played']))
                                               == 'UNKNOWN' for _, r in switched), len(switched)),
        'false_veto': sum(run['outcome'] == 'FALSE_VETO' for _, _, run in runs),
        'unsound_witness': sum(len(run['unsound']) for _, _, run in runs),
        'inconsistent_runs': sum(not run['consistent'] for _, _, run in runs),
    }
    out['cost'] = {'vct2_seconds': _dist([run['vct2_seconds'] for _, _, run in runs if run['checked']]),
                   'move_seconds': _dist([run['seconds'] for _, _, run in runs])}
    out['positions'] = [{'key': r['key'], 'class': r['class'], 'outcomes': [run['outcome'] for run in r['runs']]}
                        for r in rows]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--classes', nargs='+', default=['E1-P', 'E1-N', 'E1-C'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    from analysis.s3_vct2_v1 import check_checkpoint

    hashes = check_checkpoint(args.policy_checkpoint)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    positions = [p for p in manifest['positions'] if p['class'] in args.classes]
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row['key']] = row
    pending = [p for p in positions if p['key'] not in done]
    print(f'positions={len(positions)} done={len(positions) - len(pending)} workers={args.workers}', flush=True)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(row):
        done[row['key']] = row
        print(f"{len(done)}/{len(positions)} {row['key']} {row['class']} "
              f"{[run['outcome'] for run in row['runs']]}", flush=True)
        if args.jsonl is not None:
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')

    tasks = [(p, args.policy_checkpoint) for p in pending]
    if args.workers == 1:
        for task in tasks:
            keep(_evaluate(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(_evaluate, t) for t in tasks]):
                keep(future.result())
    rows = [done[p['key']] for p in positions]
    truths_of = {p['key']: {**p['truths'], **done[p['key']]['truths_added']} for p in positions}
    summary = summarize(rows, truths_of)
    payload = {'format': FORMAT, 'engine': NAME, 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
               'policy': hashes, 'manifest_sha256': file_sha256(args.manifest), 'split': manifest.get('split'),
               'summary': summary, 'rows': rows}
    print(json.dumps({k: v for k, v in summary.items() if k != 'positions'}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
