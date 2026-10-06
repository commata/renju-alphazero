"""H5: fix c_puct before the benchmark (docs/mcts-v8-teacher.md §12.16).

Not a strength tuning (no games, no win rate): it only rules out c values where the
PUCT tree is abnormal, on RenjuNet ``val`` positions that reach V8's tree route (the
same filter as ``h4_offline_recall.py``; test and the benchmark seeds are never used).

For every position, prior (uniform / heuristic / policy) and c in ``--c-values`` the
PUCT tree runs alone (no V8-C), with V8's root list and simulation budget, and records:

    visited     root children with at least one visit
    top_share   visits of the most visited root child / simulations
    policy_top  the tree's move is the H3 policy's top root child

Rule, fixed in advance:
- a c passes if, for every prior, mean visited >= 3 and mean top_share <= 0.75
  (the search does not collapse onto one child), and for the policy prior
  policy_top <= 0.80 (the prior does not decide alone) and policy_top is at least
  0.05 above the uniform prior's (the prior is not ignored);
- the c used is the passing value closest to 1.5 (Track A's default), the smaller on a
  tie; if none passes, report and stop (no benchmark).

    python scripts/h5_calibrate_puct.py --positions 120 --workers 4 --output runs/h5/calibration.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics, prepare_root_moves  # noqa: E402
from analysis.puct_v8 import PUCT_PRIORS, search_tree_puct  # noqa: E402
from h4_offline_recall import sample_states  # noqa: E402
from renju import Game  # noqa: E402
from search.mcts_v5 import _RootContext, _forced_v5_move  # noqa: E402
from search.mcts_v7 import _find_vcf_with_stats  # noqa: E402

CFG = V8_DEFAULTS
SAFETY = dict(candidate_limit=CFG['candidate_limit'], neighborhood_radius=CFG['neighborhood_radius'],
              safety_vcf_max_fours=CFG['safety_vcf_max_fours'], safety_vcf_node_limit=CFG['safety_vcf_node_limit'],
              safety_precheck_node_limit=CFG['safety_precheck_node_limit'],
              safety_total_node_limit=CFG['safety_total_node_limit'],
              self_forbidden_min_white=CFG['self_forbidden_min_white'])
TREE = dict(candidate_limit=CFG['candidate_limit'], neighborhood_radius=CFG['neighborhood_radius'],
            priority_top_k=CFG['priority_top_k'])
MIN_VISITED, MAX_TOP_SHARE, MAX_POLICY_TOP, MIN_PRIOR_EFFECT, DEFAULT_C = 3, 0.75, 0.80, 0.05, 1.5

_policy = None


def _init(checkpoint):
    global _policy
    from hybrid.h4_policy import RootPolicy
    _policy = RootPolicy(checkpoint, threads=1)


def tree_position(moves) -> Game | None:
    """The position if V8 would reach its tree route there (no forced move, no own VCF)."""
    game = Game()
    for move in moves:
        game.play(*move)
    if _forced_v5_move(game, context=_RootContext(game.legal_moves(), SearchDiagnostics())) is not None:
        return None
    found, _, _ = _find_vcf_with_stats(game, game.to_play, max_fours=CFG['own_vcf_max_fours'],
                                       node_limit=CFG['own_vcf_node_limit'])
    return None if found is not None else game


def run_position(task) -> dict | None:
    index, moves, target, c_values = task
    game = tree_position(moves)
    if game is None:
        return None
    root, score, _, _ = prepare_root_moves(game, _RootContext(game.legal_moves(), SearchDiagnostics()),
                                           SearchDiagnostics(), **SAFETY)
    if not root:
        return None
    budget = CFG['tactical_simulations'] if score >= CFG['tactical_score_threshold'] else CFG['simulations']
    scores = _policy(game)
    policy_top = max(root, key=lambda m: (scores.get(m, 0.0), -root.index(m)))
    out = {'index': index, 'simulations': budget, 'children': len(root), 'runs': []}
    for prior in PUCT_PRIORS:
        for c in c_values:
            started = perf_counter()
            move, children = search_tree_puct(game, root, budget, c_puct=c, prior=prior,
                                              policy=_policy if prior == 'policy' else None,
                                              random=Random(index), **TREE)
            out['runs'].append({'prior': prior, 'c': c, 'visited': sum(ch.visits > 0 for ch in children),
                                'top_share': max(ch.visits for ch in children) / budget,
                                'policy_top': move == policy_top, 'human': move == target,
                                'seconds': perf_counter() - started})
    return out


def decide(table: dict, c_values) -> tuple[float | None, dict]:
    reasons = {}
    for c in c_values:
        rows = {p: table[(p, c)] for p in PUCT_PRIORS}
        fails = [f'{p}: visited {r["visited"]:.2f} < {MIN_VISITED}' for p, r in rows.items()
                 if r['visited'] < MIN_VISITED]
        fails += [f'{p}: top_share {r["top_share"]:.2f} > {MAX_TOP_SHARE}' for p, r in rows.items()
                  if r['top_share'] > MAX_TOP_SHARE]
        if rows['policy']['policy_top'] > MAX_POLICY_TOP:
            fails.append(f'policy_top {rows["policy"]["policy_top"]:.2f} > {MAX_POLICY_TOP}')
        if rows['policy']['policy_top'] - rows['uniform']['policy_top'] < MIN_PRIOR_EFFECT:
            fails.append(f'policy prior effect {rows["policy"]["policy_top"] - rows["uniform"]["policy_top"]:+.2f}'
                         f' < {MIN_PRIOR_EFFECT}')
        reasons[str(c)] = fails
    passing = [c for c in c_values if not reasons[str(c)]]
    chosen = min(passing, key=lambda c: (abs(c - DEFAULT_C), c)) if passing else None
    return chosen, reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, default=ROOT / 'data/external/renjunet/games.jsonl.gz')
    parser.add_argument('--policy-checkpoint', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt')
    parser.add_argument('--positions', type=int, default=120, help='tree-route positions to use')
    parser.add_argument('--c-values', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # About half of the sampled states reach the tree route; sample enough and stop early.
    sampled, population = sample_states(args.games, 'val', args.positions * 4, args.seed)
    tasks = [(i, record['moves'][:ply], tuple(record['moves'][ply]), args.c_values)
             for i, (record, ply) in enumerate(sampled)]
    results, started = [], perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.policy_checkpoint),)) as pool:
        for result in pool.map(run_position, tasks, chunksize=1):
            if result is not None:
                results.append(result)
                if len(results) % 10 == 0:
                    print(f'  {len(results)}/{args.positions} tree positions, {perf_counter() - started:.0f}s',
                          flush=True)
                if len(results) >= args.positions:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    table = {}
    for prior in PUCT_PRIORS:
        for c in args.c_values:
            runs = [r for res in results for r in res['runs'] if r['prior'] == prior and r['c'] == c]
            table[(prior, c)] = {key: sum(float(r[key]) for r in runs) / len(runs)
                                 for key in ('visited', 'top_share', 'policy_top', 'human', 'seconds')}
    chosen, reasons = decide(table, args.c_values)
    print('| prior | c | visited | top_share | policy_top | human (diag.) | s/search |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for (prior, c), row in table.items():
        print(f"| {prior} | {c} | {row['visited']:.2f} | {row['top_share']:.3f} | {row['policy_top']:.3f} | "
              f"{row['human']:.3f} | {row['seconds']:.2f} |")
    for c, fails in reasons.items():
        print(f'c={c}: ' + ('pass' if not fails else '; '.join(fails)))
    print(f'selected c_puct: {chosen}' if chosen is not None else 'no c passed: do not run the benchmark')
    report = {'format': 'h5-puct-calibration-v1', 'split': 'val', 'population_states': population,
              'positions': len(results), 'c_values': args.c_values, 'selected_c': chosen, 'reasons': reasons,
              'rule': {'min_visited': MIN_VISITED, 'max_top_share': MAX_TOP_SHARE,
                       'max_policy_top': MAX_POLICY_TOP, 'min_prior_effect': MIN_PRIOR_EFFECT,
                       'default_c': DEFAULT_C},
              'table': [{'prior': p, 'c': c, **row} for (p, c), row in table.items()],
              'simulations': {str(s): sum(r['simulations'] == s for r in results) for s in
                              sorted({r['simulations'] for r in results})},
              'seconds': round(perf_counter() - started, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0 if chosen is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
