"""H5: PUCT arms against ``v8:full`` (docs/mcts-v8-teacher.md §12.16).

    python scripts/summarize_h5.py runs/h5/uniform_8401.json runs/h5/heur_8401.json \\
        runs/h5/policy_8401.json --output runs/h5/summary_8401.json

Runs of the same arm (e.g. seeds 8401 and 8402) are merged; every run must have the
opponent ``v8:full``. Score 0.5 = equal to the baseline. Per arm: score with a 95% Wilson
interval and a 95% bootstrap interval over opening pairs, colours, the safety invariant
(``summarize_h4.safety``), losses after an opponent VCT attack, tree-move safety (the
tree's move before V8-C: not proven SAFE), V8-C changes, root check minutes, tree speed,
prior statistics and the time ratio (V8 seconds / opponent seconds in the same games).
Paired differences between arms use the same opening pairs (bootstrap over pairs).

Decision rules (fixed before the runs, §12.16):
    stop       any safety violation
    candidate  puct_policy >= 0.60, puct_policy - puct_heur >= +0.10, both colours >= 0.50,
               tree not-SAFE rate <= puct_heur's + 0.03, time ratio <= 1.5
               -> second seed (8402)
    adopt      on all seeds together: puct_policy >= 0.58, bootstrap lower bound of
               (puct_policy - 0.5) > 0, puct_policy - puct_heur >= +0.05, same guards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_h4 import SCORE, safety, wilson  # noqa: E402

OPPONENT = 'v8:full'
BOOTSTRAP = 10_000


def merge(runs: list[dict]) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for run in runs:
        if run.get('opponent') != OPPONENT:
            raise ValueError(f"{run['arm']}: opponent {run.get('opponent')} != {OPPONENT}")
        arm = arms.setdefault(run['arm'], {'arm': run['arm'], 'seeds': [], 'games': [], 'config': run['v8_config']})
        if run['seed'] in arm['seeds']:
            raise ValueError(f"{run['arm']}: seed {run['seed']} given twice")
        if run['v8_config'] != arm['config']:
            raise ValueError(f"{run['arm']}: runs with different V8 configurations")
        arm['seeds'].append(run['seed'])
        for game in run['games']:
            arm['games'].append({**game, 'run_seed': run['seed']})
    return arms


def pair_scores(games) -> dict[tuple, float]:
    """Mean score per opening pair (both colours), keyed by (run seed, pair)."""
    out: dict[tuple, list] = {}
    for g in games:
        out.setdefault((g['run_seed'], g['pair']), []).append(SCORE[g['result']])
    return {k: sum(v) / len(v) for k, v in out.items()}


def bootstrap(values: list[float], seed: int = 0) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = Random(seed)
    means = sorted(mean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAP))
    return [round(means[int(0.025 * BOOTSTRAP)], 3), round(means[int(0.975 * BOOTSTRAP) - 1], 3)]


def paired(a: dict, b: dict) -> dict:
    pa, pb = pair_scores(a['games']), pair_scores(b['games'])
    common = sorted(set(pa) & set(pb))
    diffs = [pa[k] - pb[k] for k in common]
    return {'pairs': len(common), 'diff': round(mean(diffs), 3) if diffs else None, 'ci95': bootstrap(diffs)}


def tree_status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['v7_move']), 'UNCHECKED')


def summarize(arm: dict) -> dict:
    games = arm['games']
    total = sum(SCORE[g['result']] for g in games)
    moves = [m for g in games for m in g['v8_moves']]
    tree = [m for m in moves if m['route'] == 'tree']
    puct = [m for m in tree if m.get('tree', {}).get('mode') == 'puct']
    statuses = [tree_status(m) for m in tree]
    by_colour = {}
    for colour in ('black', 'white'):
        part = [g for g in games if g['v8_color'] == colour]
        s = sum(SCORE[g['result']] for g in part)
        by_colour[colour] = round(s / len(part), 3) if part else None
    v8_seconds = sum(m['seconds'] for m in moves)
    opp_seconds = sum(s for g in games for s in g['opponent_move_seconds'])
    tree_seconds = sum(m.get('tree', {}).get('seconds', 0.0) for m in tree)
    tree_sims = sum(m.get('tree', {}).get('simulations', 0) for m in tree)
    check = safety({'games': games})
    return {
        'arm': arm['arm'], 'seeds': arm['seeds'], 'games': len(games),
        'wins': sum(g['result'] == 'win' for g in games), 'draws': sum(g['result'] == 'draw' for g in games),
        'losses': sum(g['result'] == 'loss' for g in games),
        'score': round(total / len(games), 3), 'ci95_wilson': wilson(total, len(games)),
        'ci95_pairs': bootstrap(list(pair_scores(games).values())), 'by_colour': by_colour,
        'safety_violations': check['violations'],
        'proven_loss_in_lost_position_per_move': round(check['proven_loss_in_lost_position'] / len(moves), 4),
        'losses_with_opponent_vct': sum(g['result'] == 'loss' and g.get('opponent_routes', {}).get('own_vct', 0) > 0
                                        for g in games),
        'tree_moves': len(tree),
        'tree_status': {s: statuses.count(s) for s in ('SAFE', 'UNKNOWN', 'UNSAFE', 'UNCHECKED')},
        'tree_not_safe_rate': round(sum(s in ('UNKNOWN', 'UNSAFE') for s in statuses) / len(tree), 4) if tree else None,
        'v8c_changed': sum(m['changed'] for m in tree),
        'root_check_minutes': round(sum(m['root']['seconds'] for m in tree) / 60, 1),
        'tree_seconds_median': round(median(m.get('tree', {}).get('seconds', 0.0) for m in tree), 3) if tree else None,
        'tree_simulations_per_second': round(tree_sims / tree_seconds, 1) if tree_seconds else None,
        'nn_calls_per_tree_move': round(mean(m['tree']['nn_calls'] for m in puct), 1) if puct else 0,
        'prior_fallbacks': sum(m['tree']['prior_fallbacks'] for m in puct),
        'prior_entropy_mean': round(mean(m['tree']['prior_entropy'] for m in puct), 3) if puct else None,
        'tree_chose_prior_top_rate': (round(sum(m['v7_move'] == m['tree']['prior_top'] for m in puct) / len(puct), 3)
                                      if puct else None),
        'time_ratio': round(v8_seconds / opp_seconds, 3) if opp_seconds else None,
        'total_game_minutes': round(sum(g['game_seconds'] for g in games) / 60, 1),
    }


def guards(row: dict, heur: dict | None) -> list[str]:
    fails = []
    if row['safety_violations']:
        fails.append('safety violation')
    for colour, score in row['by_colour'].items():
        if score is not None and score < 0.5:
            fails.append(f'{colour} {score} < 0.50')
    if heur is not None and row['tree_not_safe_rate'] is not None and heur['tree_not_safe_rate'] is not None \
            and row['tree_not_safe_rate'] > heur['tree_not_safe_rate'] + 0.03:
        fails.append(f"tree not-SAFE {row['tree_not_safe_rate']} > puct_heur {heur['tree_not_safe_rate']} + 0.03")
    if row['time_ratio'] is not None and row['time_ratio'] > 1.5:
        fails.append(f"time ratio {row['time_ratio']} > 1.5 (run the time-matched check)")
    return fails


def verdict(rows: dict, diffs: dict) -> dict:
    if any(r['safety_violations'] for r in rows.values()):
        return {'decision': 'stop', 'reasons': ['safety violation']}
    policy, heur = rows.get('puct_policy'), rows.get('puct_heur')
    if policy is None or heur is None:
        return {'decision': 'incomplete', 'reasons': ['needs puct_policy and puct_heur']}
    diff = diffs.get('puct_policy - puct_heur', {}).get('diff')
    fails = guards(policy, heur)
    seeds = len(policy['seeds'])
    if seeds == 1:
        if policy['score'] < 0.60:
            fails.append(f"score {policy['score']} < 0.60")
        if diff is None or diff < 0.10:
            fails.append(f'policy - heur {diff} < +0.10')
        return {'decision': 'candidate: run seed 8402' if not fails else 'not a candidate', 'reasons': fails}
    if policy['score'] < 0.58:
        fails.append(f"score {policy['score']} < 0.58")
    if policy['ci95_pairs'][0] <= 0.5:
        fails.append(f"pair bootstrap lower bound {policy['ci95_pairs'][0]} <= 0.5")
    if diff is None or diff < 0.05:
        fails.append(f'policy - heur {diff} < +0.05')
    return {'decision': 'adopt policy prior' if not fails else 'do not adopt', 'reasons': fails}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    arms = merge([json.loads(p.read_text(encoding='utf-8')) for p in args.runs])
    rows = {name: summarize(arm) for name, arm in arms.items()}
    diffs = {}
    for a, b in (('puct_heur', 'puct_uniform'), ('puct_policy', 'puct_heur'), ('puct_policy', 'puct_uniform')):
        if a in arms and b in arms:
            diffs[f'{a} - {b}'] = paired(arms[a], arms[b])
    print('| arm | seeds | games | W/D/L | score | Wilson 95% | pairs 95% | black | white | safety viol. | '
          'loss w/ opp VCT | tree not-SAFE | V8-C changed | prior-top chosen | entropy | sims/s | time ratio |')
    print('|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows.values():
        print(f"| {r['arm']} | {','.join(map(str, r['seeds']))} | {r['games']} | {r['wins']}/{r['draws']}/{r['losses']} | "
              f"{r['score']} | {r['ci95_wilson']} | {r['ci95_pairs']} | {r['by_colour']['black']} | "
              f"{r['by_colour']['white']} | {len(r['safety_violations'])} | {r['losses_with_opponent_vct']} | "
              f"{r['tree_not_safe_rate']} | {r['v8c_changed']} | {r['tree_chose_prior_top_rate']} | "
              f"{r['prior_entropy_mean']} | {r['tree_simulations_per_second']} | {r['time_ratio']} |")
    for name, d in diffs.items():
        print(f"{name}: {d['diff']:+.3f} {d['ci95']} over {d['pairs']} opening pairs")
    decision = verdict(rows, diffs)
    print(f"decision: {decision['decision']}" + (f" ({'; '.join(decision['reasons'])})" if decision['reasons'] else ''))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'runs': [str(p) for p in args.runs], 'rows': rows, 'diffs': diffs,
                                           'decision': decision}, indent=1), encoding='utf-8')
    return 1 if decision['decision'] == 'stop' else 0


if __name__ == '__main__':
    raise SystemExit(main())
