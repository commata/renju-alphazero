"""H6-5: TB-V1-Value against TB-Engine-v1, the rules fixed before the games (docs/h6-value.md, H6-5).

    python scripts/summarize_h6_5.py --main runs/h6_5/value_vs_tb1_8403.json \\
        --aux runs/h6_5/value_vs_full_8403.json runs/h6_5/tb1_vs_full_8403.json \\
        --p92 docs/h6/h6_4b_s1_probes.json --output docs/h6/h6_5_summary.json

Runs (``run_mcts_v8_benchmark.py`` outputs, 25 pairs each, merged across seeds):
    main  ``puct_value_vct2_50k`` (c 1.5) vs ``v8:puct_policy_vct2_50k`` (c 1.5) = TB-V1-Value vs TB-Engine-v1
    aux   ``puct_value_vct2_50k`` vs ``v8:full`` and ``puct_policy_vct2_50k`` vs ``v8:full`` on the same seeds
          (same openings): the tree not-SAFE baseline and a paired report; not a score gate

Decision (main score, 0.5 = equal to TB-Engine-v1; seeds 8403, 8404, reserve 8405):
    8403            any guard fails or score < 0.50         -> REJECT
    8403            score >= 0.50                           -> RUN_8404
    8403+8404       score >= 0.55, pair bootstrap low > 0.5 -> ADOPT
    8403+8404       score >= 0.55, low <= 0.5               -> INCONCLUSIVE: run 8405 once
    8403+8404       score < 0.55                            -> REJECT
    8403+8404+8405  score >= 0.55, low > 0.5                -> ADOPT, else REJECT
Guards (every stage): safety violations 0 in every run; candidate tree not-SAFE (aux) <= TB-Engine-v1's
(aux) + 0.03; time ratio (candidate / TB-Engine-v1 seconds in the main games) <= 1.5; black and white each
>= 0.45 (main); P92 guard not FAIL (``run_s1_probes.py`` ``h6_5_p92_guard``; NOT_DISCRIMINATING does not
fail). P93 / P94 are diagnostics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_h5 import merge, paired, summarize  # noqa: E402

CANDIDATE, BASELINE = 'puct_value_vct2_50k', 'puct_policy_vct2_50k'
MAIN_OPPONENT, AUX_OPPONENT = f'v8:{BASELINE}', 'v8:full'
SEEDS = (8403, 8404, 8405)
RULES = {'stage1_min': 0.50, 'adopt_min': 0.55, 'pair_low_above': 0.5, 'not_safe_margin': 0.03,
         'time_ratio_max': 1.5, 'colour_min': 0.45}


def _check_run(run: dict, arms: tuple, opponent: str) -> None:
    if run['arm'] not in arms or run['opponent'] != opponent:
        raise ValueError(f"expected {' / '.join(arms)} vs {opponent}, got {run['arm']} vs {run['opponent']}")
    if run['seed'] not in SEEDS:
        raise ValueError(f"seed {run['seed']} is not an H6-5 seed {SEEDS}")
    if run['v8_config'].get('puct_c') != 1.5 or (opponent == MAIN_OPPONENT and run.get('opponent_puct_c') != 1.5):
        raise ValueError(f"{run['arm']} vs {opponent} seed {run['seed']}: c_puct must be 1.5 on every PUCT side")


def _prior_top_rate(games: list[dict]) -> float | None:
    tree = [m for g in games for m in g['v8_moves'] if m['route'] == 'tree' and m.get('tree', {}).get('prior_top')]
    return round(sum(m['v7_move'] == m['tree']['prior_top'] for m in tree) / len(tree), 4) if tree else None


def value_report(games: list[dict]) -> dict:
    moves = [m for g in games for m in g['v8_moves'] if 'value' in m]
    q = [m['tree']['root_q'] for g in games for m in g['v8_moves'] if m.get('tree', {}).get('root_q')]
    b = sum(m['value']['black_leaves'] for m in moves)
    w = sum(m['value']['white_leaves'] for m in moves)
    seconds = sorted(m['seconds'] for g in games for m in g['v8_moves'])
    return {
        'value_tree_moves': len(moves),
        'value_seconds_per_move': round(sum(m['value']['seconds'] for m in moves) / len(moves), 4) if moves else None,
        'black_leaf_mean': round(sum(m['value']['black_mean'] * m['value']['black_leaves'] for m in moves
                                     if m['value']['black_leaves']) / b, 4) if b else None,
        'white_leaf_mean': round(sum(m['value']['white_mean'] * m['value']['white_leaves'] for m in moves
                                     if m['value']['white_leaves']) / w, 4) if w else None,
        'root_q_saturated_mean': round(sum(x['saturated'] for x in q) / len(q), 4) if q else None,
        'root_visited_children_mean': round(sum(x['children'] for x in q) / len(q), 2) if q else None,
        'move_seconds_p95': seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))] if seconds else None,
        'move_seconds_max': seconds[-1] if seconds else None,
        'distinct_games': len({json.dumps(g['moves']) for g in games}),
        # summarize_h5 counts the prior top for tree mode 'puct' only; this covers 'puct_value' too
        'tree_prior_top_rate': _prior_top_rate(games),
    }


def decide(main: dict, aux_candidate: dict | None, aux_baseline: dict | None, p92: str | None) -> dict:
    seeds = sorted(main['seeds'])
    if seeds not in ([8403], [8403, 8404], [8403, 8404, 8405]):
        return {'decision': 'INCOMPLETE', 'reasons': [f'seeds {seeds} are not 8403 / 8403+8404 / +8405']}
    if aux_candidate is None or aux_baseline is None or sorted(aux_candidate['seeds']) != seeds \
            or sorted(aux_baseline['seeds']) != seeds:
        return {'decision': 'INCOMPLETE', 'reasons': ['both aux runs are needed on the same seeds']}
    if p92 is None:
        return {'decision': 'INCOMPLETE', 'reasons': ['the P92 guard result is needed']}
    fails = []
    if any(r['safety_violations'] for r in (main, aux_candidate, aux_baseline)):
        fails.append('safety violation')
    if aux_candidate['tree_not_safe_rate'] is not None and aux_baseline['tree_not_safe_rate'] is not None \
            and aux_candidate['tree_not_safe_rate'] > aux_baseline['tree_not_safe_rate'] + RULES['not_safe_margin']:
        fails.append(f"tree not-SAFE {aux_candidate['tree_not_safe_rate']} > TB-Engine-v1 "
                     f"{aux_baseline['tree_not_safe_rate']} + {RULES['not_safe_margin']}")
    if main['time_ratio'] is not None and main['time_ratio'] > RULES['time_ratio_max']:
        fails.append(f"time ratio {main['time_ratio']} > {RULES['time_ratio_max']}")
    for colour, score in main['by_colour'].items():
        if score is not None and score < RULES['colour_min']:
            fails.append(f"{colour} {score} < {RULES['colour_min']}")
    if p92 == 'FAIL':
        fails.append('P92 guard FAIL')
    score, low = main['score'], main['ci95_pairs'][0]
    if fails:
        return {'decision': 'REJECT', 'reasons': fails}
    if seeds == [8403]:
        if score < RULES['stage1_min']:
            return {'decision': 'REJECT', 'reasons': [f"8403 score {score} < {RULES['stage1_min']}"]}
        return {'decision': 'RUN_8404', 'reasons': []}
    if score < RULES['adopt_min']:
        return {'decision': 'REJECT', 'reasons': [f"score {score} < {RULES['adopt_min']}"]}
    if low > RULES['pair_low_above']:
        return {'decision': 'ADOPT', 'reasons': []}
    if seeds == [8403, 8404]:
        return {'decision': 'INCONCLUSIVE', 'reasons': [f'pair bootstrap low {low} <= 0.5: run 8405 once']}
    return {'decision': 'REJECT', 'reasons': [f'pair bootstrap low {low} <= 0.5 on three seeds']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--main', nargs='+', type=Path, required=True, help='TB-V1-Value vs TB-Engine-v1 runs')
    parser.add_argument('--aux', nargs='*', type=Path, default=[], help='both arms vs v8:full on the same seeds')
    parser.add_argument('--p92', type=Path, help='run_s1_probes.py output with h6_5_p92_guard')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    main_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.main]
    for run in main_runs:
        _check_run(run, (CANDIDATE,), MAIN_OPPONENT)
    aux_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.aux]
    for run in aux_runs:
        _check_run(run, (CANDIDATE, BASELINE), AUX_OPPONENT)
    main_arm = merge(main_runs, MAIN_OPPONENT)[CANDIDATE]
    aux = merge(aux_runs, AUX_OPPONENT) if aux_runs else {}
    main_row = summarize(main_arm)
    aux_rows = {arm: summarize(a) for arm, a in aux.items()}
    p92 = (json.loads(args.p92.read_text(encoding='utf-8'))['summary'].get('h6_5_p92_guard', {}).get('verdict')
           if args.p92 else None)
    result = {
        'format': 'h6-5-summary-v1', 'rules': RULES,
        'main': {**main_row, 'value': value_report(main_arm['games'])},
        'aux': {arm: {**row, 'value': value_report(aux[arm]['games'])} for arm, row in aux_rows.items()},
        'aux_paired': (paired(aux[CANDIDATE], aux[BASELINE]) if CANDIDATE in aux and BASELINE in aux else None),
        'p92_guard': p92,
        'verdict': decide(main_row, aux_rows.get(CANDIDATE), aux_rows.get(BASELINE), p92),
    }
    print(json.dumps({'main_score': main_row['score'], 'pairs_ci95': main_row['ci95_pairs'],
                      'by_colour': main_row['by_colour'], 'time_ratio': main_row['time_ratio'],
                      'aux_paired': result['aux_paired'], 'verdict': result['verdict']}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
