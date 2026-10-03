"""Recipe A/B verdict between two arms branched from the same checkpoint (Stage 8 C2).

Pre-registered rule (docs/stage8-plan.md §12.17). Inputs are files the training loop and
``run_stage8_head_to_head.py`` already write:

1. **Direct matches** (``--direct``): treatment vs control at the same generations. Each
   file must use its own ``--seed`` (otherwise the same openings would be counted as
   independent pairs). Decisive opening pairs (2-0 / 0-2) of all files are pooled into a
   two-sided exact sign test. p < 0.05 for the treatment -> ``adopt_stronger``; for the
   control -> ``reject``.
2. Otherwise the treatment is adopted as the *continuation recipe* (not as a stronger
   model) only if all of these hold -> ``adopt_stability``, else ``keep_control``:
   - stability over the self-play generations [--from, --to): one-sided generations
     (one colour >= 15/16) at least ``--stability-margin`` lower than the control AND a
     lower mean per-generation colour margin |B - W| / games;
   - heavy non-inferiority: games paired by (generation, opponent, game index) of
     ``genNNN_heavy.json`` at ``--points`` (same seed -> same openings, checked); the
     one-sided 95 % lower bound of the treatment-minus-control win rate is above
     ``--non-inferiority`` (default 5 points);
   - no probe regression at the last point: rows paired by id over all probe sets,
     exact McNemar on top-1, the control is not better with p < 0.05.

    python scripts/compare_recipe_arms.py --control runs/stage8_ada_c1120 \\
        --treatment runs/stage8_lr3_c1480 --from 1480 --to 1600 \\
        --points 1520 1560 1600 --direct runs/arm_h2h/lr_gen*.json \\
        --control-prefix CTL --treatment-prefix LR --output runs/arm_h2h/lr_verdict.json
"""
from __future__ import annotations

import argparse
import glob
import json
from math import comb, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from analyze_self_play_health import load_games, load_generation_events, window_stats  # noqa: E402

P_LEVEL = 0.05
Z_ONE_SIDED_95 = 1.6449
PROBE_SUFFIXES = ('', '_defense', '_vct')


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p of wins vs losses at 1/2."""
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def direct_result(paths: list[Path], treatment_prefix: str, control_prefix: str) -> dict:
    wins = losses = 0
    seeds, used = [], []
    for path in paths:
        data = _load(path)
        for match in data['matches']:
            s = match['summary']
            if s['a'].startswith(treatment_prefix) and s['b'].startswith(control_prefix):
                w, l = s['a_pair_wins'], s['a_pair_losses']
            elif s['b'].startswith(treatment_prefix) and s['a'].startswith(control_prefix):
                w, l = s['a_pair_losses'], s['a_pair_wins']
            else:
                continue
            wins, losses = wins + w, losses + l
            seeds.append(data['seed'])
            used.append({'file': str(path), 'seed': data['seed'], 'match': f"{s['a']} vs {s['b']}",
                         'treatment_pair_wins': w, 'treatment_pair_losses': l,
                         'p': sign_test(w, l)})
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f'direct matches share a seed {seeds}: the same openings would be '
                         'pooled as independent pairs; rerun them with distinct --seed')
    p = sign_test(wins, losses)
    winner = None
    if p < P_LEVEL:
        winner = 'treatment' if wins > losses else 'control'
    return {'matches': used, 'treatment_pair_wins': wins, 'treatment_pair_losses': losses,
            'p_pairs_two_sided': p, 'winner': winner}


def stability(run: Path, start: int, end: int) -> dict:
    games = load_games(run, start, end)
    if not games:
        raise SystemExit(f'no self-play records in {run} for [{start}, {end})')
    w = window_stats(sorted(games), games, load_generation_events(run))
    return {k: w[k] for k in ('from', 'to', 'games', 'mean_length', 'short_share',
                              'black_share', 'one_sided_share', 'mean_abs_color_margin',
                              'reuse')}


def _heavy_games(run: Path, generation: int) -> dict:
    data = _load(run / 'external_eval' / f'gen{generation:03d}_heavy.json')
    out = {}
    for opponent, block in data['opponents'].items():
        for index, game in enumerate(block['games']):
            opening = tuple(map(tuple, game['moves'][:game.get('opening_plies', 0)]))
            out[(generation, opponent, index)] = (game['result'] == 'win', opening,
                                                  game['model_color'])
    return out


def heavy_non_inferiority(control: Path, treatment: Path, points: list[int],
                          margin: float) -> dict:
    diffs = []
    for generation in points:
        c, t = _heavy_games(control, generation), _heavy_games(treatment, generation)
        if c.keys() != t.keys():
            raise SystemExit(f'heavy gen {generation}: the arms played different game sets')
        for key in c:
            if c[key][1:] != t[key][1:]:
                raise SystemExit(f'heavy gen {generation} {key}: different opening or colour '
                                 '(different --seed?); games cannot be paired')
            diffs.append(int(t[key][0]) - int(c[key][0]))
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    lower = mean - Z_ONE_SIDED_95 * sqrt(var / n)
    return {'paired_games': n, 'treatment_minus_control': mean, 'lower_bound_95': lower,
            'margin': margin, 'non_inferior': lower > -margin,
            'treatment_only_wins': diffs.count(1), 'control_only_wins': diffs.count(-1)}


def probe_regression(control: Path, treatment: Path, generation: int) -> dict:
    control_better = treatment_better = 0
    for suffix in PROBE_SUFFIXES:
        path = f'gen{generation:03d}{suffix}.json'
        if not (control / 'probes' / path).is_file():
            continue
        c = {r['id']: r for r in _load(control / 'probes' / path)['rows'] if 'top1' in r}
        t = {r['id']: r for r in _load(treatment / 'probes' / path)['rows'] if 'top1' in r}
        for key in c.keys() & t.keys():
            control_better += c[key]['top1'] and not t[key]['top1']
            treatment_better += t[key]['top1'] and not c[key]['top1']
    p = sign_test(control_better, treatment_better)
    return {'generation': generation, 'control_only_top1': control_better,
            'treatment_only_top1': treatment_better, 'p_two_sided': p,
            'regression': control_better > treatment_better and p < P_LEVEL}


def verdict(direct: dict, stable_c: dict, stable_t: dict, heavy: dict, probes: dict,
            stability_margin: float) -> tuple[str, dict]:
    checks = {
        'one_sided_lower': stable_t['one_sided_share'] <= stable_c['one_sided_share']
        - stability_margin,
        'color_margin_lower': stable_t['mean_abs_color_margin']
        < stable_c['mean_abs_color_margin'],
        'heavy_non_inferior': heavy['non_inferior'],
        'no_probe_regression': not probes['regression'],
    }
    if direct['winner'] == 'treatment':
        return 'adopt_stronger', checks
    if direct['winner'] == 'control':
        return 'reject', checks
    return ('adopt_stability' if all(checks.values()) else 'keep_control'), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--from', dest='start', type=int, required=True,
                        help='first self-play generation of the comparison (branch point)')
    parser.add_argument('--to', dest='end', type=int, required=True)
    parser.add_argument('--points', type=int, nargs='+', required=True,
                        help='heavy-evaluation generations; probes use the last one')
    parser.add_argument('--direct', nargs='+', required=True, help='head-to-head JSON (globs ok)')
    parser.add_argument('--control-prefix', default='CTL')
    parser.add_argument('--treatment-prefix', default='LR')
    parser.add_argument('--stability-margin', type=float, default=0.15)
    parser.add_argument('--non-inferiority', type=float, default=0.05)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.direct for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f'no direct-match files match {args.direct}')
    direct = direct_result(paths, args.treatment_prefix, args.control_prefix)
    if not direct['matches']:
        raise SystemExit('no treatment-vs-control match found (check the label prefixes)')
    stable_c = stability(args.control, args.start, args.end)
    stable_t = stability(args.treatment, args.start, args.end)
    heavy = heavy_non_inferiority(args.control, args.treatment, args.points,
                                  args.non_inferiority)
    probes = probe_regression(args.control, args.treatment, args.points[-1])
    decision, checks = verdict(direct, stable_c, stable_t, heavy, probes,
                               args.stability_margin)

    for m in direct['matches']:
        print(f"direct {m['match']} (seed {m['seed']}): pairs {m['treatment_pair_wins']}-"
              f"{m['treatment_pair_losses']} for the treatment, p={m['p']:.3g}")
    print(f"direct pooled: {direct['treatment_pair_wins']}-{direct['treatment_pair_losses']}, "
          f"p={direct['p_pairs_two_sided']:.3g}")
    for name, s in (('control', stable_c), ('treatment', stable_t)):
        print(f"{name:9s} {s['from']}-{s['to']}: one-sided {s['one_sided_share']:.0%}, "
              f"colour margin {s['mean_abs_color_margin']:.3f}, <=10 plies "
              f"{s['short_share']:.0%}, mean length {s['mean_length']:.1f}")
    print(f"heavy paired {heavy['paired_games']}: treatment - control "
          f"{heavy['treatment_minus_control']:+.3f}, 95% lower bound "
          f"{heavy['lower_bound_95']:+.3f} (margin -{heavy['margin']:.2f})")
    print(f"probes gen {probes['generation']}: control-only top1 {probes['control_only_top1']}, "
          f"treatment-only {probes['treatment_only_top1']}, p={probes['p_two_sided']:.3g}")
    print('checks:', checks)
    print('verdict:', decision)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'verdict': decision, 'checks': checks, 'direct': direct,
            'stability': {'control': stable_c, 'treatment': stable_t}, 'heavy': heavy,
            'probes': probes}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
