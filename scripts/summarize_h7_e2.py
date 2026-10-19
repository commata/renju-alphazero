"""H7-E2: TB-Engine-v1-T against TB-Engine-v1 on the track-b-post-e1 §2.3 gate (docs/h6-value.md, H7-E2).

    python scripts/summarize_h7_e2.py --runs runs/h7_e2/tt_vs_full_8421.json runs/h7_e2/tb1_vs_full_8421.json \\
        --soundness docs/h7/h7_e2_soundness_8421.json --output docs/h7/h7_e2_summary_8421.json

Both engines play ``v8:full`` on the same openings (seed / pair / colour), every run with ``--fast fast3``;
the candidate's tree moves are rechecked offline by ``scripts/h7_e2_soundness.py``. Fixed before the games:

    soundness  (hard) no SAFE/UNSAFE contradiction with the frozen solver, every recorded UNSAFE replayed
               with valid VCF lines, no safety violation in either arm, configurations as frozen
    strength   score difference candidate - baseline per opening (both colours), bootstrap over openings:
               PASS lower >= -0.05; FAIL (clear) difference < -0.05
    frequency  per game, the share of tree moves recorded UNKNOWN; difference candidate - baseline per
               game pair (same opening and colour, both with a tree move), bootstrap over openings:
               PASS upper < 0 and every seed's raw mean difference < 0; FAIL (clear) the pooled or any
               seed's difference >= 0
    cost       total V8 move seconds candidate / baseline over the paired games: PASS <= 1.5, else FAIL
    isolation  off is the frozen path: tests (tests.test_tb_engine_v1, tests.test_h7_e2) and the
               baseline configuration equal to TB-Engine-v1 (checked here)
Stages:
    8421              soundness fail -> REJECT; else RUN_8422 (the rest reported)
    8421+8422         soundness or a clear FAIL -> REJECT; every gate PASS -> ADOPT; else RUN_8423
    8421+8422+8423    every gate PASS -> ADOPT (TB-Engine-v2); else REJECT
Reported only: per-move time percentiles (p50 / p90 / p95 / p99 / max), tree statuses, the frozen ->
candidate status transitions and the legacy frozen 4M rechecks, V8-C switches, identical games.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_h5 import SCORE, merge, summarize  # noqa: E402

CANDIDATE, BASELINE = 'puct_policy_vct2_50k_tt', 'puct_policy_vct2_50k'
OPPONENT = 'v8:full'
SEEDS = (8421, 8422, 8423)
STAGES = ([8421], [8421, 8422], [8421, 8422, 8423])
RULES = {'strength_low_min': -0.05, 'cost_max': 1.5, 'fast': 'fast3', 'bootstrap': 10_000, 'seed': 0,
         'pairs_per_seed': 50}


def _status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['v7_move']), 'UNCHECKED')


def check_run(run: dict) -> None:
    from analysis.tb_engine_v1 import TB_ENGINE_V1

    if run['arm'] not in (CANDIDATE, BASELINE) or run['opponent'] != OPPONENT:
        raise ValueError(f"expected {CANDIDATE} / {BASELINE} vs {OPPONENT}, got {run['arm']} vs {run['opponent']}")
    if run['seed'] not in SEEDS:
        raise ValueError(f"seed {run['seed']} is not an H7-E2 seed {SEEDS}")
    want = {**TB_ENGINE_V1, 'root_vcf_table': True} if run['arm'] == CANDIDATE else TB_ENGINE_V1
    if run['v8_config'] != want:
        raise ValueError(f"{run['arm']} seed {run['seed']}: configuration is not the frozen one")
    if run.get('fast', {}).get('requested') != RULES['fast']:
        raise ValueError(f"{run['arm']} seed {run['seed']}: not run with --fast {RULES['fast']}")
    table = any('table' in m['root'] for g in run['games'] for m in g['v8_moves'] if m['route'] == 'tree')
    if table != (run['arm'] == CANDIDATE):
        raise ValueError(f"{run['arm']} seed {run['seed']}: the table is {'on' if table else 'off'}")


def _games(arm: dict) -> dict:
    return {(g['run_seed'], g['pair'], g['v8_color']): g for g in arm['games']}


def _bootstrap(clusters: dict[tuple, list[float]]) -> dict:
    keys = sorted(k for k, v in clusters.items() if v)
    if not keys:
        return {'n': 0, 'openings': 0, 'diff': None, 'ci95': None}

    def mean_of(ks):
        vals = [x for k in ks for x in clusters[k]]
        return sum(vals) / len(vals)

    rng, b = Random(RULES['seed']), RULES['bootstrap']
    draws = sorted(mean_of([rng.choice(keys) for _ in keys]) for _ in range(b))
    return {'n': sum(len(clusters[k]) for k in keys), 'openings': len(keys), 'diff': round(mean_of(keys), 4),
            'ci95': [round(draws[int(0.025 * b)], 4), round(draws[int(0.975 * b) - 1], 4)]}


def strength(cand: dict, base: dict) -> dict:
    clusters = {}
    for key in sorted(set(cand) & set(base)):
        clusters.setdefault(key[:2], []).append(SCORE[cand[key]['result']] - SCORE[base[key]['result']])
    out = _bootstrap(clusters)
    out['verdict'] = ('INCOMPLETE' if out['diff'] is None else 'PASS' if out['ci95'][0] >= RULES['strength_low_min']
                      else 'FAIL' if out['diff'] < RULES['strength_low_min'] else 'OPEN')
    return out


def unknown_rate(game) -> float | None:
    tree = [_status(m) for m in game['v8_moves'] if m['route'] == 'tree']
    return sum(s == 'UNKNOWN' for s in tree) / len(tree) if tree else None


def frequency(cand: dict, base: dict) -> dict:
    clusters, per_seed = {}, {}
    for key in sorted(set(cand) & set(base)):
        a, b = unknown_rate(cand[key]), unknown_rate(base[key])
        if a is None or b is None:
            continue
        clusters.setdefault(key[:2], []).append(a - b)
        per_seed.setdefault(key[0], []).append(a - b)
    out = _bootstrap(clusters)
    out['per_seed'] = {str(s): sum(v) / len(v) for s, v in sorted(per_seed.items())}  # raw, not rounded
    if out['diff'] is None:
        out['verdict'] = 'INCOMPLETE'
    elif out['diff'] >= 0 or any(v >= 0 for v in out['per_seed'].values()):
        out['verdict'] = 'FAIL'
    elif out['ci95'][1] < 0:
        out['verdict'] = 'PASS'
    else:
        out['verdict'] = 'OPEN'
    return out


def cost(cand: dict, base: dict) -> dict:
    keys = sorted(set(cand) & set(base))
    c = sum(m['seconds'] for k in keys for m in cand[k]['v8_moves'])
    b = sum(m['seconds'] for k in keys for m in base[k]['v8_moves'])
    ratio = round(c / b, 3) if b else None
    return {'candidate_seconds': round(c, 1), 'baseline_seconds': round(b, 1), 'ratio': ratio,
            'verdict': 'INCOMPLETE' if ratio is None else 'PASS' if ratio <= RULES['cost_max'] else 'FAIL'}


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)

    def at(q):
        return round(v[min(len(v) - 1, int(q * len(v)))], 3)

    return {'moves': len(v), 'p50': at(0.50), 'p90': at(0.90), 'p95': at(0.95), 'p99': at(0.99), 'max': round(v[-1], 3)}


def report(arm: dict) -> dict:
    moves = [m for g in arm['games'] for m in g['v8_moves']]
    tree = [m for m in moves if m['route'] == 'tree']
    return {'move_seconds': percentiles([m['seconds'] for m in moves]),
            'tree_move_seconds': percentiles([m['seconds'] for m in tree]),
            'root_check_seconds': percentiles([m['root']['seconds'] for m in tree]),
            'v8c_switches': {k: sum(m['root'].get('switch') == k for m in tree) for k in ('proven_loss', 'unknown')},
            'table_hits': sum(m['root'].get('table', {}).get('hits', 0) for m in tree)}


def soundness(runs: list[dict], files: list[dict], seeds: list[int]) -> dict:
    covered = sorted({s for f in files for s in f['summary']['seeds']})
    reasons = []
    if covered != seeds:
        reasons.append(f'soundness rechecks cover seeds {covered}, not {seeds}')
    for f in files:
        if f.get('fast', {}).get('requested') != RULES['fast']:
            reasons.append('every soundness recheck must run with --fast fast3')
    rows = [r for f in files for r in f['rows']]
    tree = sum(1 for r in runs if r['arm'] == CANDIDATE for g in r['games'] for m in g['v8_moves']
               if m['route'] == 'tree')
    if len(rows) != tree:
        reasons.append(f'{len(rows)} rechecked tree moves, the candidate runs have {tree}')
    s = [f['summary'] for f in files]
    out = {'contradictions': [x for f in s for x in f['contradictions']],
           'unsafe_not_replayed': [x for f in s for x in f['unsafe_not_replayed']],
           'line_fail': sum(f['line_fail'] for f in s), 'unsafe': sum(f['unsafe'] for f in s),
           'transitions': {}, 'legacy_4m': {}, 'tool': reasons}
    for f in s:
        for name in ('transitions', 'legacy_4m'):
            for k, v in f[name].items():
                out[name][k] = out[name].get(k, 0) + v
    return out


def decide(seeds: list[int], sound: dict, safety: dict, gates: dict) -> dict:
    if seeds not in STAGES:
        return {'decision': 'INCOMPLETE', 'reasons': [f'seeds {seeds} are not 8421 / 8421+8422 / +8423']}
    if sound['tool'] or any(g['verdict'] == 'INCOMPLETE' for g in gates.values()):
        return {'decision': 'INCOMPLETE', 'reasons': sound['tool'] + [f'{k} incomplete' for k, g in gates.items()
                                                                       if g['verdict'] == 'INCOMPLETE']}
    hard = ([f"{len(sound['contradictions'])} SAFE/UNSAFE contradictions"] if sound['contradictions'] else []) + \
           ([f"{len(sound['unsafe_not_replayed'])} UNSAFE not replayed"] if sound['unsafe_not_replayed'] else []) + \
           ([f"{sound['line_fail']} VCF lines fail to replay"] if sound['line_fail'] else []) + \
           [f'{arm}: {n} safety violations' for arm, n in safety.items() if n]
    if hard:
        return {'decision': 'REJECT', 'reasons': hard}
    if seeds == [8421]:
        return {'decision': 'RUN_8422', 'reasons': ['soundness PASS; the other gates are reported, applied from 8421+8422']}
    fails = [f'{k} FAIL' for k, g in gates.items() if g['verdict'] == 'FAIL']
    if fails:
        return {'decision': 'REJECT', 'reasons': fails}
    open_items = [f'{k} open' for k, g in gates.items() if g['verdict'] != 'PASS']
    if not open_items:
        return {'decision': 'ADOPT', 'reasons': ['TB-Engine-v1-T becomes TB-Engine-v2']}
    if seeds == [8421, 8422]:
        return {'decision': 'RUN_8423', 'reasons': open_items}
    return {'decision': 'REJECT', 'reasons': open_items + ['still open on three seeds']}


def evaluate(runs: list[dict], sound_files: list[dict]) -> dict:
    for run in runs:
        check_run(run)
    arms = merge(runs, OPPONENT)
    if set(arms) != {CANDIDATE, BASELINE}:
        raise ValueError('both arms are needed')
    seeds = sorted(arms[CANDIDATE]['seeds'])
    if sorted(arms[BASELINE]['seeds']) != seeds:
        raise ValueError('both arms must cover the same seeds')
    cand, base = _games(arms[CANDIDATE]), _games(arms[BASELINE])
    if set(cand) != set(base):
        raise ValueError('the two arms did not play the same openings and colours')
    rows = {arm: summarize(a) for arm, a in arms.items()}
    gates = {'strength': strength(cand, base), 'frequency': frequency(cand, base), 'cost': cost(cand, base)}
    sound = soundness(runs, sound_files, seeds)
    safety = {arm: len(r['safety_violations']) for arm, r in rows.items()}
    identical = sum(cand[k]['moves'] == base[k]['moves'] for k in cand)
    return {'seeds': seeds, 'arms': rows, 'gates': gates, 'soundness': sound,
            'report': {arm: report(a) for arm, a in arms.items()} | {'identical_games': identical, 'games': len(cand)},
            'verdict': decide(seeds, sound, safety, gates)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--soundness', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    sound = [json.loads(p.read_text(encoding='utf-8')) for p in args.soundness]
    try:
        result = {'format': 'h7-e2-summary-v1', 'rules': RULES, **evaluate(runs, sound)}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({'seeds': result['seeds'], 'gates': result['gates'],
                      'soundness': {k: (len(v) if isinstance(v, list) else v) for k, v in result['soundness'].items()},
                      'verdict': result['verdict']}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
