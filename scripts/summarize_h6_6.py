"""H6-6: TB-V1-Value against TB-Engine-v1 with guard v2, fixed before the games (docs/h6-value.md, H6-6).

    python scripts/summarize_h6_6.py --main runs/h6_6/value_vs_tb1_8404.json \\
        --aux runs/h6_6/value_vs_full_8404.json runs/h6_6/tb1_vs_full_8404.json \\
        --d2 docs/h6/h6_6_d2_8404.json --p92 docs/h6/h6_4b_s1_probes.json --output docs/h6/h6_6_summary_8404.json

Same engines and runs as H6-5 (A: candidate vs TB-Engine-v1; B, C: each vs v8:full on the same openings),
every run with ``--fast fast3`` (exact speed-up, the same for both engines). Seeds 8404, 8405, reserve 8406.
Every standard not-SAFE tree move of B and C is rechecked offline with V8-C at 4M (``h6_5d.py d2 --fast
fast3``); a tree move that was SAFE stays SAFE. Per tree move this gives V8-C@4M SAFE / UNSAFE / UNKNOWN.

Guard v2 (RULES):
    hard       safety violations 0 in every run (any stage)                                   -> REJECT
    basic      A: black and white >= 0.45, time ratio <= 1.5; P92 guard not FAIL               -> REJECT
    proof      for 4M UNSAFE and for 4M UNKNOWN separately: rate difference candidate (B) - baseline (C),
               paired bootstrap over openings (seed / pair / colour), 95% interval:
               upper <= +0.03 PASS, lower > +0.03 REJECT, else INCONCLUSIVE
Stages (A score, 0.5 = equal to TB-Engine-v1):
    8404            hard / basic fail or score < 0.50 -> REJECT; else RUN_8405 (proof reported, not applied)
    8404+8405       hard / basic fail, score < 0.55 or a proof REJECT -> REJECT;
                    score >= 0.55, pair bootstrap low > 0.5 and both proofs PASS -> ADOPT;
                    otherwise INCONCLUSIVE -> run 8406 once
    +8406           the same on three seeds; INCONCLUSIVE -> REJECT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h6_5d import replay_kind  # noqa: E402
from summarize_h5 import merge, summarize  # noqa: E402

CANDIDATE, BASELINE = 'puct_value_vct2_50k', 'puct_policy_vct2_50k'
MAIN_OPPONENT, AUX_OPPONENT = f'v8:{BASELINE}', 'v8:full'
SEEDS = (8404, 8405, 8406)
STAGES = ([8404], [8404, 8405], [8404, 8405, 8406])
RULES = {'stage1_min': 0.50, 'adopt_min': 0.55, 'pair_low_above': 0.5, 'proof_margin': 0.03,
         'time_ratio_max': 1.5, 'colour_min': 0.45, 'fast': 'fast3', 'bootstrap': 10_000, 'seed': 0}


def _tree_status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['v7_move']), 'UNCHECKED')


def check_run(run: dict, arms: tuple, opponent: str) -> None:
    if run['arm'] not in arms or run['opponent'] != opponent:
        raise ValueError(f"expected {' / '.join(arms)} vs {opponent}, got {run['arm']} vs {run['opponent']}")
    if run['seed'] not in SEEDS:
        raise ValueError(f"seed {run['seed']} is not an H6-6 seed {SEEDS}")
    if run['v8_config'].get('puct_c') != 1.5 or (opponent == MAIN_OPPONENT and run.get('opponent_puct_c') != 1.5):
        raise ValueError(f"{run['arm']} vs {opponent} seed {run['seed']}: c_puct must be 1.5 on every PUCT side")
    if run.get('fast', {}).get('requested') != RULES['fast']:
        raise ValueError(f"{run['arm']} vs {opponent} seed {run['seed']}: not run with --fast {RULES['fast']}")


def proof_status(runs: list[dict], d2_files: list[dict]) -> dict[str, dict[str, list[str]]]:
    """arm -> opening (seed/pair/colour) -> V8-C@4M status of each tree move. Raises when a standard
    not-SAFE tree move has no 4M recheck or d2 did not reproduce a recorded status."""
    large = {}
    for d2 in d2_files:
        # recomputed from the rows with h6_5d.replay_kind (a stored summary may predate it)
        bad = [r['id'] for r in d2['rows']
               if replay_kind(r['recorded_status'], r['standard']['status'], r['large']['status']) == 'mismatch']
        if bad:
            raise ValueError(f'd2 is not consistent with the recorded statuses: {bad[:3]}')
        if d2.get('fast', {}).get('requested') != RULES['fast']:
            raise ValueError('every d2 recheck must run with --fast fast3')
        for r in d2['rows']:
            large[r['id']] = r['large']['status']
    out = {}
    for run in runs:
        for g in run['games']:
            opening = g['key'].split('/', 1)[1]
            statuses = out.setdefault(run['arm'], {}).setdefault(opening, [])
            for m in g['v8_moves']:
                if m['route'] != 'tree':
                    continue
                status = _tree_status(m)
                if status != 'SAFE':
                    key = f"{run['arm']}|{g['key']}|{m['ply']}"
                    if key not in large:
                        raise ValueError(f'no 4M recheck for {key}')
                    status = large[key]
                statuses.append(status)
    return out


def proof_guard(statuses: dict, target: str) -> dict:
    """Rate of ``target`` among tree moves, candidate - baseline, paired bootstrap over openings."""
    cand, base = statuses.get(CANDIDATE, {}), statuses.get(BASELINE, {})
    keys = sorted(set(cand) & set(base))
    if not keys:
        return {'verdict': 'INCOMPLETE'}

    def rate(side, ks):
        n = sum(len(side[k]) for k in ks)
        return sum(s == target for k in ks for s in side[k]) / n if n else 0.0

    def diff(ks):
        return rate(cand, ks) - rate(base, ks)

    rng = Random(RULES['seed'])
    b = RULES['bootstrap']
    draws = sorted(diff([rng.choice(keys) for _ in keys]) for _ in range(b))
    low, high = draws[int(0.025 * b)], draws[int(0.975 * b) - 1]
    margin = RULES['proof_margin']
    verdict = 'PASS' if high <= margin else 'REJECT' if low > margin else 'INCONCLUSIVE'
    return {'target': target, 'openings': len(keys), 'candidate_rate': round(rate(cand, keys), 4),
            'baseline_rate': round(rate(base, keys), 4), 'diff': round(diff(keys), 4),
            'ci95': [round(low, 4), round(high, 4)], 'margin': margin, 'verdict': verdict}


def decide(main: dict, aux: dict, proofs: dict, p92: str | None) -> dict:
    seeds = sorted(main['seeds'])
    if seeds not in STAGES:
        return {'decision': 'INCOMPLETE', 'reasons': [f'seeds {seeds} are not 8404 / 8404+8405 / +8406']}
    if set(aux) != {CANDIDATE, BASELINE} or any(sorted(a['seeds']) != seeds for a in aux.values()):
        return {'decision': 'INCOMPLETE', 'reasons': ['both aux runs are needed on the same seeds']}
    if p92 is None or any(p['verdict'] == 'INCOMPLETE' for p in proofs.values()):
        return {'decision': 'INCOMPLETE', 'reasons': ['the P92 guard and the 4M rechecks are needed']}
    fails = []
    if any(r['safety_violations'] for r in [main, *aux.values()]):
        fails.append('safety violation')
    for colour, score in main['by_colour'].items():
        if score is not None and score < RULES['colour_min']:
            fails.append(f"{colour} {score} < {RULES['colour_min']}")
    if main['time_ratio'] is not None and main['time_ratio'] > RULES['time_ratio_max']:
        fails.append(f"time ratio {main['time_ratio']} > {RULES['time_ratio_max']}")
    if p92 == 'FAIL':
        fails.append('P92 guard FAIL')
    if fails:
        return {'decision': 'REJECT', 'reasons': fails}
    score, low = main['score'], main['ci95_pairs'][0]
    if seeds == [8404]:
        if score < RULES['stage1_min']:
            return {'decision': 'REJECT', 'reasons': [f"8404 score {score} < {RULES['stage1_min']}"]}
        return {'decision': 'RUN_8405', 'reasons': ['proof guards are reported, applied from 8404+8405 on']}
    rejected = [f"4M {p['target']} lower {p['ci95'][0]} > +{RULES['proof_margin']}" for p in proofs.values()
                if p['verdict'] == 'REJECT']
    if score < RULES['adopt_min'] or rejected:
        return {'decision': 'REJECT',
                'reasons': ([f"score {score} < {RULES['adopt_min']}"] if score < RULES['adopt_min'] else []) + rejected}
    open_items = [f"4M {p['target']} interval {p['ci95']} not below +{RULES['proof_margin']}"
                  for p in proofs.values() if p['verdict'] != 'PASS']
    if low <= RULES['pair_low_above']:
        open_items.insert(0, f'pair bootstrap low {low} <= 0.5')
    if not open_items:
        return {'decision': 'ADOPT', 'reasons': []}
    if seeds == [8404, 8405]:
        return {'decision': 'INCONCLUSIVE', 'reasons': open_items + ['run 8406 once']}
    return {'decision': 'REJECT', 'reasons': open_items + ['still open on three seeds']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--main', nargs='+', type=Path, required=True)
    parser.add_argument('--aux', nargs='+', type=Path, required=True)
    parser.add_argument('--d2', nargs='+', type=Path, required=True, help='h6_5d.py d2 outputs on the aux runs')
    parser.add_argument('--p92', type=Path, required=True, help='run_s1_probes.py output with h6_5_p92_guard')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    main_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.main]
    aux_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.aux]
    for run in main_runs:
        check_run(run, (CANDIDATE,), MAIN_OPPONENT)
    for run in aux_runs:
        check_run(run, (CANDIDATE, BASELINE), AUX_OPPONENT)
    main_row = summarize(merge(main_runs, MAIN_OPPONENT)[CANDIDATE])
    aux = {arm: summarize(a) for arm, a in merge(aux_runs, AUX_OPPONENT).items()}
    statuses = proof_status(aux_runs, [json.loads(p.read_text(encoding='utf-8')) for p in args.d2])
    proofs = {t: proof_guard(statuses, t) for t in ('UNSAFE', 'UNKNOWN')}
    p92 = json.loads(args.p92.read_text(encoding='utf-8'))['summary'].get('h6_5_p92_guard', {}).get('verdict')
    result = {'format': 'h6-6-summary-v1', 'rules': RULES, 'main': main_row, 'aux': aux, 'proof_4m': proofs,
              'p92_guard': p92, 'verdict': decide(main_row, aux, proofs, p92)}
    print(json.dumps({'score': main_row['score'], 'pairs_ci95': main_row['ci95_pairs'], 'by_colour': main_row['by_colour'],
                      'time_ratio': main_row['time_ratio'], 'proof_4m': proofs, 'verdict': result['verdict']}, indent=1),
          flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
