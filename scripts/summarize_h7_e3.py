"""H7-E3: TB-V2-Value against TB-Engine-v2 with guard v2 unchanged (docs/h6-value.md, H7-E3).

    python scripts/summarize_h7_e3.py --main runs/h7_e3/value_vs_tb2_8407.json \\
        --aux runs/h7_e3/value_vs_full_8407.json runs/h7_e3/tb2_vs_full_8407.json \\
        --d2 docs/h7/h7_e3_d2_8407.json --p92 docs/h7/h7_e3_p92.json --output docs/h7/h7_e3_summary_8407.json

H6-6 on the adopted engine: A = TB-V2-Value (``puct_value_vct2_50k_tt``) vs TB-Engine-v2
(``v8:puct_policy_vct2_50k_tt``); B, C = each vs ``v8:full`` on the same openings; every run ``--fast fast3``.
Guard v2 is ``summarize_h6_6``'s, number for number (``RULES`` is imported, the decision is the same
function of the same inputs). Two things follow the engine, nothing else:
    proof      the official 4M recheck is TB-Engine-v2's solver at 4M (``h6_5d.py d2 --table``: the
               call-local no-win table per VCF call, the same budgets); the frozen 4M recheck of the same
               moves (``--legacy-frozen``) gives the same two proof guards as a diagnostic only
    P92        ``run_s1_probes.py``'s ``h7_e3_p92_guard`` (TB-V2-Value vs TB-Engine-v2, same rule)
Stages (A score, 0.5 = equal to TB-Engine-v2): 8407 -> 8407+8408 -> +8409 once, as 8404 / 8405 / 8406 in H6-6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import summarize_h6_6 as h66  # noqa: E402
from h6_5d import replay_kind  # noqa: E402
from summarize_h5 import merge, summarize  # noqa: E402

CANDIDATE, BASELINE = 'puct_value_vct2_50k_tt', 'puct_policy_vct2_50k_tt'
MAIN_OPPONENT, AUX_OPPONENT = f'v8:{BASELINE}', 'v8:full'
SEEDS = (8407, 8408, 8409)
STAGES = ([8407], [8407, 8408], [8407, 8408, 8409])
RULES = dict(h66.RULES)  # guard v2, unchanged
H66_SEED = dict(zip(SEEDS, h66.SEEDS))  # the stage of each seed in H6-6's decision function


def check_run(run: dict, arms: tuple, opponent: str) -> None:
    if run['arm'] not in arms or run['opponent'] != opponent:
        raise ValueError(f"expected {' / '.join(arms)} vs {opponent}, got {run['arm']} vs {run['opponent']}")
    if run['seed'] not in SEEDS:
        raise ValueError(f"seed {run['seed']} is not an H7-E3 seed {SEEDS}")
    if not run['v8_config'].get('root_vcf_table'):
        raise ValueError(f"{run['arm']} seed {run['seed']}: not on TB-Engine-v2 (root_vcf_table off)")
    if run['v8_config'].get('puct_c') != 1.5 or (opponent == MAIN_OPPONENT and run.get('opponent_puct_c') != 1.5):
        raise ValueError(f"{run['arm']} vs {opponent} seed {run['seed']}: c_puct must be 1.5 on every PUCT side")
    if run.get('fast', {}).get('requested') != RULES['fast']:
        raise ValueError(f"{run['arm']} vs {opponent} seed {run['seed']}: not run with --fast {RULES['fast']}")


def proof_status(runs: list[dict], d2_files: list[dict], key: str = 'large') -> dict[str, dict[str, list[str]]]:
    """arm -> opening -> 4M status of each tree move (``key``: 'large' = 4M + table, the official recheck;
    'legacy_large' = frozen 4M). Raises when a recheck is missing, not on the table, or inconsistent."""
    large = {}
    for d2 in d2_files:
        if not d2.get('table') or not d2.get('legacy_frozen'):
            raise ValueError('every E3 d2 recheck must run with --table --legacy-frozen')
        if d2.get('fast', {}).get('requested') != RULES['fast']:
            raise ValueError('every d2 recheck must run with --fast fast3')
        bad = [r['id'] for r in d2['rows']
               if replay_kind(r['recorded_status'], r['standard']['status'], r['large']['status']) == 'mismatch']
        if bad:
            raise ValueError(f'd2 is not consistent with the recorded statuses: {bad[:3]}')
        for r in d2['rows']:
            large[r['id']] = r[key]['status']
    out = {}
    for run in runs:
        for g in run['games']:
            opening = g['key'].split('/', 1)[1]
            statuses = out.setdefault(run['arm'], {}).setdefault(opening, [])
            for m in g['v8_moves']:
                if m['route'] != 'tree':
                    continue
                status = h66._tree_status(m)
                if status != 'SAFE':
                    k = f"{run['arm']}|{g['key']}|{m['ply']}"
                    if k not in large:
                        raise ValueError(f'no 4M recheck for {k}')
                    status = large[k]
                statuses.append(status)
    return out


def proof_guard(statuses: dict, target: str) -> dict:
    """``summarize_h6_6.proof_guard`` with this experiment's arms."""
    return h66.proof_guard({h66.CANDIDATE: statuses.get(CANDIDATE, {}), h66.BASELINE: statuses.get(BASELINE, {})},
                           target)


def decide(main: dict, aux: dict, proofs: dict, p92: str | None) -> dict:
    """``summarize_h6_6.decide`` on the same inputs, seeds 8407 / 8408 / 8409 in the places of 8404 / 8405 / 8406."""
    seeds = sorted(main['seeds'])
    if seeds not in STAGES:
        return {'decision': 'INCOMPLETE', 'reasons': [f'seeds {seeds} are not 8407 / 8407+8408 / +8409']}
    if set(aux) != {CANDIDATE, BASELINE}:
        return {'decision': 'INCOMPLETE', 'reasons': ['both aux runs are needed on the same seeds']}
    mapped = {**main, 'seeds': [H66_SEED[s] for s in seeds]}
    aux66 = {h66.CANDIDATE: {**aux[CANDIDATE], 'seeds': [H66_SEED.get(s, s) for s in aux[CANDIDATE]['seeds']]},
             h66.BASELINE: {**aux[BASELINE], 'seeds': [H66_SEED.get(s, s) for s in aux[BASELINE]['seeds']]}}
    out = h66.decide(mapped, aux66, proofs, p92)
    names = {'RUN_8405': 'RUN_8408', 'INCONCLUSIVE': 'INCONCLUSIVE'}
    reasons = [r.replace('8404', '8407').replace('8405', '8408').replace('8406', '8409') for r in out['reasons']]
    return {'decision': names.get(out['decision'], out['decision']), 'reasons': reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--main', nargs='+', type=Path, required=True)
    parser.add_argument('--aux', nargs='+', type=Path, required=True)
    parser.add_argument('--d2', nargs='+', type=Path, required=True, help='h6_5d.py d2 --table --legacy-frozen outputs')
    parser.add_argument('--p92', type=Path, required=True, help='run_s1_probes.py output with h7_e3_p92_guard')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    main_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.main]
    aux_runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.aux]
    try:
        for run in main_runs:
            check_run(run, (CANDIDATE,), MAIN_OPPONENT)
        for run in aux_runs:
            check_run(run, (CANDIDATE, BASELINE), AUX_OPPONENT)
        main_row = summarize(merge(main_runs, MAIN_OPPONENT)[CANDIDATE])
        aux = {arm: summarize(a) for arm, a in merge(aux_runs, AUX_OPPONENT).items()}
        d2 = [json.loads(p.read_text(encoding='utf-8')) for p in args.d2]
        statuses = proof_status(aux_runs, d2)
        legacy = proof_status(aux_runs, d2, 'legacy_large')
    except ValueError as exc:
        parser.error(str(exc))
    proofs = {t: proof_guard(statuses, t) for t in ('UNSAFE', 'UNKNOWN')}
    legacy_proofs = {t: proof_guard(legacy, t) for t in ('UNSAFE', 'UNKNOWN')}
    p92 = json.loads(args.p92.read_text(encoding='utf-8'))['summary'].get('h7_e3_p92_guard', {}).get('verdict')
    result = {'format': 'h7-e3-summary-v1', 'rules': RULES, 'main': main_row, 'aux': aux,
              'proof_4m_table': proofs, 'proof_4m_legacy_frozen': legacy_proofs, 'p92_guard': p92,
              'verdict': decide(main_row, aux, proofs, p92)}
    print(json.dumps({'score': main_row['score'], 'pairs_ci95': main_row['ci95_pairs'], 'by_colour': main_row['by_colour'],
                      'time_ratio': main_row['time_ratio'], 'proof_4m_table': proofs,
                      'proof_4m_legacy_frozen (report only)': legacy_proofs, 'verdict': result['verdict']}, indent=1),
          flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
