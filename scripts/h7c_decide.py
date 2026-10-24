"""H7-c next-cycle decision from a cycle's diagnosis and the search-budget ablation (docs/h7-c1.md), pre-registered.

Reads ``h7c_diagnose.py`` (D2, D4) and ``h7c_budget.py summary`` outputs and the cycle's self-play summary, applies
the fixed rules below and writes which case holds. It decides nothing that the rules do not: every number is here.

    B-P  policy target trigger (D2): the visits rows' own cross-entropy reduction (n-weighted) is at most 0, or the
         proof_one_hot rows give at least ``proof_share_min`` of the validation cross-entropy reduction (they are about
         21% of the rows), or the visits rows' top-1 change (n-weighted over both sides) is at most 0
    B-V  value target trigger (D4 training metric): the relative WMSE gain on the rows the search sends to the network
         is at most 0, or below ``network_share_min`` of the gain on all rows
    A    the ablation reads BUDGET_HELPS (or S400 is STRONGER at 150 pairs): policy targets from a re-search of the
         self-play tree positions at the smallest STRONGER budget; affordable when the projected re-search time
         (tree positions x that budget's measured seconds per tree move / workers) is at most ``cost_multiple`` x the
         cycle's self-play wall time, else the choice goes to the user (A_COST_REVIEW)

    case   A                        when A holds (A before B: one variable per cycle, and A has direct game evidence)
           B_POLICY / B_VALUE       else when a B trigger fires (B-P before B-V: P-only lost more than V-only in c1)
           RUN_SATURATION           else when no budget was STRONGER and the S400 150-pair stage has not been run
           V9_ENTRY_E1              else when that stage reads SATURATED (MCTS-v9 entry condition E1)
           REPLICATE                else (the stage reads INCONCLUSIVE): the next cycle with the recipe unchanged

    python scripts/h7c_decide.py --diagnosis docs/h7c/c1/diagnosis.json --sb docs/h7c/sb_summary.json \\
        --selfplay docs/h7c/c1/selfplay_c1_seed8801.json --output docs/h7c/c1/decision.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = 'h7c-decision-v1'
RULES = {'proof_share_min': 0.5, 'visits_top1_delta_max': 0.0, 'network_share_min': 0.5, 'cost_multiple': 2.0,
         'workers': 6}


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def policy_trigger(d2: dict) -> dict:
    reduction = {k: -v['delta_ce'] * v['n'] for k, v in d2.items()}
    total = sum(reduction.values())
    proof = sum(v for k, v in reduction.items() if k.startswith('proof_one_hot/'))
    visits = {k: v for k, v in d2.items() if k.startswith('visits/')}
    n = sum(v['n'] for v in visits.values())
    top1 = sum(v['delta_top1'] * v['n'] for v in visits.values()) / n if n else None
    visits_reduction = sum(v for k, v in reduction.items() if k.startswith('visits/'))
    share = proof / total if total > 0 else None
    fired = visits_reduction <= 0 or (share is not None and share >= RULES['proof_share_min']) or (
        top1 is not None and top1 <= RULES['visits_top1_delta_max'])
    return {'ce_reduction_total': round(total, 4), 'visits_ce_reduction': round(visits_reduction, 4),
            'proof_share_of_reduction': None if share is None else round(share, 4),
            'visits_top1_delta': None if top1 is None else round(top1, 5), 'fired': fired}


def value_trigger(d4: dict) -> dict:
    routes = d4['training_metric']['by_route']
    overall, network = routes['all'].get('wmse_relative_improvement'), routes['network'].get('wmse_relative_improvement')
    fired = network is None or network <= 0 or (overall is not None and overall > 0
                                                and network < RULES['network_share_min'] * overall)
    return {'overall_relative_gain': overall, 'network_relative_gain': network, 'fired': fired}


def budget_case(sb: dict, selfplay: dict) -> dict:
    tree_positions = selfplay['summary']['routes'].get('tree', 0)
    wall = selfplay['wall_seconds']
    factor = None
    if sb['reading'] == 'BUDGET_HELPS':
        factor = min(r['factor'] for r in sb['arms'].values() if r['stronger'])
    elif (sb.get('saturation') or {}).get('reading') == 'STRONGER':
        factor = sb['saturation']['rule']['factor']
    if factor is None:
        return {'helps': False}
    arm = next(r for r in sb['arms'].values() if r['factor'] == factor)
    seconds = arm['cost']['tree_seconds_mean']
    projected = tree_positions * seconds / RULES['workers']
    ceiling = RULES['cost_multiple'] * wall
    return {'helps': True, 'factor': factor, 'budget': {'simulations': 50 * factor, 'tactical_simulations': 100 * factor},
            'tree_positions': tree_positions, 'tree_seconds_mean': seconds, 'projected_seconds': round(projected, 1),
            'ceiling_seconds': round(ceiling, 1), 'affordable': projected <= ceiling}


def decide(diagnosis: dict, sb: dict, selfplay: dict) -> dict:
    p, v, a = policy_trigger(diagnosis['parts']['d2']), value_trigger(diagnosis['parts']['d4']), budget_case(sb, selfplay)
    saturation = (sb.get('saturation') or {}).get('reading')
    if a['helps']:
        case = 'A' if a['affordable'] else 'A_COST_REVIEW'
    elif p['fired']:
        case = 'B_POLICY'
    elif v['fired']:
        case = 'B_VALUE'
    elif saturation is None:
        case = 'RUN_SATURATION'
    elif saturation == 'SATURATED':
        case = 'V9_ENTRY_E1'
    else:
        case = 'REPLICATE'
    return {'rules': RULES, 'sb_reading': sb['reading'], 'saturation': saturation, 'budget': a, 'policy_trigger': p,
            'value_trigger': v, 'case': case,
            'v9_e1': {'not_detected': sb['reading'] == 'NOT_DETECTED', 'saturated': saturation == 'SATURATED',
                      'no_b_trigger': not (p['fired'] or v['fired']), 'met': case == 'V9_ENTRY_E1'}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--sb', type=Path, required=True)
    parser.add_argument('--selfplay', type=Path, required=True, help="the cycle's self-play summary")
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    diagnosis, sb, selfplay = _json(args.diagnosis), _json(args.sb), _json(args.selfplay)
    if diagnosis.get('format') != 'h7c-diagnosis-v2' or sb.get('format') != 'h7c-budget-v1-summary' or \
            selfplay.get('format') != 'h7c-selfplay-v1-summary' or diagnosis['cycle'] != selfplay['identity']['cycle']:
        parser.error('the inputs are not a diagnosis, an ablation summary and the same cycle\'s self-play summary')
    missing = [part for part in ('d2', 'd4') if part not in diagnosis['parts']]
    if missing:
        parser.error(f'the diagnosis lacks {missing}')
    out = {'format': FORMAT, 'cycle': diagnosis['cycle'], **decide(diagnosis, sb, selfplay),
           'inputs_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for name, path in (('diagnosis', args.diagnosis), ('sb', args.sb),
                                                ('selfplay', args.selfplay))}}
    print(json.dumps({k: out[k] for k in ('case', 'sb_reading', 'saturation', 'policy_trigger', 'value_trigger',
                                           'budget')}, indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
