"""R2-D4 development diagnosis. Never a promotion, confirmation or training rule."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .h7r_r2 import require, file_sha256, is_sha, multiplicities
from .h7r_eval import check_invalid_policy, EXECUTION_FAILURES

FORMAT = 'h7r-d4-contract-v1'
PRIMARY = ('seed31001:c5', 'seed31002:c5', 'seed31003:c5')
COMPARISONS = (*PRIMARY, 'c8:c5')
STATS = {'samples': 20000, 'seed': 2026103102, 'order_seed': 2026103101,
         'min_observations': 20, 'min_resolved': 20, 'min_resolved_games': 10,
         'individual_level': 1 - .05 / 3, 'joint_level': .95}
POLICY = {'version': 'h7r-invalid-v1', 'structural_abnormal_end': 'evaluation_draw',
          'unresolved': 'worst_case', 'unresolved_reasons': list(EXECUTION_FAILURES),
          'max_invalid_share': .05, 'denominator': 'disagreement_observations'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_contract(path, root):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    require(c['format'] == FORMAT and c['contract_id'] == 'R2-D4-BASELINE-1'
            and c['status'] == 'fixed' and c['purpose'] == 'development_baseline_link', 'D4 contract not fixed')
    require(c['comparisons'] == list(COMPARISONS) and c['stats'] == STATS, 'D4 comparisons/statistics differ')
    check_invalid_policy(c['invalid_policy'])
    require(c['invalid_policy'] == POLICY, 'D4 invalid policy differs')
    require(c['selection'] == 'all_cache_games_and_rows' and c['budget'] == 'TB-V3-fast5-S50-original-actor-seeds'
            and c['device'] == 'cuda' and c['execution'] == {'prepare': True, 'smoke': False, 'full': False},
            'D4 scope differs')
    require(set(c['models']) == {'c5', 'c8', 'seed31001', 'seed31002', 'seed31003'}, 'D4 model set differs')
    for model in c['models'].values():
        require(all(is_sha(model[k]) for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',
                                             'value_checkpoint_sha256', 'value_metadata_sha256')), 'model SHA missing')
    for ref in [*c['sources'], *c['runtime_sources']]:
        p = Path(root) / ref['path']
        require(is_sha(ref['sha256']) and file_sha256(p) == ref['sha256'], f'D4 input changed: {p}')
    eps = json.loads((Path(root) / c['epsilon_result']).read_text())
    rep = json.loads((Path(root) / c['reproduction_result']).read_text())
    from .h7r_r2 import reproduction_verdict, REPRODUCED
    require(reproduction_verdict(rep['checks'], rep['tolerance']) == rep['verdict']
            and rep['verdict']['verdict'] in REPRODUCED and eps['epsilon']['status'] == 'COMPUTED'
            and eps['reproduction_sha256'] == file_sha256(Path(root) / c['reproduction_result'])
            and set(eps['seeds']) == {'31001', '31002', '31003'}, 'R2-0B prerequisite failed')
    return c, file_sha256(path)


def interval(draws, level):
    import torch
    defined = torch.isfinite(draws)
    undefined = int((~defined).sum())
    values = draws[defined]
    ci = torch.quantile(values, torch.tensor([(1-level)/2, (1+level)/2], dtype=torch.float64)).tolist() \
        if values.numel() else None
    return {'level': level, 'ci': ci, 'undefined': undefined}


def ratio(counts, numerator, denominator):
    import torch
    n = counts @ torch.tensor(numerator, dtype=torch.float64)
    d = counts @ torch.tensor(denominator, dtype=torch.float64)
    return n / d  # NaN denominators are reported, never silently omitted from a ruling.


def reading(result, level):
    if result['failures']:
        return 'INCONCLUSIVE'
    lo, hi = result['intervals'][str(level)]['lower'], result['intervals'][str(level)]['upper']
    if lo['undefined'] or hi['undefined'] or lo['ci'] is None or hi['ci'] is None:
        return 'INCONCLUSIVE'
    if lo['ci'][0] > 0:
        return 'BETTER'
    if hi['ci'][1] < 0:
        return 'WORSE'
    return 'INCONCLUSIVE'


def selection_stats(observations, outcomes, game_indices, counts, stats=STATS):
    """All disagreement observations, including ties and both bounds for unresolved branches.
    The common bootstrap includes source games with no disagreements, too.
    """
    n_games = len(game_indices)
    indices = {g: i for i, g in enumerate(game_indices)}
    require(len(indices) == n_games and counts.shape[1] == n_games, 'bootstrap game list differs')
    n, low, high, rn, rdelta, better, worse = ([0.] * n_games for _ in range(7))
    unresolved = structural = 0
    both = {'win': 0, 'draw': 0, 'loss': 0}
    for obs in observations:
        i = indices[obs['game_index']]
        a, b = (outcomes[obs[k]] for k in ('key_a', 'key_b'))
        require(a['kind'] in ('resolved', 'structural', 'unresolved')
                and b['kind'] in ('resolved', 'structural', 'unresolved'), 'unhandled branch')
        ua, ub = a['kind'] == 'unresolved', b['kind'] == 'unresolved'
        for o in (a, b):
            require(o['score'] is None if o['kind'] == 'unresolved' else o['score'] in (0, .5, 1), 'bad branch score')
        n[i] += 1
        low[i] += (0 if ua else a['score']) - (1 if ub else b['score'])
        high[i] += (1 if ua else a['score']) - (0 if ub else b['score'])
        unresolved += ua or ub
        structural += a['kind'] == 'structural' or b['kind'] == 'structural'
        if not (ua or ub):
            delta = a['score'] - b['score']
            rn[i] += 1
            rdelta[i] += delta
            better[i] += delta > 0
            worse[i] += delta < 0
            if a['score'] == b['score']:
                both[{1: 'win', .5: 'draw', 0: 'loss'}[a['score']]] += 1
    total, resolved = int(sum(n)), int(sum(rn))
    resolved_games = sum(x > 0 for x in rn)
    failures = []
    if total < stats['min_observations']: failures.append('TOO_FEW_OBSERVATIONS')
    if resolved < stats['min_resolved']: failures.append('TOO_FEW_RESOLVED')
    if resolved_games < stats['min_resolved_games']: failures.append('TOO_FEW_RESOLVED_GAMES')
    if total and Fraction(unresolved, total) > Fraction(1, 20): failures.append('UNRESOLVED_OVER_5_PERCENT')
    ldraw, hdraw = ratio(counts, low, n), ratio(counts, high, n)
    decisive_games = sum(b + w > 0 for b, w in zip(better, worse))
    out = {'observations': total, 'resolved': resolved, 'resolved_games': resolved_games,
           'decisive': int(sum(better) + sum(worse)), 'decisive_games': decisive_games,
           'unresolved': unresolved, 'structural_observations': structural, 'failures': failures,
           'mean_delta_bounds': [sum(low)/total, sum(high)/total] if total else None,
           'resolved_mean_delta': sum(rdelta)/resolved if resolved else None,
           'resolved_game_macro': sum(d/k for d, k in zip(rdelta, rn) if k)/resolved_games if resolved_games else None,
           'game_macro_bounds': [sum(d/k for d, k in zip(v, n) if k)/sum(k > 0 for k in n)
                                 for v in (low, high)] if total else None,
           'better': int(sum(better)), 'worse': int(sum(worse)),
           'equal': resolved - int(sum(better) + sum(worse)), 'both': both,
           'better_share': sum(better)/(sum(better)+sum(worse)) if sum(better)+sum(worse) else None,
           'better_share_ci95': interval(ratio(counts, better, [a+b for a, b in zip(better, worse)]), .95),
           'resolved_game_macro_ci95': interval(ratio(counts, [d/k if k else 0 for d, k in zip(rdelta, rn)],
                                                            [int(k > 0) for k in rn]), .95),
           'intervals': {str(level): {'lower': interval(ldraw, level), 'upper': interval(hdraw, level)}
                         for level in (stats['joint_level'], stats['individual_level'])}}
    if any(cell['undefined'] for level in out['intervals'].values() for cell in level.values()):
        failures.append('UNDEFINED_BOOTSTRAP')  # replicates without a denominator: the interval is not read
    out['label95'] = reading(out, stats['joint_level'])
    out['label_individual'] = reading(out, stats['individual_level'])
    out['upper95_descriptive'] = out['intervals'][str(stats['joint_level'])]['upper']['ci']
    if out['upper95_descriptive'] is not None:
        out['upper95_descriptive'] = out['upper95_descriptive'][1]
    return out


def joint_reading(comparisons):
    rows = [comparisons[name] for name in PRIMARY]
    if all(r['label95'] == 'BETTER' for r in rows): return 'ALL_THREE_BETTER'
    if all(r['label95'] == 'WORSE' for r in rows): return 'ALL_THREE_WORSE'
    labels = {r['label_individual'] for r in rows}
    if {'BETTER', 'WORSE'} <= labels: return 'MIXED_DIRECTION'
    if 'BETTER' in labels and 'WORSE' not in labels: return 'PARTIAL_SIGNAL'
    return 'NOT_ESTABLISHED'
