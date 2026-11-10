"""R2-D4-PRECISION-1: how precise the two D4 readings were, what a worthwhile effect would cost to detect, and what
to do next. Read-only: no training, no Rapfi, no change to any reading (docs/h7r/r2/R2-D4-PRECISION-1-draft.md).

    summary    from the committed D4 summaries and run logs only: each comparison's game-bootstrap standard error,
               the games needed for the pre-fixed effects, the measured cost, and the pre-fixed proposal rule
    journals   on the desktop, from the D1 D4 run journals (replay.jsonl, branches.jsonl): rebuild every observation
               and branch outcome exactly as the D4 tool did and require the committed statistics back; then the
               per-game concentration of observations and decisive observations, a cluster (ratio) and a jackknife
               standard error, the design effect against independent observations, the largest one-game influence,
               and the projection again with the largest of the three standard errors

The effect sizes, power, budget and the proposal rule are fixed in this file and in the draft before the journal
analysis is run; they do not come from the D4 point estimates.

    python scripts/h7r_d4_precision.py summary --output docs/h7r/r2/R2-D4-PRECISION-1/summary.json
    python scripts/h7r_d4_precision.py journals --run-dir runs/h7r/R2-D1-D4-1/full \\
        --cache runs/h7r/R1-HDEV-1/cache --output docs/h7r/r2/R2-D4-PRECISION-1/journals.json
"""
from __future__ import annotations

import argparse
from math import ceil, sqrt
import json
from pathlib import Path
import re
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FORMAT = 'h7r-d4-precision-v1'
# --- fixed before the journal analysis (draft section 2) -------------------------------------------------------
EFFECTS = (0.02, 0.03, 0.05)          # mean continuation-score delta per disagreement observation
PRIMARY_EFFECT = 0.03                 # the Arena's 3 %p minimum improvement / about one disagreement per game
POWER = 0.80
ALPHA_ONE_SIDED = 0.025               # label95: the 95 % interval above 0
BUDGET_HOURS = 24.0                   # desktop time an evaluation expansion may cost to be proposed
PRIMARY_K = 3                         # ALL_THREE_BETTER needs all three primary pairs
# --- measured inputs --------------------------------------------------------------------------------------------
G0 = 396                              # sealed H-dev games in both D4 runs
SELFPLAY_MINUTES_PER_400 = 39.3       # R1-HDEV-1 generation (docs/h7-rl-plan.md 9.14); 396 of 400 usable
STUDIES = {
    'D1': {'summary': 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4/full/summary.json',
           'logs': 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4/full/run-logs', 'tag': 'd1-d4',
           'primary': ('wide31001:narrow31001', 'wide31002:narrow31002', 'wide31003:narrow31003'), 'models': 7},
    'BASELINE': {'summary': 'docs/h7r/r2/R2-D4-BASELINE-1/records/full/summary.json',
                 'logs': 'docs/h7r/r2/R2-D4-BASELINE-1/records/full/run-logs', 'tag': 'd4',
                 'primary': ('seed31001:c5', 'seed31002:c5', 'seed31003:c5'), 'models': 5},
}
Z = NormalDist().inv_cdf
Z_ALPHA = Z(1 - ALPHA_ONE_SIDED)


# --- precision and sizing ---------------------------------------------------------------------------------------

def bootstrap_se(cell: dict) -> float:
    """The committed 95 % game-bootstrap interval as a standard error (the wider of the two bounds' intervals)."""
    lo = cell['intervals']['0.95']['lower']['ci'][0]
    hi = cell['intervals']['0.95']['upper']['ci'][1]
    return (hi - lo) / (2 * Z_ALPHA)


def z_power(power: float, k: int = 1, bound: str = 'single') -> float:
    """Per-pair z for ``power``: one pair; all ``k`` pairs if independent (power ** (1 / k) each); all ``k`` if
    perfectly dependent (the single-pair z)."""
    if bound == 'independent':
        return Z(power ** (1 / k))
    return Z(power)


def games_needed(se: float, delta: float, z: float, g0: int = G0) -> int:
    """Games for a one-sided ALPHA test of ``delta`` with per-pair z, the standard error scaling as 1 / sqrt(games)."""
    return ceil(g0 * ((Z_ALPHA + z) * se / delta) ** 2)


def mde(se: float, z: float, games: int, g0: int = G0) -> float:
    """The effect detectable with per-pair z at ``games``."""
    return (Z_ALPHA + z) * se * sqrt(g0 / games)


def phase_minutes(log_text: str, tag: str) -> dict:
    """Replay and Rapfi minutes of one D4 run log (timestamped lines, first to last tagged line)."""
    spans = {}
    for line in log_text.splitlines():
        m = re.match(r'^(\d+\.\d+) \[' + re.escape(tag) + r'-(replay|rapfi)\]', line)
        if m:
            t = float(m.group(1))
            lo, hi = spans.get(m.group(2), (t, t))
            spans[m.group(2)] = (min(lo, t), max(hi, t))
    if set(spans) != {'replay', 'rapfi'}:
        raise ValueError(f'the log has no replay / Rapfi phases for [{tag}-...]')
    return {k: round((hi - lo) / 60, 2) for k, (lo, hi) in spans.items()}


def cost(study: dict, root=ROOT) -> dict:
    logs = sorted((Path(root) / study['logs']).glob('*.log'))
    if len(logs) != 1:
        raise ValueError(f"{study['logs']}: expected the one full-run log, found {len(logs)}")
    minutes = phase_minutes(logs[0].read_text(encoding='utf-8', errors='replace'), study['tag'])
    selfplay = SELFPLAY_MINUTES_PER_400  # 400 generated for 396 usable (R1-HDEV-1)
    per_g0 = minutes['replay'] + minutes['rapfi'] + selfplay
    return {'log': logs[0].relative_to(root).as_posix(), 'replay_minutes': minutes['replay'],
            'rapfi_minutes': minutes['rapfi'], 'selfplay_minutes': round(selfplay, 1), 'models': study['models'],
            'hours_per_396_games': round(per_g0 / 60, 2),
            'not_included': 'H-dev labels and cache building for new games (not measured), sealing and records; the '
                            'cost is a lower bound'}


def projection(ses: dict, hours_per_g0: float) -> dict:
    """Per effect: games and hours for each primary pair at POWER, and for all three pairs (the largest SE) under
    the independent and the perfectly dependent bound."""
    worst = max(ses.values())
    out = {}
    for delta in EFFECTS:
        per_pair = {p: games_needed(se, delta, z_power(POWER)) for p, se in ses.items()}
        joint = {b: games_needed(worst, delta, z_power(POWER, PRIMARY_K, b)) for b in ('single', 'independent')}
        out[str(delta)] = {'per_pair_games': per_pair,
                           'all_three_games': {'dependent_bound': joint['single'],
                                               'independent_bound': joint['independent']},
                           'all_three_hours': {'dependent_bound': round(joint['single'] / G0 * hours_per_g0, 1),
                                               'independent_bound': round(joint['independent'] / G0 * hours_per_g0,
                                                                          1)}}
    budget_games = int(BUDGET_HOURS / hours_per_g0 * G0)
    out['mde'] = {'at_396_games': round(mde(worst, z_power(POWER, PRIMARY_K, 'independent'), G0), 4),
                  f'at_budget_{budget_games}_games': round(mde(worst, z_power(POWER, PRIMARY_K, 'independent'),
                                                               max(budget_games, 1)), 4),
                  'basis': 'all three pairs, independent bound, the largest primary SE'}
    return out


def proposal(proj: dict) -> dict:
    """The pre-fixed rule (draft section 4): expand the evaluation only if detecting PRIMARY_EFFECT on all three
    pairs (independent bound) fits BUDGET_HOURS; otherwise not, and the next D4 of a D2 / D5 candidate at the same
    sample could only establish effects of about the 396-game MDE."""
    hours = proj[str(PRIMARY_EFFECT)]['all_three_hours']['independent_bound']
    if hours <= BUDGET_HOURS:
        return {'evaluation_expansion': 'PROPOSE', 'hours': hours,
                'reason': f'detecting {PRIMARY_EFFECT} on all three pairs fits {BUDGET_HOURS} h'}
    return {'evaluation_expansion': 'DO_NOT_PROPOSE', 'hours': hours,
            'reason': f'detecting {PRIMARY_EFFECT} on all three pairs needs about {hours} h (> {BUDGET_HOURS} h), '
                      'a lower bound',
            'next_d4_mde_at_396_games': proj['mde']['at_396_games'],
            'note': 'the choice between D2 / D5 and Track A is the user\'s; this tool gives the cost and the MDE only'}


def summary_report(root=ROOT) -> dict:
    out = {'format': FORMAT, 'mode': 'summary',
           'fixed': {'effects': EFFECTS, 'primary_effect': PRIMARY_EFFECT, 'power': POWER,
                     'alpha_one_sided': ALPHA_ONE_SIDED, 'budget_hours': BUDGET_HOURS, 'primary_pairs': PRIMARY_K},
           'studies': {}}
    for name, study in STUDIES.items():
        s = json.loads((Path(root) / study['summary']).read_text(encoding='utf-8'))
        rows = {}
        for p, cell in s['comparisons'].items():
            rows[p] = {'observations': cell['observations'], 'observations_per_game': round(cell['observations'] / G0, 3),
                       'decisive': cell['decisive'], 'decisive_games': cell['decisive_games'],
                       'resolved_games': cell['resolved_games'], 'bootstrap_se': round(bootstrap_se(cell), 5),
                       'arena_3pp_as_d4_delta': round(0.03 / (cell['observations'] / G0), 4)}
        c = cost(study, root)
        ses = {p: rows[p]['bootstrap_se'] for p in study['primary']}
        proj = projection(ses, c['hours_per_396_games'])
        out['studies'][name] = {'summary': study['summary'], 'reading': s['reading'], 'comparisons': rows,
                                'cost': c, 'projection': proj, 'proposal': proposal(proj)}
    return out


# --- journals (desktop) -----------------------------------------------------------------------------------------

def strict_journal(path, field, identity_sha, expected) -> dict:
    """Every line a complete record of this run (read-only: a torn or foreign line is refused, nothing is written)."""
    out = {}
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        rec = json.loads(line)
        if rec.get('identity_sha256') != identity_sha or rec.get(field) not in expected or rec[field] in out:
            raise ValueError(f'{path}:{i + 1}: not a record of this run')
        out[rec[field]] = rec
    if set(out) != set(expected):
        raise ValueError(f'{path}: {len(out)} of {len(expected)} records')
    return out


def per_game(obs: list[dict], outcomes: dict, games: list) -> dict:
    """One comparison's observations and decisive observations by game, with concentration, cluster (ratio
    linearization) and jackknife standard errors, the design effect and the largest one-game influence."""
    idx = {g: i for i, g in enumerate(games)}
    n, s, dec = [0] * len(games), [0.0] * len(games), [0] * len(games)
    deltas = []
    for o in obs:
        a, b = outcomes[o['key_a']], outcomes[o['key_b']]
        if a['kind'] == 'unresolved' or b['kind'] == 'unresolved':
            raise ValueError('an unresolved branch: this analysis needs a run without them')
        d = a['score'] - b['score']
        i = idx[o['game_index']]
        n[i] += 1
        s[i] += d
        dec[i] += d != 0
        deltas.append(d)
    G, N, S = len(games), sum(n), sum(s)
    if N < 2:
        raise ValueError('too few observations')
    m = S / N
    cluster = sqrt(G / (G - 1) * sum((si - m * ni) ** 2 for si, ni in zip(s, n))) / N
    loo = [(S - si) / (N - ni) for si, ni in zip(s, n) if N - ni > 0]
    jack = sqrt((G - 1) / G * sum((x - sum(loo) / len(loo)) ** 2 for x in loo))
    mean_d = sum(deltas) / N
    iid = sqrt(sum((d - mean_d) ** 2 for d in deltas) / (N - 1) / N)
    order = sorted(range(G), key=lambda i: -n[i])
    top = max(1, round(0.1 * G))
    by_dec = sorted(range(G), key=lambda i: -dec[i])
    return {'games': G, 'observations': N, 'decisive': sum(dec), 'mean_delta': m,
            'games_with_observations': sum(x > 0 for x in n), 'games_with_decisive': sum(x > 0 for x in dec),
            'max_observations_in_a_game': max(n), 'max_decisive_in_a_game': max(dec),
            'top10pct_games_share_of_observations': sum(n[i] for i in order[:top]) / N,
            'top10pct_games_share_of_decisive': (sum(dec[i] for i in by_dec[:top]) / sum(dec)) if sum(dec) else None,
            'cluster_se': cluster, 'jackknife_se': jack, 'iid_se': iid,
            'design_effect': (cluster / iid) ** 2 if iid else None,
            'largest_one_game_influence': max(abs(m - x) for x in loo)}


def journals_report(args, root=ROOT) -> dict:
    from random import Random

    import h7c_adjudicate as ad
    import h7r_d1_d4 as dd
    import h7r_d4 as base
    from hybrid import h7r_d4 as d4

    D = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4'
    c = json.loads((Path(root) / f'{D}/d1-d4-contract.json').read_text(encoding='utf-8'))
    summary = json.loads((Path(root) / f'{D}/full/summary.json').read_text(encoding='utf-8'))
    run = Path(args.run_dir)
    identity = json.loads((run / 'identity.json').read_text(encoding='utf-8'))
    if identity != json.loads((Path(root) / f'{D}/full/identity.json').read_text(encoding='utf-8')):
        raise ValueError('the run identity differs from the committed one')
    identity_sha = d4.digest(identity)
    if summary['identity_sha256'] != identity_sha:
        raise ValueError('the committed summary is not of this run')
    manifest, games, _ = dd.inputs(c, dd.contract_base_sha256(c), args.cache)
    if manifest != json.loads((Path(root) / c['output']['sample']).read_text(encoding='utf-8')):
        raise ValueError('the sealed H-dev differs from the committed eligible manifest')
    replay = strict_journal(run / 'replay.jsonl', 'game_index', identity_sha, set(games))
    records = [replay[i] for i in sorted(games)]
    obs = dd.observations_for(base.eligible_replays(records, manifest), games)
    table = ad.branch_table(obs)
    keys = sorted(table)
    Random(c['stats']['order_seed']).shuffle(keys)
    tasks = {k: {**table[k], 'order': i, 'identity_sha256': identity_sha} for i, k in enumerate(keys)}
    done = strict_journal(run / 'branches.jsonl', 'key', identity_sha, set(keys))
    outcomes = {k: base.validate_branch(r, tasks[k], c['invalid_policy']) for k, r in done.items()}
    counts = d4.multiplicities(len(games), c['stats']['samples'], c['stats']['seed'])
    recomputed = {p: d4.selection_stats([o for o in obs if o['compare'] == p], outcomes, sorted(games), counts,
                                        c['stats']) for p in dd.COMPARISONS}
    for p in dd.REFERENCE:
        for k in ('label95', 'label_individual'):
            recomputed[p].pop(k)
    if json.loads(json.dumps(recomputed)) != summary['comparisons']:
        raise ValueError('the journals do not give the committed statistics back')
    study = STUDIES['D1']
    rows = {p: per_game([o for o in obs if o['compare'] == p], outcomes, sorted(games)) for p in dd.COMPARISONS}
    for p, row in rows.items():
        row['bootstrap_se'] = bootstrap_se(summary['comparisons'][p])
        row['se_used'] = max(row['bootstrap_se'], row['cluster_se'], row['jackknife_se'])
    c_ = cost(study, root)
    proj = projection({p: rows[p]['se_used'] for p in study['primary']}, c_['hours_per_396_games'])
    return {'format': FORMAT, 'mode': 'journals', 'run_identity_sha256': identity_sha,
            'reproduced': 'every observation and branch outcome rebuilt from the journals gives the committed '
                          'statistics back',
            'branches': len(outcomes), 'per_game': rows, 'cost': c_, 'projection': proj, 'proposal': proposal(proj)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--output', required=True)
    j = sub.add_parser('journals')
    j.add_argument('--run-dir', required=True)
    j.add_argument('--cache', required=True)
    j.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    out = Path(args.output)
    try:
        if out.exists():
            raise ValueError(f'{out} exists: never overwritten')
        report = summary_report() if args.command == 'summary' else journals_report(args)
        from hybrid.h7r_ledger import publish_json
        publish_json(out, report)
    except (ValueError, KeyError, OSError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    print(json.dumps({name: st['proposal'] for name, st in report['studies'].items()} if 'studies' in report
                     else report['proposal'], ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
