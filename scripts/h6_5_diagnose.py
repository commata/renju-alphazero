"""H6-5D: where the tree not-SAFE moves of an H6-5 run come from (recorded games only, no search).

Report only; it never changes an H6-5 decision. For each run (``run_mcts_v8_benchmark.py`` output) it splits
the tree moves whose V8-C status is not SAFE (UNKNOWN / UNSAFE, the H6-5 guard's count) by game result
(full-board draws apart), ply, V8-C budget exhaustion and the status of the move finally played, and gives
the share of tree moves equal to the policy prior's top child. For a candidate / baseline pair on the same
openings it adds a paired bootstrap of the not-SAFE rate difference over games (moves of one game are not
independent).

    python scripts/h6_5_diagnose.py --candidate runs/h6_5/value_vs_full_8403.json \\
        --baseline runs/h6_5/tb1_vs_full_8403.json --output docs/h6/h6_5d_8403.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

BOOTSTRAP = 10_000
PLY_BUCKETS = ((0, 19), (20, 39), (40, 59), (60, 999))
NOT_SAFE = ('UNKNOWN', 'UNSAFE')


def tree_status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['v7_move']), 'UNCHECKED')


def played_status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['played']), 'UNCHECKED')


def game_kind(game) -> str:
    if game['result'] == 'draw':
        return 'draw_full_board' if game['length'] >= 225 else 'draw'
    return 'decisive'


def _rate(n, d):
    return {'not_safe': n, 'tree_moves': d, 'rate': round(n / d, 4) if d else None}


def per_game(run: dict) -> dict[str, list[int]]:
    """Opening key (seed / pair / colour) -> [tree moves, not-SAFE tree moves]."""
    out = {}
    for g in run['games']:
        tree = [m for m in g['v8_moves'] if m['route'] == 'tree']
        out[g['key'].split('/', 1)[1]] = [len(tree), sum(tree_status(m) in NOT_SAFE for m in tree)]
    return out


def describe(run: dict) -> dict:
    rows = [(g, m) for g in run['games'] for m in g['v8_moves'] if m['route'] == 'tree']
    bad = [(g, m) for g, m in rows if tree_status(m) in NOT_SAFE]
    with_top = [m for _, m in rows if m.get('tree', {}).get('prior_top') is not None]
    counts = sorted((c for _, c in per_game(run).values()), reverse=True)
    played = {}
    for _, m in bad:
        s = played_status(m)
        played[s] = played.get(s, 0) + 1
    return {
        'arm': run['arm'], 'opponent': run['opponent'], 'seed': run['seed'],
        'tree_status': {s: sum(tree_status(m) == s for _, m in rows) for s in ('SAFE', 'UNKNOWN', 'UNSAFE')},
        'all': _rate(len(bad), len(rows)),
        'by_game_kind': {k: _rate(sum(game_kind(g) == k for g, _ in bad), sum(game_kind(g) == k for g, _ in rows))
                         for k in ('decisive', 'draw_full_board', 'draw')},
        'by_ply': {f'{lo}-{hi}': _rate(sum(lo <= m['ply'] <= hi for _, m in bad), sum(lo <= m['ply'] <= hi for _, m in rows))
                   for lo, hi in PLY_BUCKETS},
        'v8c_budget_exhausted': {'not_safe': sum(m['root']['exhausted'] for _, m in bad),
                                 'tree_moves': sum(m['root']['exhausted'] for _, m in rows)},
        'played_status_of_not_safe': played,
        'v8c_switch_of_not_safe': {k or 'none': sum((m['root'].get('switch') or '') == k for _, m in bad)
                                   for k in ('proven_loss', 'unknown', '')},
        'not_safe_per_game_top': counts[:8],
        'games_with_not_safe': sum(c > 0 for c in counts),
        'tree_prior_top_rate': round(sum(m['v7_move'] == m['tree']['prior_top'] for m in with_top) / len(with_top), 4)
        if with_top else None,
        'not_safe_move_seconds_median': sorted(m['seconds'] for _, m in bad)[len(bad) // 2] if bad else None,
        'tree_move_seconds_median': sorted(m['seconds'] for _, m in rows)[len(rows) // 2] if rows else None,
    }


def paired_rate_bootstrap(candidate: dict, baseline: dict, margin: float = 0.03, seed: int = 0) -> dict:
    a, b = per_game(candidate), per_game(baseline)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {'games': 0}

    def diff(ks):
        da, db = sum(a[k][0] for k in ks), sum(b[k][0] for k in ks)
        return (sum(a[k][1] for k in ks) / da if da else 0.0) - (sum(b[k][1] for k in ks) / db if db else 0.0)

    rng = Random(seed)
    draws = sorted(diff([rng.choice(keys) for _ in keys]) for _ in range(BOOTSTRAP))
    return {'games': len(keys), 'diff': round(diff(keys), 4),
            'ci95': [round(draws[int(0.025 * BOOTSTRAP)], 4), round(draws[int(0.975 * BOOTSTRAP) - 1], 4)],
            'share_above_margin': round(sum(d > margin for d in draws) / BOOTSTRAP, 3), 'margin': margin}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='same seed and opponent (paired by opening)')
    parser.add_argument('--extra', nargs='*', type=Path, default=[], help='more runs to describe (e.g. the main run)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    cand = json.loads(args.candidate.read_text(encoding='utf-8'))
    result = {'format': 'h6-5d-v1', 'report_only': True, 'candidate': describe(cand)}
    if args.baseline:
        base = json.loads(args.baseline.read_text(encoding='utf-8'))
        if (base['seed'], base['opponent']) != (cand['seed'], cand['opponent']):
            parser.error('the baseline must have the same seed and opponent')
        result['baseline'] = describe(base)
        result['paired_not_safe_diff'] = paired_rate_bootstrap(cand, base)
    result['extra'] = [describe(json.loads(p.read_text(encoding='utf-8'))) for p in args.extra]
    text = json.dumps(result, indent=1) + '\n'
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
