"""S1 probes: P92 / P93 / P94 from the web-play loss, per arm and seed (docs/mcts-v8-teacher.md §12.18-§12.19).

Analysis only: each probe position is replayed and the V8 arm picks one move; nothing is
played on. Per run it records the route, the played move, the root children (visits, mean
value), the V8-A/B/C statuses and, for policy arms, the raw policy probability of the moves
P92 compares. Coordinates in the output are 1-indexed like the doc.

P92 (black to move, truth UNRESOLVED) is a ranking probe: does the arm prefer a move in the
bottom-right response region R (rows 9-15, cols 9-15) over the far move (3,4) it lost with?
P93 (white to move, white has a proven VCT2 win) and P94 (black to move, every move proven
lost at VCT1) are sanity probes: their routes and statuses are recorded, no value is judged
(there is no value net before H6).

    python scripts/run_s1_probes.py --arms full puct_heur puct_policy --seeds 10 --puct-c 1.5 \\
        --policy-checkpoint runs/h3_policy_64x4/best.pt --output runs/s1/probes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.mcts_v8_agent import MCTSV8Agent  # noqa: E402
from analysis.mcts_v8 import V8_DEFAULTS  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import (  # noqa: E402
    ARMS, _git_commit, derive_seed, load_policy, needs_policy, provenance, v8_config,
)

DEFAULT_PROBES = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'
FORMAT = 's1-probes-v1'
FAR_MOVE = (2, 3)  # (3,4) 1-indexed: black's losing move at ply 93
REGION = frozenset((r, c) for r in range(8, 15) for c in range(8, 15))  # rows/cols 9-15, 1-indexed
SATURATION_EPS = 0.05


def one(move) -> list[int] | None:
    return None if move is None else [move[0] + 1, move[1] + 1]


def replay(moves, plies: int) -> Game:
    game = Game()
    for move in moves[:plies]:
        game.play(*move)
    return game


def root_order(children) -> list[tuple]:
    """Root children in the tree's own preference order (visits, mean value, move), as V8-C uses."""
    return sorted(children, key=lambda c: (-c[1], -c[2], tuple(c[0])))


def p92_metrics(children, played, priors=None) -> dict:
    """Ranking of the best region-R child against the far move; ``children`` = (move, visits, mean)."""
    order = root_order([(tuple(m), v, q) for m, v, q in children])
    rank = {m: i + 1 for i, (m, _, _) in enumerate(order)}
    visits = {m: v for m, v, _ in order}
    region = [m for m, _, _ in order if m in REGION]
    best_r = region[0] if region else None
    far_rank = rank.get(FAR_MOVE)
    total = sum(visits.values()) or 1
    return {
        'chose_far': tuple(played) == FAR_MOVE,
        'chose_region': tuple(played) in REGION,
        'root_has_region': best_r is not None,
        'region_best': one(best_r),
        'region_best_rank': rank.get(best_r),
        'far_rank': far_rank,  # None: (3,4) is not a root child
        'region_beats_far_rank': best_r is not None and (far_rank is None or rank[best_r] < far_rank),
        'region_beats_far_visits': best_r is not None and visits[best_r] > visits.get(FAR_MOVE, 0),
        'region_visit_share': round(sum(visits[m] for m in region) / total, 4),
        'far_visit_share': round(visits.get(FAR_MOVE, 0) / total, 4),
        'region_best_prior': None if priors is None or best_r is None else round(priors.get(best_r, 0.0), 5),
        'far_prior': None if priors is None else round(priors.get(FAR_MOVE, 0.0), 5),
    }


def root_value_spread(children) -> dict:
    """C4 baseline: spread of the root children's mean values (rollout leaves now, value net in H6)."""
    values = [q for _, _, q in children]
    if not values:
        return {'range': None, 'std': None, 'saturated': None}
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    return {'range': round(spread, 4), 'std': round((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 4),
            'saturated': spread <= SATURATION_EPS}


def run_probe(name: str, spec: dict, moves, arm: str, seed: int, config: dict, policy) -> dict:
    game = replay(moves, spec['plies_played'])
    agent = MCTSV8Agent(seed=derive_seed(seed, 's1', name, arm), root_policy=policy,
                        **{k: v for k, v in config.items() if k in V8_DEFAULTS})
    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    d = agent.diagnostics
    children = [(tuple(m), v, q) for m, v, q in d.v8_root_visits]
    priors = policy(game) if policy is not None and name == 'P92' else None
    record = {
        'probe': name, 'arm': arm, 'seed': seed, 'seconds': round(seconds, 3),
        'route': d.v8_route, 'played': one(move), 'tree_move': one(d.v8_v7_move), 'changed': d.v8_changed,
        'root_children': [[one(m), v, round(q, 4)] for m, v, q in root_order(children)],
        'root_value_spread': root_value_spread(children),
        'root_checked': [[one(m), s] for m, s in d.v8_root_checked],
        'root_exhausted': d.v8_root_budget_exhausted,
        'vct_checked': [[one(m), s] for m, s in d.v8_vct_checked],
        'vct_widened': d.v8_vct_widened, 'vct_exhausted': d.v8_vct_budget_exhausted,
        'attack_status': d.v8_attack_status, 'attack_candidates': d.v8_attack_candidates,
        'tree_mode': d.v8_tree_mode,
    }
    if name == 'P92':
        record['p92'] = p92_metrics(children, move, priors)
    return record


def summarize(records: list[dict]) -> dict:
    out = {}
    for arm in sorted({r['arm'] for r in records}):
        rows = {}
        for name in ('P92', 'P93', 'P94'):
            runs = [r for r in records if r['arm'] == arm and r['probe'] == name]
            if not runs:
                continue
            row = {'runs': len(runs), 'routes': {}, 'played': {}}
            for r in runs:
                row['routes'][r['route']] = row['routes'].get(r['route'], 0) + 1
                key = f"{r['played'][0]},{r['played'][1]}"
                row['played'][key] = row['played'].get(key, 0) + 1
            spreads = [r['root_value_spread']['saturated'] for r in runs if r['root_value_spread']['saturated'] is not None]
            row['saturated_root_runs'] = sum(spreads)
            if name == 'P92':
                m = [r['p92'] for r in runs]
                row.update({
                    'chose_far': sum(x['chose_far'] for x in m),
                    'chose_region': sum(x['chose_region'] for x in m),
                    'root_has_region': sum(x['root_has_region'] for x in m),
                    'region_beats_far_rank': sum(x['region_beats_far_rank'] for x in m),
                    'region_beats_far_visits': sum(x['region_beats_far_visits'] for x in m),
                    'region_visit_share_mean': round(sum(x['region_visit_share'] for x in m) / len(m), 4),
                    'far_visit_share_mean': round(sum(x['far_visit_share'] for x in m) / len(m), 4),
                })
                row['gate'] = {  # §12.19 S1 decision, conditions 1-2 (P92)
                    'chose_far_le_2_of_10': row['chose_far'] * 10 <= 2 * len(m),
                    'region_beats_far_both_ge_8_of_10':
                        sum(x['region_beats_far_rank'] and x['region_beats_far_visits'] for x in m) * 10 >= 8 * len(m),
                }
            if name == 'P94':
                row['all_checked_unsafe'] = sum(bool(r['vct_checked'] or r['root_checked'])
                                                and all(s == 'UNSAFE' for _, s in r['vct_checked'] + r['root_checked'])
                                                for r in runs)
            rows[name] = row
        out[arm] = rows
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--arms', nargs='+', default=['full', 'puct_heur', 'puct_policy'], choices=sorted(ARMS))
    parser.add_argument('--names', nargs='+', default=['P92', 'P93', 'P94'], choices=['P92', 'P93', 'P94'])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--base-seed', type=int, default=9201)
    parser.add_argument('--puct-c', type=float, default=1.5, help='c_puct for puct_* arms (H5 fixed 1.5)')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--simulations', type=int, help='smoke tests only')
    parser.add_argument('--tactical-simulations', type=int, help='smoke tests only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    probe_file = json.loads(args.probes.read_text(encoding='utf-8'))
    moves = [tuple(m) for m in probe_file['moves']]
    overrides = {k: v for k, v in (('simulations', args.simulations),
                                   ('tactical_simulations', args.tactical_simulations)) if v is not None}
    policy = None
    if any(needs_policy(v8_config(arm)) for arm in args.arms):
        try:
            policy = load_policy(args.policy_checkpoint)
        except (OSError, ValueError, ImportError) as exc:
            parser.error(f'policy arm: {exc}')
    records = []
    for arm in args.arms:
        config = v8_config(arm, overrides)
        if config.get('tree_mode') == 'puct':
            config['puct_c'] = args.puct_c
        arm_policy = policy if needs_policy(config) else None
        for name in args.names:
            for index in range(args.seeds):
                record = run_probe(name, probe_file['probes'][name], moves, arm,
                                   derive_seed(args.base_seed, index), config, arm_policy)
                records.append(record)
                print(f"{arm} {name} seed#{index} route={record['route']} played={record['played']} "
                      f"{record['seconds']:.1f}s", flush=True)
    summary = summarize(records)
    payload = {'format': FORMAT, 'git_commit': _git_commit(),
               'provenance': provenance(args.policy_checkpoint if policy is not None else None),
               'probes': str(args.probes), 'arms': args.arms, 'seeds': args.seeds, 'base_seed': args.base_seed,
               'puct_c': args.puct_c, 'overrides': overrides, 'summary': summary, 'runs': records}
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
