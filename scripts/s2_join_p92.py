"""S2: join the P92 truth with the raw policy and the arms' root candidates (docs/mcts-v8-teacher.md §12.23).

Inputs (all committed): ``s2_p92_truth.json`` (``s2_position_truth.py``), ``s2_policy_diag.json``
(full P92 policy distribution) and the S1 probe outputs (root children per arm and seed).
Output: per move its truth status, policy rank and probability, region R membership and
how many seeds each arm had it as a root child; totals such as the policy mass on proven
losses. Coordinates are 1-indexed.

    python scripts/s2_join_p92.py --output docs/mcts-v8-results/s2_p92_joined.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402

RESULTS = ROOT / 'docs' / 'mcts-v8-results'
PROBES = [RESULTS / f's1_probes_{arm}.json' for arm in ('full', 'puct_heur', 'puct_policy')]


def in_region(move) -> bool:
    return 9 <= move[0] <= 15 and 9 <= move[1] <= 15


def join(truth: dict, policy: dict, probes: list[dict], name: str = 'P92', top: int = 12) -> dict:
    dist = policy['probes'][name]['distribution']
    prob = {tuple(m): p for m, p in dist}
    rank = {tuple(m): i + 1 for i, (m, _) in enumerate(dist)}
    roots = {}
    for result in probes:
        for run in result['runs']:
            if run['probe'] == name:
                arm = roots.setdefault(run['arm'], {'runs': 0, 'counts': {}})
                arm['runs'] += 1
                for m, _, _ in run['root_children']:
                    arm['counts'][tuple(m)] = arm['counts'].get(tuple(m), 0) + 1
    rows = []
    for row in truth['moves']:
        move = tuple(row['move'])
        rows.append({'move': list(move), 'status': row['status'], 'lost_depth': row['lost_depth'],
                     'policy_rank': rank.get(move), 'policy_prob': prob.get(move, 0.0),
                     'region': in_region(move),
                     'root_seeds': {arm: info['counts'].get(move, 0) for arm, info in sorted(roots.items())}})
    rows.sort(key=lambda r: (r['policy_rank'] is None, r['policy_rank'] or 0))
    lost = [r for r in rows if r['status'] == 'PROVEN_LOSS']
    unknown = [r for r in rows if r['status'] == 'UNKNOWN']
    region = [r for r in rows if r['region']]
    totals = {
        'legal_moves': len(rows), 'proven_loss': len(lost), 'unknown': len(unknown),
        'other': len(rows) - len(lost) - len(unknown),
        'lost_depths': {str(d): sum(r['lost_depth'] == d for r in lost) for d in (0, 1, 2)},
        'policy_mass_proven_loss': round(sum(r['policy_prob'] for r in lost), 6),
        'policy_mass_unknown': round(sum(r['policy_prob'] for r in unknown), 6),
        f'policy_top{top}_all_proven_loss': all(r['status'] == 'PROVEN_LOSS' for r in rows[:top]),
        'best_policy_rank_not_proven_loss': min((r['policy_rank'] for r in rows if r['status'] != 'PROVEN_LOSS'),
                                                default=None),
        'region_moves': len(region), 'region_proven_loss': sum(r['status'] == 'PROVEN_LOSS' for r in region),
        'root_runs': {arm: info['runs'] for arm, info in sorted(roots.items())},
        'unknown_moves': [{k: r[k] for k in ('move', 'policy_rank', 'policy_prob', 'root_seeds')} for r in unknown],
    }
    legacy = sum('budget' not in row for row in truth['moves'])
    provenance = {'truth_rows_with_budget': len(truth['moves']) - legacy, 'truth_legacy_rows': legacy,
                  'truth_budget': truth.get('budget'), 'truth_git_commit': truth.get('git_commit'),
                  'truth_resume': truth.get('resume')}
    return {'totals': totals, 'provenance': provenance, 'moves': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--truth', type=Path, default=RESULTS / 's2_p92_truth.json')
    parser.add_argument('--policy-diag', type=Path, default=RESULTS / 's2_policy_diag.json')
    parser.add_argument('--probe-results', type=Path, nargs='*', default=PROBES)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))  # noqa: E731
    joined = join(load(args.truth), load(args.policy_diag), [load(p) for p in args.probe_results])
    inputs = [args.truth, args.policy_diag, *args.probe_results]
    payload = {'format': 's2-p92-joined-v1', 'git_commit': _git_commit(),
               'inputs': {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): file_sha256(p) for p in inputs},
               **joined}
    print(json.dumps(joined['totals'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
