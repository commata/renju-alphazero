"""E3 judgment: one E3 arm against S3-VCT2-v1 on the same seeds and opponent (§12.32).

Baseline arm ``puct_policy_vct2`` (S3-VCT2-v1); tested arm ``puct_policy_vct2_50k`` (E3-A) or
``puct_policy_vct2_stage`` (E3-S); opponent ``v8:full+vct2atk`` (200k); seeds 8415/8416.

Every veto of the tested arm (a tree or stage move whose VCT2 check proved the played move lost)
is verified after the games (``e3_dev_replay.verify``: selective 200k + the full depth 0-2 class):
the played move decides RESCUED / LOSS_TO_LOSS / SWITCH_UNRESOLVED / KEPT, and a vetoed move whose
full class is VCT2_CLEAR counts as unsound.

Reading (fixed before the runs, §12.32), first match wins:
    SAFETY_VIOLATION   a safety violation in either arm, or an unsound veto
    HARM               paired diff < -0.05, or its 95% upper bound < 0
    EFFICACY           paired diff 95% lower bound > 0
    MECHANISM_ESTABLISHED_BUT_MATCH_UNPROVEN
                       the arm vetoed more often than the baseline, at least one RESCUED, no
                       LOSS_TO_LOSS, paired diff > 0 (its interval contains 0)
    NOT_ESTABLISHED    otherwise
Cost is reported beside the reading: the tested engine's end-to-end time ratio must be <= 1.5
(``cost_ok``) for the arm to be adopted, whatever the reading.

    python scripts/e3_compare.py --baseline runs/e3/s3_8415.json runs/e3/s3_8416.json \\
        --arm runs/e3/e3a_8415.json runs/e3/e3a_8416.json --workers 6 \\
        --jsonl runs/e3/e3a_verify.jsonl --output runs/e3/e3a_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'scripts', ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from s3_compare import MAX_RATIO, MIN_DIFF, _dist, _seconds, opponent_time, punishment  # noqa: E402
from summarize_h5 import merge, paired, summarize  # noqa: E402
from scripts.e1_build_suite import CLEAR, _run_jobs  # noqa: E402
from scripts.e3_dev_replay import DEFAULT_TRUTH_BUDGET, VERIFY_BUDGET, verify  # noqa: E402

BASELINE = 'puct_policy_vct2'
ARMS = ('puct_policy_vct2_50k', 'puct_policy_vct2_stage')
OPPONENT = 'v8:full+vct2atk'


def vetoes(arm: dict) -> list[dict]:
    """Moves whose VCT2 check (tree or stage) proved the move it looked at first lost."""
    out = []
    for g in arm['games']:
        for m in g['v8_moves']:
            for kind, rec in (('tree', m.get('vct2')), ('stage', m.get('stage_vct2'))):
                if rec and rec.get('checked') and rec['checked'][0][1] == 'UNSAFE':
                    out.append({'key': f"{g['key']}/{m['ply']}", 'kind': kind, 'history': g['moves'][:m['ply']],
                                'vetoed': rec['checked'][0][0], 'played': g['moves'][m['ply']],
                                'switched': bool(rec.get('switched')), 'result': g['result']})
    return out


def _verify_task(task):
    event, truth_budget = task
    final = verify(event['history'], event['played'], truth_budget)
    vetoed = final if event['vetoed'] == event['played'] else verify(event['history'], event['vetoed'], truth_budget)
    return {'key': event['key'], 'final': final, 'vetoed': vetoed,
            'budget': {'verify': VERIFY_BUDGET, 'truth': truth_budget}}


def event_outcome(event: dict, row: dict) -> str:
    if not event['switched']:
        return 'KEPT'
    final = row['final']
    if final['selective_200k'] == 'UNSAFE' or final['full']['status'] == 'PROVEN_LOSS':
        return 'LOSS_TO_LOSS'
    return 'RESCUED' if final['full']['status'] == CLEAR else 'SWITCH_UNRESOLVED'


def reading(rows: dict, diff: dict, base_vetoes: int, arm_vetoes: int, outcomes: dict, unsound: int) -> str:
    if unsound or any(r['safety_violations'] for r in rows.values()):
        return 'SAFETY_VIOLATION'
    low, high = diff['ci95']
    if diff['diff'] is None or diff['diff'] < MIN_DIFF or high < 0:
        return 'HARM'
    if low > 0:
        return 'EFFICACY'
    if (arm_vetoes > base_vetoes and outcomes.get('RESCUED', 0) >= 1 and outcomes.get('LOSS_TO_LOSS', 0) == 0
            and diff['diff'] > 0):
        return 'MECHANISM_ESTABLISHED_BUT_MATCH_UNPROVEN'
    return 'NOT_ESTABLISHED'


def compare(base_runs, arm_runs, *, verified: dict, opponent=OPPONENT) -> dict:
    base_arms, arm_arms = merge(base_runs, opponent), merge(arm_runs, opponent)
    if len(base_arms) != 1 or len(arm_arms) != 1:
        raise ValueError('give one arm per side')
    base, arm = next(iter(base_arms.values())), next(iter(arm_arms.values()))
    if base['arm'] != BASELINE or arm['arm'] not in ARMS:
        raise ValueError(f"expected {BASELINE} against one of {ARMS}, got {base['arm']} / {arm['arm']}")
    if sorted(base['seeds']) != sorted(arm['seeds']):
        raise ValueError(f"seeds differ: {base['seeds']} vs {arm['seeds']}")
    rows = {'baseline': summarize(base), 'arm': summarize(arm)}
    diff = paired(arm, base)
    events = vetoes(arm)
    outcomes, unsound = {}, 0
    for e in events:
        row = verified[e['key']]
        e['outcome'] = event_outcome(e, row)
        outcomes[e['outcome']] = outcomes.get(e['outcome'], 0) + 1
        unsound += row['vetoed']['full']['status'] == CLEAR
    base_vetoes = len(vetoes(base))
    ratio = round(sum(_seconds(arm)) / sum(_seconds(base)), 3)
    return {
        'opponent': opponent, 'arm': arm['arm'], 'seeds': sorted(arm['seeds']), 'rows': rows,
        'paired_diff_arm_minus_baseline': diff,
        'mechanism': {'baseline_vetoes': base_vetoes, 'arm_vetoes': len(events),
                      'arm_vetoes_by_kind': {k: sum(e['kind'] == k for e in events) for k in ('tree', 'stage')},
                      'outcomes': outcomes, 'unsound': unsound,
                      'events': [{k: v for k, v in e.items() if k != 'history'} for e in events]},
        'punishment': {'baseline': punishment(base), 'arm': punishment(arm)},
        'cost': {'end_to_end_ratio': ratio, 'cost_ok': ratio <= MAX_RATIO,
                 'move_seconds': {'baseline': _dist(_seconds(base)), 'arm': _dist(_seconds(arm))}},
        'opponent_time': {'baseline': opponent_time(base), 'arm': opponent_time(arm)},
        'reading': reading(rows, diff, base_vetoes, len(events), outcomes, unsound),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--baseline', type=Path, nargs='+', required=True)
    parser.add_argument('--arm', type=Path, nargs='+', required=True)
    parser.add_argument('--opponent', default=OPPONENT)
    parser.add_argument('--truth-budget', type=int, default=DEFAULT_TRUTH_BUDGET['node_budget'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path, help='verification rows (resume)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))  # noqa: E731
    base_runs, arm_runs = [load(p) for p in args.baseline], [load(p) for p in args.arm]
    truth_budget = {**DEFAULT_TRUTH_BUDGET, 'node_budget': args.truth_budget}
    events = vetoes(next(iter(merge(arm_runs, args.opponent).values())))
    verified = _run_jobs([(e, truth_budget) for e in events], _verify_task, args.workers, args.jsonl,
                         lambda t: t[0]['key'], {'verify': VERIFY_BUDGET, 'truth': truth_budget})
    result = compare(base_runs, arm_runs, verified=verified, opponent=args.opponent)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows',)}, indent=1, default=str)[:20000], flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
