"""E1-holdout verdict: S3-VCT2-v1 against E3-A on the same holdout manifest (docs/mcts-v8-teacher.md §12.34).

Inputs: the holdout manifest (``e1_build_suite.py`` with ``--sources``) and two ``e1_evaluate.py`` outputs on
it, ``--arm s3`` and ``--arm e3a`` (same seeds per position, so both engines see the same tree).

Terms
    FALSE_VETO        a vetoed move whose 10M truth is VCT2_CLEAR
    UNSOUND_WITNESS   E3-A's proof of a vetoed move, re-derived at E3-A's budget, is not reproduced by the
                      full solver (``selective_vct.verify_witness`` False)
    UNRESOLVED        a verdict needs a truth that is UNKNOWN at 10M; reported, not counted as a failure

Gates (fixed before the runs)
    G1  E3-A UNSOUND_WITNESS = 0 and FALSE_VETO = 0
    G2  every (position, seed) S3 vetoed, E3-A vetoed too (same tree, same deterministic solver, larger cap:
        a violation is an implementation bug)
    G3  every (position, seed) that is RESCUED for S3 is RESCUED for E3-A (a violation is a bug, as G2)
    G4  E3-A runs that switched to a move whose truth is PROVEN_LOSS <= 1
    G5  E3-A veto_recall >= S3 and RESCUED >= S3 -- only when the suite has >= 5 E1-P positions
PASS when G1-G4 hold (and G5 when it applies): TB-Engine-v1 = E3-A. Otherwise FAIL: TB-Engine-v1 = S3-VCT2-v1.

    python scripts/e1_holdout_judge.py --manifest runs/e1h/e1h_manifest.json \\
        --s3 runs/e1h/eval_s3.json --e3a runs/e1h/eval_e3a.json --output runs/e1h/e1h_verdict.json
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

from renju import Game  # noqa: E402
from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, UNKNOWN  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256  # noqa: E402

MIN_E1P = 5
MAX_LOSS_SWITCH = 1


def _key(move) -> str:
    return f'{move[0]},{move[1]}'


def witness_ok(history, move, budget) -> bool | None:
    """Re-derive the selective depth-2 proof of ``move`` at ``budget`` and check it with the full solver."""
    from analysis.selective_vct import SelectiveSolver, _plain, _run, verify_witness
    from analysis.threats import UNSAFE

    game = Game()
    for m in history:
        game.play(*m)
    solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    status, witness = _run(solver, game, tuple(move), 2)
    if status != UNSAFE:
        return False  # the logged veto is not reproduced at all
    return verify_witness(game, tuple(move), {'status': PROVEN_LOSS, 'depth': 2, 'witness': _plain(witness)})


def _runs(evaluation: dict) -> dict:
    return {(row['key'], run['seed']): (row, run) for row in evaluation['rows'] for run in row['runs']}


def judge(manifest: dict, s3: dict, e3a: dict, *, witness=None) -> dict:
    if s3.get('arm', 's3') != 's3' or e3a.get('arm') != 'e3a':
        raise ValueError('give an --arm s3 evaluation and an --arm e3a evaluation')
    positions = {p['key']: p for p in manifest['positions']}
    a, b = _runs(s3), _runs(e3a)
    if set(a) != set(b):
        raise ValueError('the two evaluations do not cover the same (position, seed) runs')
    truths = {k: dict(p['truths']) for k, p in positions.items()}
    for evaluation in (s3, e3a):
        for row in evaluation['rows']:
            truths[row['key']].update(row.get('truths_added', {}))
    status = lambda key, m: (truths[key].get(_key(m)) or {}).get('status')  # noqa: E731
    vetoed = lambda run: bool(run['checked']) and run['checked'][0][1] == 'UNSAFE'  # noqa: E731

    false_veto, unsound, unresolved, g2, g3, loss_switch = [], [], [], [], [], []
    for (key, seed), (row, run) in b.items():
        _, base = a[(key, seed)]
        if vetoed(run):
            if status(key, run['pre']) == CLEAR:
                false_veto.append([key, seed])
            elif witness is not None and witness(positions[key]['history'], run['pre']) is False:
                unsound.append([key, seed])
        if vetoed(base) and not (vetoed(run) and run['pre'] == base['pre']):
            g2.append([key, seed])
        if base['outcome'] == 'RESCUED' and run['outcome'] != 'RESCUED':
            g3.append([key, seed])
        if run['switched'] and status(key, run['played']) == PROVEN_LOSS:
            loss_switch.append([key, seed])
        for r in (base, run):
            if r['outcome'] == 'TRUTH_UNRESOLVED' or status(key, r['played']) in (UNKNOWN, None):
                unresolved.append([key, seed, 's3' if r is base else 'e3a'])

    def recall(runs):
        lossy = [(k, r) for (k, _), (row, r) in runs.items()
                 if row['class'] in ('E1-P', 'E1-N') and status(k, r['pre']) == PROVEN_LOSS]
        return [sum(vetoed(r) for _, r in lossy), len(lossy)]

    rescued = lambda runs: sum(r['outcome'] == 'RESCUED' for _, r in runs.values())  # noqa: E731
    e1p = sum(p['class'] == 'E1-P' for p in positions.values())
    rec_a, rec_b = recall(a), recall(b)
    ratio = lambda x: x[0] / x[1] if x[1] else None  # noqa: E731
    gates = {
        'G1': not false_veto and not unsound,
        'G2': not g2,
        'G3': not g3,
        'G4': len(loss_switch) <= MAX_LOSS_SWITCH,
    }
    if e1p >= MIN_E1P:
        gates['G5'] = ((ratio(rec_b) or 0) >= (ratio(rec_a) or 0)) and rescued(b) >= rescued(a)
    else:
        gates['G5'] = 'N/A'
    passed = all(v is True for k, v in gates.items() if v != 'N/A')
    by_arm = {}
    for p in positions.values():
        arm = p.get('source_arm', '?')
        by_arm.setdefault(arm, {}).setdefault(p['class'], 0)
        by_arm[arm][p['class']] += 1
    return {
        'verdict': 'PASS' if passed else 'FAIL',
        'tb_engine_v1': 'E3-A' if passed else 'S3-VCT2-v1',
        'bug_suspected': bool(g2 or g3),
        'gates': gates,
        'e1p_positions': e1p, 'classes_by_source_arm': by_arm,
        'ply_distribution': sorted(p['ply'] for p in positions.values() if p['class'] == 'E1-P'),
        'veto_recall': {'s3': rec_a, 'e3a': rec_b},
        'rescued': {'s3': rescued(a), 'e3a': rescued(b)},
        'outcomes': {name: {o: sum(r['outcome'] == o for _, r in runs.values())
                            for o in sorted({r['outcome'] for _, r in runs.values()})}
                     for name, runs in (('s3', a), ('e3a', b))},
        'false_veto': false_veto, 'unsound_witness': unsound, 'witness_checked': witness is not None,
        'g2_violations': g2, 'g3_violations': g3, 'loss_switches': loss_switch, 'unresolved': unresolved,
        'cost': {'s3': s3['summary']['cost'], 'e3a': e3a['summary']['cost']},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--s3', type=Path, required=True, help='e1_evaluate.py --arm s3 output')
    parser.add_argument('--e3a', type=Path, required=True, help='e1_evaluate.py --arm e3a output')
    parser.add_argument('--skip-witness', action='store_true', help='tests only: do not re-derive proofs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))  # noqa: E731
    manifest, s3, e3a = load(args.manifest), load(args.s3), load(args.e3a)
    if manifest.get('split') != 'holdout':
        parser.error(f"{args.manifest} is not a holdout manifest (split {manifest.get('split')!r})")
    for name, ev in (('s3', s3), ('e3a', e3a)):
        if ev['manifest_sha256'] != file_sha256(args.manifest):
            parser.error(f'the {name} evaluation was run on another manifest')
    witness = None
    if not args.skip_witness:
        from analysis.e3_arms import E3_A
        budget = {'node_limit': E3_A['vct2_node_limit'], 'call_limit': E3_A['vct2_call_limit'],
                  'node_budget': E3_A['vct2_node_budget']}
        witness = lambda history, move: witness_ok(history, move, budget)  # noqa: E731
    result = {'format': 'e1-holdout-verdict-v1', 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
              'inputs': {str(p): file_sha256(p) for p in (args.manifest, args.s3, args.e3a)},
              **judge(manifest, s3, e3a, witness=witness)}
    print(json.dumps({k: v for k, v in result.items() if k not in ('unresolved',)}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
