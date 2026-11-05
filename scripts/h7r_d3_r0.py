"""R2-D3-NO-OUTCOME-1 R0: the reproduction check of the c8 recipe with the current trainer, before any D3 run.

D3 compares new runs of the current trainer with the R2-0B control runs that the earlier trainer made (they are not
re-trained). R0 re-runs the c8 recipe once, unmasked and with the selection option omitted, at the reproduction
seed, and reuses the R2-0B machinery as it is:

    train   ``h7r_r2_seed.train`` (the c8 recipe, seed 20261025; the A1 amendment pins the reference record, the
            config and the inputs), into the R0 run directory
    check   ``h7r_r2_seed.reproduce`` against the reference c8 run: the A1 verdict (``h7r_r2.reproduction_verdict``)
            with the A1 tolerances unchanged; then, in addition, a DIRECT comparison with values recorded in the
            past, so that a change of the assignment code cannot match itself:
              - the re-run's data assignment fingerprint against the recorded R2-0B fingerprint
              - the fingerprint the current code reconstructs for the reference run against the same record
              - the re-run's selected step against the recorded step
            R0 passes only when the A1 verdict is BITWISE_REPRODUCED or METRIC_ONLY AND every direct comparison
            holds; anything else is NOT_REPRODUCED (exit 1), and D3 does not start.

``train`` and ``check`` run at one clean commit: the run record names no R0 contract, so ``check`` refuses a run whose
recorded commit is not its own (that commit holds the approved contract). Both commands need the fixed, committed
and approved R0 contract (``h7r-d3-r0-contract-v1``): the R2 contract and
A1 amendment it builds on, the trainer sources it was fixed for, the recorded values with the records that hold them
(checked against those records: the pinned constants cannot be typos), the A1 tolerances (checked equal), the seed
and the output paths. Nothing is written over; the R0 result is a new file.

    python -u scripts/h7r_d3_r0.py train --r0-contract docs/h7r/r2/R2-D3-NO-OUTCOME-1/r0-contract.json \\
        --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \\
        --replay-cache data/tactical/h6/value_cache_v3 --parent runs/h7c/c5/value/best.pt --device cuda
    python -u scripts/h7r_d3_r0.py check --r0-contract docs/h7r/r2/R2-D3-NO-OUTCOME-1/r0-contract.json \\
        --cache runs/h7r/R1-HDEV-1/cache --reference-model runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d3-r0-contract-v1'
RESULT_FORMAT = 'h7r-d3-r0-v1'
VERDICT_VERSION = 'h7r-d3-r0-verdict-v1'
CONTRACT_ID = 'R2-D3-NO-OUTCOME-1-R0'
TRAINER_SOURCES = ('src/hybrid/h7c_train.py', 'scripts/h7c_train_value.py', 'scripts/h7r_r2_seed.py',
                   'scripts/h7r_d3_r0.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'r2_contract', 'amendment', 'trainer_sources', 'recorded',
        'seed', 'tolerance', 'output'}


def _ref(value) -> bool:
    return isinstance(value, dict) and set(value) == {'path', 'sha256'} and isinstance(value['path'], str) \
        and r2.is_sha(value['sha256'])


def _pinned(root: Path, ref: dict, what: str) -> dict:
    path = root / ref['path']
    r2.require(path.is_file() and r2.file_sha256(path) == ref['sha256'], f"{ref['path']}: not the {what} R0 pins")
    return json.loads(path.read_text(encoding='utf-8'))


def load_r0(path, root: Path = ROOT, *, check_sources: bool = True) -> tuple[dict, str, dict]:
    """The R0 contract, fixed, committed and approved, and every value it pins checked against its source."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no R0 contract')
    c = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the R0 contract keys differ from {sorted(KEYS)}')
    r2.require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed'
               and c['approved'] is True, f'{path}: not the fixed, approved R0 contract')
    r2.require(committed(path, root), f'{path}: not committed')
    for key in ('r2_contract', 'amendment'):
        r2.require(_ref(c[key]), f'R0 {key}: a path and a full SHA-256 are required')
    amendment = _pinned(root, c['amendment'], 'amendment')
    _pinned(root, c['r2_contract'], 'R2 contract')
    b = amendment['r2_0b']
    r2.require(b.get('approved') is True, 'the A1 amendment does not approve the R2-0B machinery')
    r2.require(type(c['seed']) is int and c['seed'] == b['reproduction_seed'], 'R0 seed is not the A1 reproduction seed')
    r2.require(c['tolerance'] == b['reproduction']['tolerance'], 'R0 tolerances are not the A1 tolerances')
    rec = c['recorded']
    r2.require(isinstance(rec, dict) and set(rec) == {'reproduction', 'run', 'reference_record', 'assignment_sha256',
                                                      'best_step'}
               and all(_ref(rec[k]) for k in ('reproduction', 'run', 'reference_record'))
               and r2.is_sha(rec['assignment_sha256']) and type(rec['best_step']) is int,
               'R0 recorded values are malformed')
    reproduction = _pinned(root, rec['reproduction'], 'R2-0B reproduction record')
    run = _pinned(root, rec['run'], 'R2-0B reproduction run record')
    r2.require(rec['reference_record'] == b['reference_record'], 'R0 names another reference record than A1')
    reference = _pinned(root, rec['reference_record'], 'reference c8 record')
    fp, step = rec['assignment_sha256'], rec['best_step']
    r2.require(list(reproduction['checks']['assignment']) == [fp, fp] and run['assignment']['sha256'] == fp,
               'the recorded fingerprint is not the one the R2-0B records hold')
    r2.require(list(reproduction['checks']['best_step']) == [step, step] and run['best_step'] == step
               and reference['best']['step'] == step, 'the recorded step is not the one the R2-0B / c8 records hold')
    r2.require(reproduction['verdict']['verdict'] in r2.REPRODUCED and run['seed'] == c['seed']
               and run['role'] == 'reproduction', 'the recorded R2-0B reproduction did not pass')
    sources = c['trainer_sources']
    r2.require(isinstance(sources, dict) and set(sources) == set(TRAINER_SOURCES)
               and all(r2.is_sha(v) for v in sources.values()), 'R0 trainer sources are malformed')
    if check_sources:
        changed = [s for s in TRAINER_SOURCES if r2.file_sha256(root / s) != sources[s]]
        r2.require(not changed, f'the checkout differs from the trainer sources R0 was fixed for: {changed}')
    out = c['output']
    r2.require(isinstance(out, dict) and set(out) == {'run_dir', 'reproduction', 'result'}
               and all(isinstance(v, str) for v in out.values())
               and all(v.startswith('runs/') for v in (out['run_dir'], out['reproduction'])),
               'R0 output paths: run_dir and reproduction under runs/, and the result')
    return c, r2.file_sha256(path), amendment


def r0_verdict(a1: dict, recorded: dict, rerun_assignment, reference_assignment, rerun_step) -> dict:
    """PASS only with a passed A1 verdict AND the recorded fingerprint and step matched directly."""
    reasons = []
    if not isinstance(a1, dict) or a1.get('verdict') not in r2.REPRODUCED:
        reasons.append(f"A1 verdict {(a1 or {}).get('verdict')}: {(a1 or {}).get('reasons')}")
    fp = recorded['assignment_sha256']
    if rerun_assignment != fp:
        reasons.append(f'the re-run assignment {str(rerun_assignment)[:12]} is not the recorded R2-0B {fp[:12]}')
    if reference_assignment != fp:
        reasons.append(f'the current code reconstructs the reference assignment as {str(reference_assignment)[:12]}, '
                       f'not the recorded {fp[:12]}')
    if type(rerun_step) is not int or rerun_step != recorded['best_step']:
        reasons.append(f"the re-run selected step {rerun_step!r} is not the recorded {recorded['best_step']}")
    return {'version': VERDICT_VERSION, 'verdict': 'NOT_REPRODUCED' if reasons else 'PASS',
            'a1_verdict': a1, 'reasons': reasons}


def git_state(root: Path) -> dict:
    import subprocess

    def git(*a):
        return subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def seed_args(c: dict, args, **extra) -> argparse.Namespace:
    root = Path(args.root)
    return argparse.Namespace(contract=root / c['r2_contract']['path'], amendment=root / c['amendment']['path'],
                              device=args.device, allow_dirty=False, root=args.root, **extra)


def train(args, log=print) -> dict:
    import h7r_r2_seed as seed

    root = Path(args.root)
    c, csha, _ = load_r0(args.r0_contract, root)
    run_dir = root / c['output']['run_dir']
    r2.require(not run_dir.exists(), f'{run_dir} exists: never overwritten')
    log(f"[d3-r0] R0 contract {csha[:12]}: the c8 recipe at seed {c['seed']} with the current trainer "
        f"({c['trainer_sources']['src/hybrid/h7c_train.py'][:12]}), selection option omitted", flush=True)
    return seed.train(seed_args(c, args, seed=c['seed'], cycle_cache=Path(args.cycle_cache),
                                cycle_rows=Path(args.cycle_rows), replay_cache=Path(args.replay_cache),
                                parent=Path(args.parent), out_dir=run_dir, reproduction=None, reason=None), log=log)


def check(args, log=print) -> dict:
    import h7r_r2_seed as seed
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    c, csha, _ = load_r0(args.r0_contract, root)
    out = {k: root / v for k, v in c['output'].items()}
    r2.require(not out['reproduction'].exists() and not out['result'].exists(), 'an R0 output exists: never overwritten')
    run = json.loads((out['run_dir'] / 'r2_train.json').read_text(encoding='utf-8'))
    r2.require(run.get('sources', {}).get('src/hybrid/h7c_train.py') == c['trainer_sources']['src/hybrid/h7c_train.py'],
               'the R0 run was not trained with the trainer R0 was fixed for')
    # The run record names no R0 contract: the run is bound to it through the commit. ``train`` and ``check`` must
    # run at the same clean commit, the one holding this approved R0 contract (load_r0 checked it is committed).
    train_git, check_git = run.get('git'), git_state(root)
    r2.require(isinstance(train_git, dict) and train_git.get('dirty') is False and check_git['dirty'] is False
               and train_git.get('commit') == check_git['commit'],
               f"the R0 run was trained at {str((train_git or {}).get('commit'))[:12]}, not at the clean commit "
               f"{check_git['commit'][:12]} that checks it: train and check run at one commit")
    log('[d3-r0] A1 reproduction check of the R0 run against the reference c8 run', flush=True)
    rep = seed.reproduce(seed_args(c, args, cache=Path(args.cache), reference_model=args.reference_model,
                                   run_dir=out['run_dir'], output=out['reproduction']), log=log)
    r2.require(rep.get('git') == check_git, 'the commit changed during the check')
    r2.require(rep['tolerance'] == c['tolerance'], 'the reproduction used other tolerances')
    verdict = r0_verdict(rep['verdict'], c['recorded'], rep['assignment']['rerun_sha256'],
                         rep['assignment']['reference']['sha256'], rep['best_step']['rerun'])
    recorded_env = _pinned(root, c['recorded']['run'], 'R2-0B reproduction run record')['environment']
    body = {'format': RESULT_FORMAT, 'r0_contract_sha256': csha, 'git': rep['git'], 'train_git': train_git,
            'trainer_sources': c['trainer_sources'], 'run_dir': c['output']['run_dir'],
            'run_sources': run.get('sources'),
            'reproduction': {'path': c['output']['reproduction'], 'sha256': r2.file_sha256(out['reproduction'])},
            'direct': {'recorded_assignment_sha256': c['recorded']['assignment_sha256'],
                       'rerun_assignment_sha256': rep['assignment']['rerun_sha256'],
                       'reconstructed_reference_assignment_sha256': rep['assignment']['reference']['sha256'],
                       'recorded_best_step': c['recorded']['best_step'], 'rerun_best_step': rep['best_step']['rerun']},
            'a1_checks': rep['checks'], 'tolerance': rep['tolerance'],
            'environment': {'now': rep['environment'], 'recorded_r2_0b': recorded_env,
                            'same': rep['environment'] == recorded_env},
            'verdict': verdict,
            'scope': 'the precondition of D3: one re-run that finds reproduction problems; no D3 run, no ruling'}
    publish_json(out['result'], body)
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--r0-contract', type=Path, required=True)
    t.add_argument('--cycle-cache', required=True)
    t.add_argument('--cycle-rows', required=True)
    t.add_argument('--replay-cache', required=True)
    t.add_argument('--parent', required=True, help='c5 value checkpoint (its best.json beside it)')
    t.add_argument('--device', default='cuda')
    k = sub.add_parser('check')
    k.add_argument('--r0-contract', type=Path, required=True)
    k.add_argument('--cache', required=True, help='the H-dev cache')
    k.add_argument('--reference-model', required=True, help='c8 value checkpoint,metadata')
    k.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    args.root = ROOT
    try:
        if args.command == 'train':
            out = train(args)
            print(f"[d3-r0] R0 run {out['verdict']}, best step {out['best_step']}", flush=True)
            return 0 if out['verdict'] == 'CANDIDATE' else 1
        out = check(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    v = out['verdict']
    print(f"[d3-r0] {v['verdict']} (A1 {v['a1_verdict']['verdict']}); direct {out['direct']}", flush=True)
    for reason in v['reasons']:
        print(f'  reason: {reason}', flush=True)
    return 0 if v['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
