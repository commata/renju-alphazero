"""R2-D4 Rapfi path probe: the registered Rapfi through the D4 branch path on two fixed public-fixture branches.

A protocol and worker-path check of the registered engine (``h7c_adjudicate.engine_spec`` and ``_chunk`` in a spawn
``ProcessPoolExecutor``, a fresh engine process per branch, played to the end), never a strength evaluation and no
guarantee for the D4 full run. It uses no H-dev, H-conf or A-conf position: the two branches (one with black, one with
white as the fixed mover) are fixed in the committed plan ``docs/h7r/r2/R2-D4-BASELINE-1/rapfi-probe-plan.json``
with their moves, keys, input hashes and scope. Results never enter a D4 journal or statistic.

    run     plan committed and approved, clean commit, helper sources = the plan's, every branch legal and not over
            after the fixed move, Rapfi files = the contract's registered identity before and after; output folder
            new and outside the D4 smoke / full folders; branches.jsonl (append + fsync), identity.json, summary.json
    check   read-only: the plan, identity and summary links, the strict journal, every branch replayed through the
            D4 validator (``branch_outcome``); PASS only when both branches ended normally (or as a verified
            structural draw) with a legal engine continuation and the black forbidden points were cross-checked at
            least once; any execution failure, undefined INVALID or integrity error is FAIL. Writes one new file.

A failed probe keeps its records; no other position is tried in its place.

    python -u scripts/h7r_d4_rapfi_probe.py run --rapfi-dir /mnt/c/engines/rapfi
    python -u scripts/h7r_d4_rapfi_probe.py check --output docs/h7r/r2/R2-D4-BASELINE-1/records/rapfi-probe/check.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d4 as d4  # noqa: E402

PLAN = 'docs/h7r/r2/R2-D4-BASELINE-1/rapfi-probe-plan.json'
PLAN_FORMAT = 'h7r-d4-rapfi-probe-v1'
RUN_FORMAT = 'h7r-d4-rapfi-probe-run-v1'
SUMMARY_FORMAT = 'h7r-d4-rapfi-probe-summary-v1'
CHECK_FORMAT = 'h7r-d4-rapfi-probe-check-v1'
EXECUTION_REASONS = ('timeout', 'engine_error', 'engine_illegal', 'forbid_mismatch')
FORBIDDEN_FOLDERS = ('runs/h7r/R2-D4-BASELINE-1/smoke', 'runs/h7r/R2-D4-BASELINE-1/full')


def need(ok, message: str) -> None:
    if not ok:
        raise ValueError(message)


def checked_plan(path, root: Path) -> tuple[dict, str, dict]:
    """The committed, approved plan; its contract, helper sources and branches as they are now."""
    import h7c_adjudicate as ad
    from hybrid.h7c_nmask import committed
    from renju import BLACK, Game

    path = Path(path)
    plan = json.loads(path.read_text(encoding='utf-8'))
    need(plan.get('format') == PLAN_FORMAT and plan.get('approved') is True, f'{path}: not an approved probe plan')
    need(committed(path, root), f'{path}: the plan is not committed')
    contract_path = root / plan['contract']['path']
    need(d4.file_sha256(contract_path) == plan['contract']['sha256'], 'the D4 contract differs from the plan')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    need(plan['rapfi_identity'] == contract['rapfi_identity'] and plan['rapfi_program'] == contract['rapfi_program'],
         'the plan names another Rapfi than the D4 contract')
    need(plan['invalid_policy'] == contract['invalid_policy'], 'the plan names another INVALID policy')
    for ref in plan['inputs'] + plan['helper_sources']:
        need(d4.file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: differs from the plan")
    out = (root / plan['out_dir']).resolve()
    need(all(out != (root / f).resolve() and (root / f).resolve() not in out.parents for f in FORBIDDEN_FOLDERS),
         'the probe folder must be outside the D4 smoke / full folders')
    names = [b['name'] for b in plan['branches']]
    need(len(plan['branches']) == 2 and len(set(names)) == 2 and
         sorted(b['mover'] for b in plan['branches']) == ['black', 'white'],
         'exactly two branches, one black and one white fixed mover')
    for b in plan['branches']:
        game = Game()
        for move in b['prefix']:
            game.play(*move)  # IllegalMove for an illegal prefix
        need(not game.done and ('black' if game.to_play == BLACK else 'white') == b['mover'],
             f"branch {b['name']}: the prefix is over or the mover differs")
        need(tuple(b['move']) in set(game.legal_moves()), f"branch {b['name']}: the fixed move is illegal")
        game.play(*b['move'])
        need(not game.done, f"branch {b['name']}: the fixed move ends the game (no continuation to probe)")
        need(b['key'] == ad.branch_key(b['prefix'], b['move']), f"branch {b['name']}: key differs")
    return plan, d4.file_sha256(path), contract


def tasks_of(plan: dict, identity_sha: str, spec: dict) -> list[dict]:
    return [{'key': b['key'], 'moves': b['prefix'], 'move': b['move'], 'order': i, 'identity_sha256': identity_sha,
             'spec': spec} for i, b in enumerate(plan['branches'])]


def run(args, log=print) -> dict:
    import h7c_adjudicate as ad
    import h7r_d4 as cli
    from hybrid.h7r_ledger import publish_json
    from hybrid.progress import Progress, run_chunks

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    state = cli.git_state(root)
    plan, plan_sha, contract = checked_plan(args.plan, root)
    spec = ad.engine_spec(root / plan['rapfi_program'], args.rapfi_dir)
    need(spec['identity'] == plan['rapfi_identity'], 'Rapfi files differ from the registered identity')
    folder = Path(args.out_dir).resolve()
    need(folder == (root / plan['out_dir']).resolve(), f"use the plan's output folder {plan['out_dir']}")
    need(not folder.exists(), f'{folder} exists: a probe is never run again into it')
    folder.mkdir(parents=True)
    identity = {'format': RUN_FORMAT, 'plan_sha256': plan_sha, 'git': state, 'rapfi': spec['identity'],
                'helper_sources': plan['helper_sources'], 'workers': plan['workers'], 'mp_context': 'spawn',
                'environment': {'python': platform.python_version(), 'platform': platform.platform()}}
    identity_sha = d4.digest(identity)
    publish_json(folder / 'identity.json', identity)
    journal = folder / 'branches.jsonl'
    tasks = tasks_of(plan, identity_sha, spec)
    done = {}

    def keep(_, records):
        for record in records:
            need(record.get('key') in {t['key'] for t in tasks} and record['key'] not in done, 'unexpected record')
            cli.append_record(journal, record)
            done[record['key']] = record
            log(f"[d4-rapfi-probe] {record['key'][:12]} {record.get('status')} {record.get('reason')} "
                f"continuation {len(record.get('continuation') or [])} forbid_checked {record.get('forbid_checked')}",
                flush=True)

    log(f"[d4-rapfi-probe] {len(tasks)} branches, workers {plan['workers']} (spawn)", flush=True)
    with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=get_context('spawn')) as pool:
        run_chunks(pool, ad._chunk, [[t] for t in tasks], Progress('d4-rapfi-probe', len(tasks), 'branches'),
                   len, on_result=keep)
    need(set(done) == {t['key'] for t in tasks}, 'incomplete probe')
    after = ad.engine_spec(root / plan['rapfi_program'], args.rapfi_dir)['identity']
    need(after == spec['identity'], 'Rapfi files changed during the probe')
    need(cli.git_state(root) == state, 'HEAD changed during the probe')
    summary = {'format': SUMMARY_FORMAT, 'identity': identity, 'identity_sha256': identity_sha,
               'files': {'branches.jsonl': d4.file_sha256(journal)}, 'branches': len(done),
               'elapsed_seconds': round(time.monotonic() - started, 3),
               'scope': plan['scope']}
    publish_json(folder / 'summary.json', summary)
    return summary


def check(args) -> dict:
    """Read-only verification; one new output file."""
    import h7r_d4 as cli
    import h7r_d4_smoke_check as smoke
    from hybrid.h7r_ledger import publish_json

    root = Path(getattr(args, 'root', ROOT))
    output = Path(args.output)
    need(not output.exists(), f'{output} exists')
    report = {'format': CHECK_FORMAT, 'integrity': None, 'failures': [], 'verdict': None, 'branches': {}}
    try:
        plan, plan_sha, contract = checked_plan(args.plan, root)
        folder = Path(args.out_dir)
        need(folder.resolve() == (root / plan['out_dir']).resolve(), "not the plan's output folder")
        identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        identity_sha = d4.digest(identity)
        need(summary.get('format') == SUMMARY_FORMAT and summary.get('identity') == identity
             and summary.get('identity_sha256') == identity_sha, 'the summary is not bound to identity.json')
        need(identity.get('format') == RUN_FORMAT and identity.get('plan_sha256') == plan_sha
             and identity.get('helper_sources') == plan['helper_sources'] and identity.get('rapfi') == plan['rapfi_identity']
             and identity.get('mp_context') == 'spawn' and identity.get('workers') == plan['workers'],
             'the run identity names another plan, helper source, Rapfi or worker setting')
        head = cli.git_state(root)
        need(identity.get('git') == head, 'the run commit is not the checked commit')
        journal = folder / 'branches.jsonl'
        need(summary.get('files') == {'branches.jsonl': d4.file_sha256(journal)}, 'journal SHA differs from summary')
        tasks = {t['key']: t for t in tasks_of(plan, identity_sha, None)}
        records = smoke.strict_journal(journal, 'key', identity_sha, set(tasks))
        need(summary.get('branches') == len(records), 'summary branch count differs')
        need(isinstance(summary.get('elapsed_seconds'), (int, float)) and summary['elapsed_seconds'] >= 0,
             'elapsed seconds not finite')
        names = {b['key']: b['name'] for b in plan['branches']}
        worker, forbid = 0.0, 0
        for key, record in records.items():
            need(isinstance(record.get('seconds'), (int, float)) and record['seconds'] >= 0, 'branch seconds')
            need(type(record.get('forbid_checked')) is int and record['forbid_checked'] >= 0, 'forbid count')
            outcome = cli.validate_branch(record, tasks[key], plan['invalid_policy'])  # Game replay, policy
            moves = record.get('continuation') or []
            row = {'status': record.get('status'), 'reason': record.get('reason'), 'kind': outcome['kind'],
                   'continuation': len(moves), 'forbid_checked': record['forbid_checked'], 'length': record.get('length'),
                   'seconds': record['seconds']}
            report['branches'][names[key]] = row
            worker += record['seconds']
            forbid += record['forbid_checked']
            if outcome['kind'] == 'structural':
                need(record.get('status') == 'INVALID' and record.get('reason') == 'abnormal_end',
                     f'{names[key]}: structural draw without its original INVALID mark')
            elif outcome['kind'] != 'resolved':
                need(outcome['reason'] in EXECUTION_REASONS, f"{names[key]}: undefined INVALID {outcome['reason']!r}")
        report.update(integrity='PASS', plan_sha256=plan_sha, identity_sha256=identity_sha,
                      rapfi_identity=plan['rapfi_identity'], forbid_checked_total=forbid,
                      timing={'elapsed_seconds': summary['elapsed_seconds'], 'branch_worker_seconds': round(worker, 3),
                              'note': 'elapsed is the parallel wall time; worker seconds add up both branches'})
    except (ValueError, KeyError, OSError, json.JSONDecodeError, smoke.Failure) as exc:
        report['integrity'] = 'FAIL'
        report['failures'].append(f'{type(exc).__name__}: {exc}')
    if report['integrity'] == 'PASS':
        for name, row in report['branches'].items():
            if row['kind'] not in ('resolved', 'structural'):
                report['failures'].append(f"{name}: execution failure {row['reason']}")
            elif row['continuation'] == 0:
                report['failures'].append(f'{name}: no engine continuation')
        if report['forbid_checked_total'] == 0:
            report['failures'].append('the black forbidden points were never cross-checked')
    report['verdict'] = 'PASS' if report['integrity'] == 'PASS' and not report['failures'] else 'FAIL'
    report['scope'] = ('registered Rapfi protocol and worker-path check on two public-fixture branches; no strength '
                       'evaluation, no guarantee for the D4 full run, no change to the smoke lock / resume NOT_TESTED')
    publish_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'check'):
        s = sub.add_parser(name)
        s.add_argument('--plan', default=PLAN)
        s.add_argument('--out-dir', default='runs/h7r/R2-D4-BASELINE-1/rapfi-probe')
        if name == 'run':
            s.add_argument('--rapfi-dir', required=True)
        else:
            s.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            out = run(args)
            print(f"[d4-rapfi-probe] done: {out['branches']} branches in {out['elapsed_seconds']} s "
                  '(run check next)', flush=True)
            return 0
        report = check(args)
    except ValueError as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    print(f"[d4-rapfi-probe] {report['verdict']} (integrity {report['integrity']}); branches {report['branches']}; "
          f"forbid checked {report.get('forbid_checked_total')}", flush=True)
    for failure in report['failures']:
        print(f'  failure: {failure}', flush=True)
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
