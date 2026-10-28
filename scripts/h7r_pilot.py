"""H7-R R1 pilot runner (docs/h7-rl-plan.md 9.9) for a fixed pilot contract with an ``execution_plan``.

    preflight  no Arena sample: Git state, NumPy / torch, CUDA, the fast5 binding (the threat-window scan in use),
               the contract's files, and each model loaded on the device with one move inferred
    run        the contract's arms in its fixed order (``h7r_eval.play``; an interrupted arm resumes from its JSONL),
               with the session log and the soft wall-time budget (checked before and after every arm: on reaching
               it the active arm finishes and no further arm starts; resuming past it needs
               ``--budget-extension-hours`` with ``--approval``); an error stops the session, nothing is retried
    report     every arm record re-checked; per suite the score_split ranking (opening-cluster bootstrap), move_split,
               per-arm colour scores, the positive control; the budget from the session log

Measurement only: nothing here trains, selects a model, or changes a champion, the canonical engine or the web
model. A dirty tree is refused (``--allow-dirty`` only for tests).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
from time import monotonic
from uuid import uuid4
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7r_eval as ev  # noqa: E402
from hybrid import h7r_pilot as pilot  # noqa: E402
import h7r_eval as cli  # noqa: E402

REPORT_FORMAT = 'h7r-pilot-report-v1'
PREFLIGHT_FORMAT = 'h7r-pilot-preflight-v1'
PREFLIGHT_POSITION = [[7, 7]]  # white to move after the centre stone: not an opening of any suite, no sample


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _runs_folder(path, root: Path) -> Path:
    folder = Path(path).resolve()
    ev.require(any(folder.is_relative_to((root / d).resolve()) for d in ('runs', 'data')),
               f'{path}: pilot records stay under runs/ or data/ (never committed)')
    return folder


def _weights(contract: dict) -> dict:
    """Planning weight of each arm in S50-pair units (a larger-budget control counts the contract's multiplier)."""
    m = contract['planning']['cost']['s400_arm_cost_multiplier_assumed']
    return {a['name']: a['pairs'] * (m if a['tested']['budget'] > a['opponent']['budget'] else 1.0)
            for a in contract['arms']}


# --- preflight -----------------------------------------------------------------------------------------------------

def preflight(args, log=print, root: Path = ROOT) -> dict:
    import torch

    out = {'format': PREFLIGHT_FORMAT, 'utc': _now(), 'git': cli._git(True), 'checks': {}}
    checks = out['checks']
    checks['clean_tree'] = out['git'].get('dirty') is False and bool(out['git'].get('commit'))
    try:
        import numpy
        array = torch.arange(3).numpy()
        back = torch.from_numpy(array)
        out['numpy'] = {'present': True, 'version': numpy.__version__, 'torch_roundtrip': back.tolist() == [0, 1, 2]}
    except ImportError:
        out['numpy'] = {'present': False, 'note': 'no H7-R pilot, report or holdout path uses NumPy; recorded only'}
    except Exception as exc:  # torch built without NumPy support, or an incompatible NumPy
        out['numpy'] = {'present': True, 'torch_roundtrip': False, 'error': f'{type(exc).__name__}: {exc}'}
    checks['numpy_consistent'] = out['numpy'].get('present') is False or out['numpy'].get('torch_roundtrip') is True
    out['torch'] = {'version': torch.__version__, 'cuda': torch.cuda.is_available(),
                    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    checks['device'] = not args.device.startswith('cuda') or torch.cuda.is_available()
    contract, sha = ev.load_contract(Path(args.contract), root, check_files=True)  # files and their SHA-256
    out['contract'] = {'id': contract['contract_id'], 'status': contract['status'], 'sha256': sha}
    checks['contract_files'] = True
    import h7c_arena as ar
    import h7c_budget as sb
    from renju import Game
    from rt30 import fast3, fast5

    ar.install_runtime()
    out['runtime'] = {'fast5_installed': fast5.installed(), 'active': fast3.active()}
    checks['fast5'] = out['runtime']['fast5_installed']
    out['models'] = {}
    for name, model in contract['models'].items():
        game = Game()
        for move in PREFLIGHT_POSITION:
            game.play(*move)
        agent = sb.make_budget_agent(1, str(root / model['policy']), str(root / model['value']), 0, args.device)
        move = agent.select_move(game)
        legal = tuple(move) in set(game.legal_moves())
        out['models'][name] = {'move': list(move), 'legal': legal}
        log(f'[h7r-pilot] preflight {name}: one move inferred on {args.device} {list(move)}', flush=True)
    checks['models'] = all(m['legal'] for m in out['models'].values())
    out['pass'] = all(checks.values())
    out['note'] = 'no Arena game was played and no contract seed was used'
    return out


# --- run -----------------------------------------------------------------------------------------------------------

def run(args, log=print, root: Path = ROOT, clock=monotonic) -> dict:
    from hybrid.h7r_ledger import file_lock, publish_json

    root = Path(getattr(args, 'root', None) or root)
    git = cli._git(args.allow_dirty)
    ev.require(git.get('commit') and (args.allow_dirty or git.get('dirty') is False),
               'the pilot runs from clean committed code')
    contract, sha = cli._contract(args, root, playing=True)  # fixed, committed, files = their SHA-256
    ev.require(pilot.is_run_by_plan(contract), f"{contract['contract_id']}: not a fixed pilot contract run by plan")
    rules = pilot.check_rules(contract, args.contract)
    folder = _runs_folder(args.runs_dir, root)
    (folder / 'arms').mkdir(parents=True, exist_ok=True)
    log_path = folder / 'sessions.jsonl'
    weights = _weights(contract)
    total_weight = sum(weights.values())
    with file_lock(folder / '.lock', nonblocking=True):  # one runner per pilot folder
        events = pilot.load_log(log_path)
        pilot.check_identity(events, git['commit'], sha)
        session = uuid4().hex
        if args.budget_extension_hours is not None:
            ev.require(args.budget_extension_hours > 0 and (args.approval or '').strip(),
                       '--budget-extension-hours needs a positive value and --approval naming who approved it')
            extended = pilot.budget_state([*events, {'event': 'extension', 'hours': args.budget_extension_hours,
                                                     'approval': args.approval}], rules['soft_wall_hours'])
            ev.require(not extended['reached'], f"an extension of {args.budget_extension_hours} h does not reopen "
                                                f"the budget ({extended['cumulative_hours']} h used): nothing recorded")
            pilot.append(log_path, {'event': 'extension', 'session': session, 'utc': _now(),
                                    'hours': args.budget_extension_hours, 'approval': args.approval.strip()})
            events = pilot.load_log(log_path)
        state = pilot.budget_state(events, rules['soft_wall_hours'])
        ev.require(not state['reached'], f"the soft budget is reached ({state['cumulative_hours']} h of "
                                         f"{state['limit_hours']} h): resuming needs a separate approval "
                                         '(--budget-extension-hours H --approval TEXT)')
        started = clock()
        lock = threading.Lock()
        status = {'arm': None, 'done': 0}

        def elapsed():
            return round(clock() - started, 1)

        def write(event):
            with lock:
                pilot.append(log_path, {'session': session, 'elapsed': elapsed(), **event})

        def state_now():  # in-session reads never repair the log, and wait for a write in progress
            with lock:
                return pilot.budget_state(pilot.load_log(log_path, repair=False), rules['soft_wall_hours'],
                                          (session, elapsed()))

        def done_weight():
            return sum(weights[n] for n in rules['order'] if (folder / 'arms' / f'{n}.json').exists())

        def line():
            now = state_now()
            done = done_weight()
            rate = now['cumulative_seconds'] / done if done else None
            eta = None if rate is None else (total_weight - done) * rate / 3600
            return (f"[h7r-pilot] {done:,.0f}/{total_weight:,.0f} pair-units  {100 * done / total_weight:.1f}%  "
                    f"session {elapsed() / 3600:.2f} h  cumulative {now['cumulative_hours']:.2f} / "
                    f"{now['limit_hours']:.2f} h  ETA {'?' if eta is None else f'{eta:.1f}'} h  "
                    f"arms {status['done']}/{len(rules['order'])}  running {status['arm'] or '-'}")

        stop = threading.Event()

        def beat():
            while not stop.wait(args.heartbeat_seconds or rules['heartbeat_seconds']):
                write({'event': 'heartbeat'})
                log(line(), flush=True)

        write({'event': 'session_start', 'utc': _now(), 'commit': git['commit'], 'contract_sha256': sha,
               'contract_id': contract['contract_id'], 'workers': rules['workers'], 'device': args.device})
        thread = threading.Thread(target=beat, daemon=True) if args.heartbeat_seconds != 0 else None
        if thread:
            thread.start()
        reason = 'complete'
        try:
            for name in rules['order']:
                record_path = folder / 'arms' / f'{name}.json'
                if record_path.exists():
                    cli.load_arm(record_path, contract, sha, root)  # a finished arm is re-checked, never replayed
                    status['done'] += 1
                    continue
                if state_now()['reached']:
                    write({'event': 'budget_stop', 'before_arm': name})
                    reason = 'budget'
                    break
                write({'event': 'arm_start', 'arm': name})
                status['arm'] = name
                log(f'[h7r-pilot] arm {name} ({status["done"] + 1}/{len(rules["order"])})', flush=True)
                play = argparse.Namespace(contract=args.contract, arm=name, device=args.device,
                                          workers=rules['workers'], jsonl=folder / 'arms' / f'{name}.jsonl',
                                          allow_dirty=args.allow_dirty, pilot_session=session,
                                          threads_for_tests=getattr(args, 'threads_for_tests', False))
                try:
                    record = cli.play(play, log=log, root=root)
                    publish_json(record_path, record)
                    cli.load_arm(record_path, contract, sha, root)
                except BaseException as exc:
                    write({'event': 'arm_end', 'arm': name, 'status': 'error',
                           'detail': f'{type(exc).__name__}: {exc}'[:500]})
                    reason = 'error'
                    raise
                write({'event': 'arm_end', 'arm': name, 'status': 'complete',
                       'arm_wall_seconds': record['wall_seconds'], 'games_played_now': record['games_played_now']})
                status['done'] += 1
                status['arm'] = None
                if state_now()['reached']:
                    remaining = [n for n in rules['order'] if not (folder / 'arms' / f'{n}.json').exists()]
                    if remaining:
                        write({'event': 'budget_stop', 'after_arm': name})
                        reason = 'budget'
                        break
        finally:
            stop.set()
            if thread:
                thread.join()
            write({'event': 'session_end', 'utc': _now(), 'reason': reason})
        events = pilot.load_log(log_path)
        state = pilot.budget_state(events, rules['soft_wall_hours'])
        done = [n for n in rules['order'] if (folder / 'arms' / f'{n}.json').exists()]
        log(line(), flush=True)
        return {'format': 'h7r-pilot-run-v1', 'contract_id': contract['contract_id'], 'contract_sha256': sha,
                'commit': git['commit'], 'session': session, 'reason': reason, 'arms_complete': done,
                'arms_remaining': [n for n in rules['order'] if n not in done], 'budget': state}


# --- report --------------------------------------------------------------------------------------------------------

def report(args, root: Path = ROOT) -> dict:
    root = Path(getattr(args, 'root', None) or root)
    contract, sha = cli._contract(args, root)
    ev.require(pilot.is_run_by_plan(contract), f"{contract['contract_id']}: not a fixed pilot contract run by plan")
    rules = pilot.check_rules(contract, args.contract)
    folder = _runs_folder(args.runs_dir, root)
    events = pilot.load_log(folder / 'sessions.jsonl')
    commits = {e['commit'] for e in events if e['event'] == 'session_start'}
    ev.require(len(commits) <= 1, f'sessions of different commits: {sorted(commits)}')
    if commits:
        pilot.check_identity(events, next(iter(commits)), sha)
    arms = {}
    for name in rules['order']:
        path = folder / 'arms' / f'{name}.json'
        if path.exists():
            arm = cli.load_arm(path, contract, sha, root)
            arms[name] = {'table': arm['table'], 'summary': arm['summary'],
                          'commit': arm['record']['identity']['git'].get('commit'), 'sha256': ev.file_sha256(path)}
    arm_commits = {a['commit'] for a in arms.values()}
    ev.require(arm_commits <= commits or not events, f'arm records of another commit: {sorted(arm_commits - commits)}')
    out = pilot.report(contract, rules, arms, events)
    return {'format': REPORT_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'commit': next(iter(commits)) if commits else None, 'rank_rule': contract['suite_selection']['rank_by'],
            'arm_records_sha256': {n: a['sha256'] for n, a in arms.items()}, **out}


# --- command line --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, help='keep the environment record (e.g. in the pilot runs folder)')
    r = sub.add_parser('run')
    r.add_argument('--runs-dir', type=Path, required=True)
    r.add_argument('--device', default='cuda')
    r.add_argument('--budget-extension-hours', type=float, help='only with a separate approval (--approval)')
    r.add_argument('--approval', help='who approved the extension, and where')
    r.add_argument('--heartbeat-seconds', type=float, help='tests only (default: the contract)')
    r.add_argument('--allow-dirty', action='store_true', help='tests only')
    e = sub.add_parser('report')
    e.add_argument('--runs-dir', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    for q in sub.choices.values():
        q.add_argument('--contract', type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'output', None) is not None and Path(args.output).exists():
        parser.error(f'{args.output} exists: an H7-R record is never overwritten')
    try:
        out = {'preflight': preflight, 'run': run, 'report': report}[args.command](args)
    except (ValueError, FileNotFoundError) as exc:  # input errors; anything else keeps its traceback
        parser.error(str(exc))
    if getattr(args, 'output', None) is not None:
        from hybrid.h7r_ledger import publish_json
        publish_json(args.output, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('arms', 'suites')}, indent=1, default=str)[:4000],
          flush=True)
    if args.command == 'preflight' and not out['pass']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
