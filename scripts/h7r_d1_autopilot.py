"""R2-D1-AUTOPILOT-1 driver: the delegation's progression rules as code, for an agent on the desktop.

The agent never decides what comes next; it asks this tool, does exactly that, and reports when the tool says stop.

    next       pull, then the next action from the committed records and the launch ledger (JSON on stdout):
                 wait     a launched block has not finished (``wait``)
                 run      launch a block (``launch``)
                 recover  launch the record-preserving block of a failed step (``launch``), then stop
                 approve  commit a sub-approval after its gate (``approve MODE``)
                 stop     end the delegation run here and report (``final`` true: the delegation itself ends)
    launch     the action of ``next`` when it is run / recover: blocks extracted from the D1 draft into ``--home``,
               one tmux session ``d1auto``, output and exit code in the ledger folder
    wait       poll the running block for up to ``--minutes``; exit 0 when it finished, 3 while it still runs
    approve    re-check ``next`` == approve MODE and its gate, flip ``approved.MODE`` in the D4 contract, append the
               state file, commit and push
    report     the committed records so far, compactly (JSON)

Steps, in order (D1 draft sections 14.8 and 16.3):

    hdev      d1-full-3.sh          done: full/d1_eval.json committed; gate: label DEVELOPMENT_SIGNAL
    timing    d1-v2-1.sh            done: factory-v2/timing.json committed; gate: passed
    eligible  d1-d4-1.sh            done: d4/eligible.json committed
    smoke     d1-d4-run.sh smoke    needs approved.smoke (gate: D4 entry checks); done: d4/smoke/summary.json
                                    committed; gate: SMOKE, actor reproduced, same experiment
    full      d1-d4-run.sh full     needs approved.full (gate: the smoke gate and at least MIN_HOURS_FOR_FULL left);
                                    done: d4/full/summary.json committed; then stop with its reading

Failures: a step that failed stops the run. The one retry is the delegation's: a D4 run (smoke or full) whose tool
log shows no ``STOP:`` refusal (a crash or kill, not a check) is resumed once with the same command and HEAD. A step
whose block did not reach its records commit gets its record-preserving block once (hdev: d1-full-3F.sh; D4:
d1-d4-F.sh), then the run stops. Past the delegation's 24 hours nothing is launched or approved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid.h7c_nmask import committed  # noqa: E402

D = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1'
DRAFT = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1-draft.md'
CONTRACT = f'{D}/d4/d1-d4-contract.json'
STATE = f'{D}/autopilot/state.md'
LEDGER_DIR = 'runs/h7r/R2-D1-HEAD-WIDTH-1/autopilot'
D4_LOGS = 'runs/h7r/R2-D1-D4-1/logs'
TMUX = 'd1auto'
KST = timezone(timedelta(hours=9))
START = datetime(2026, 11, 9, 15, 24, 27, tzinfo=KST)  # state file: the first P1 block run
DEADLINE = START + timedelta(hours=24)
MIN_HOURS_FOR_FULL = 6.0  # D4 full estimate 3.5-5 h (D1 draft 16.2) plus margin

STEPS = ('hdev', 'timing', 'eligible', 'smoke', 'full')
BLOCKS = {'hdev': ('d1-full-3.sh', []), 'timing': ('d1-v2-1.sh', []), 'eligible': ('d1-d4-1.sh', []),
          'smoke': ('d1-d4-run.sh', ['smoke']), 'full': ('d1-d4-run.sh', ['full'])}
RECOVER = {'hdev': ('d1-full-3F.sh', []), 'smoke': ('d1-d4-F.sh', ['smoke']), 'full': ('d1-d4-F.sh', ['full'])}
DONE = {'hdev': f'{D}/full/d1_eval.json', 'timing': f'{D}/factory-v2/timing.json', 'eligible': f'{D}/d4/eligible.json',
        'smoke': f'{D}/d4/smoke/summary.json', 'full': f'{D}/d4/full/summary.json'}


def _git(root, *args) -> str:
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def committed_json(root, rel):
    path = Path(root) / rel
    if path.is_file() and committed(path, root):
        return json.loads(path.read_text(encoding='utf-8'))
    return None


def extract_blocks(root=ROOT) -> dict:
    """The desktop blocks, exactly as the D1 draft holds them."""
    doc = (Path(root) / DRAFT).read_text(encoding='utf-8')

    def blocks(start, end):
        return re.findall(r'```bash\n(.*?)```', doc.split(start, 1)[1].split(end, 1)[0], re.S)
    b14 = blocks('\n### 14.8 실행 절차', '\n### 14.9')
    b16 = blocks('\n### 16.3 실행 절차', '\n### 16.4')
    if len(b14) != 7 or len(b16) != 4:
        raise ValueError(f'the D1 draft blocks changed: {len(b14)} / {len(b16)}')
    return {'d1-full-3.sh': b14[4], 'd1-full-3R.sh': b14[5], 'd1-full-3F.sh': b14[6], 'd1-v2-1.sh': b16[0],
            'd1-d4-1.sh': b16[1], 'd1-d4-run.sh': b16[2], 'd1-d4-F.sh': b16[3]}


# --- ledger ----------------------------------------------------------------------------------------------------------

def ledger(root) -> list[dict]:
    path = Path(root) / LEDGER_DIR / 'launches.jsonl'
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            entry = json.loads(line)
            out.append({**entry, **outcome(root, entry)})
    return out


def outcome(root, entry) -> dict:
    """Exit code (None while running) and record markers of one launch, from its output file."""
    path = Path(root) / entry['out']
    text = path.read_text(encoding='utf-8', errors='replace') if path.exists() else ''
    m = re.search(r'^BLOCK EXIT (-?\d+)\s*$', text, re.M)
    return {'exit': int(m.group(1)) if m else None, 'recorded': '__RECORDED__' in text,
            'not_recorded': '__NOT_RECORDED__' in text}


def tmux_alive() -> bool:
    return subprocess.run(['tmux', 'has-session', '-t', TMUX], capture_output=True).returncode == 0


def d4_tool_logs(root, mode: str, since: float) -> list[Path]:
    return sorted(p for p in (Path(root) / D4_LOGS).glob(f'{mode}-*.log') if p.stat().st_mtime >= since - 1)


def retryable(root, step: str, launch: dict) -> bool:
    """A D4 run that died without a tool refusal: its tool log exists and has no ``STOP:`` line."""
    if step not in ('smoke', 'full') or not launch['not_recorded']:
        return False
    logs = d4_tool_logs(root, step, launch['started'])
    return bool(logs) and not any('STOP:' in p.read_text(encoding='utf-8', errors='replace') for p in logs)


# --- decision --------------------------------------------------------------------------------------------------------

def act(action: str, **kw) -> dict:
    return {'action': action, **kw}


def gate(root, step: str, rec: dict, c: dict):
    """None when the completed step lets the run continue, else the stop action."""
    if step == 'hdev':
        label = rec['hdev_reading']['label']
        if label != 'DEVELOPMENT_SIGNAL':
            return act('stop', final=True, reason=f'full-scale H-dev reading {label}: no D4; the delegation ends '
                                                  '(P4); report the records')
    if step == 'timing' and rec['result']['passed'] is not True:
        r = rec['result']
        return act('stop', final=True, reason=f"factory v2 time gate FAIL (mean x{r['mean_ratio']:.3f}, p95 "
                                              f"x{r['p95_ratio']:.3f}): no D4; the delegation ends")
    if step == 'smoke':
        import h7r_d1_d4 as dd
        if not (rec['identity']['experiment_sha256'] == dd.experiment_sha256(c) and rec['reading'] == 'SMOKE'
                and rec['actor_check']['reproduced'] is True):
            return act('stop', final=True, reason='the D4 smoke is not a reproduced smoke of this experiment')
    if step == 'full':
        reading = rec['reading']
        nxt = ('P5 confirmation: the cloud session fixes the H-conf / A-conf contracts before any confirmation data'
               if reading == 'ALL_THREE_BETTER' else 'the delegation ends (P3): report the records')
        return act('stop', final=True, reading=reading, reason=f'D4 reading {reading}; next: {nxt}')
    return None


def approval_gate(root, mode: str, c: dict, now: datetime):
    """None when ``approved.MODE`` may be committed, else the reason."""
    import h7r_d1_d4 as dd
    if mode == 'smoke':
        try:
            dd.entry_checks(c, root)
        except (ValueError, KeyError) as exc:
            return f'D4 entry checks: {exc}'
        if committed_json(root, DONE['eligible']) is None:
            return 'the eligible manifest is not committed'
        return None
    left = (DEADLINE - now).total_seconds() / 3600
    if left < MIN_HOURS_FOR_FULL:
        return f'{left:.1f} h left of the delegation: below {MIN_HOURS_FOR_FULL} h for the D4 full run'
    return None


def decide(root=ROOT, now: datetime | None = None, alive=None) -> dict:
    now = now or datetime.now(KST)
    alive = tmux_alive if alive is None else alive
    launches = ledger(root)
    if launches and launches[-1]['exit'] is None:
        if alive():
            return act('wait', step=launches[-1]['step'], out=launches[-1]['out'])
        return act('stop', final=False, reason=f"{launches[-1]['out']}: the block ended without BLOCK EXIT "
                                               '(tmux session gone); inspect and report')
    c = json.loads((Path(root) / CONTRACT).read_text(encoding='utf-8'))
    for step in STEPS:
        rec = committed_json(root, DONE[step])
        if rec is not None:
            stop = gate(root, step, rec, c)
            if stop:
                return stop
            continue
        if now >= DEADLINE:
            return act('stop', final=True, reason=f'the delegation limit ({DEADLINE.isoformat()}) has passed '
                                                  f'before {step}')
        mine = [e for e in launches if e['step'] == step and e['kind'] == 'run']
        recovered = [e for e in launches if e['step'] == step and e['kind'] == 'recover']
        if not mine:
            if step in ('smoke', 'full') and c['approved'][step] is not True:
                why = approval_gate(root, step, c, now)
                if why:
                    return act('stop', final=True, reason=f'approved.{step} cannot be given: {why}')
                return act('approve', step=step, mode=step)
            block, args = BLOCKS[step]
            return act('run', step=step, block=block, args=args)
        last = mine[-1]
        if recovered:
            return act('stop', final=False, reason=f'{step} failed (exit {last["exit"]}, {last["out"]}); its '
                                                   f'records were preserved ({recovered[-1]["out"]}); report')
        if last['exit'] == 0 and not last['recorded']:
            return act('stop', final=False, reason=f'{step}: exit 0 without a records commit ({last["out"]})')
        if len(mine) == 1 and retryable(root, step, last):
            block, args = BLOCKS[step]
            return act('run', step=step, block=block, args=args, retry=True,
                       reason='the D4 run died without a tool refusal: one resume (delegation retry rule)')
        if step in RECOVER and not last['recorded']:
            block, args = RECOVER[step]
            return act('recover', step=step, block=block, args=args,
                       reason=f'{step} failed (exit {last["exit"]}) before its records commit')
        return act('stop', final=False, reason=f'{step} failed (exit {last["exit"]}); records '
                                               f"{'committed' if last['recorded'] else 'not committed'} "
                                               f'({last["out"]}); report')
    return act('stop', final=True, reason='every step is complete')


# --- commands --------------------------------------------------------------------------------------------------------

def require_clean(root):
    if _git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('the tracked tree is not clean')
    if _git(root, 'branch', '--show-current') != 'feat/h7-c1':
        raise ValueError('not on feat/h7-c1')


def cmd_next(args) -> dict:
    root = Path(args.root)
    require_clean(root)
    if not args.no_pull:
        _git(root, 'pull', '--ff-only', '-q', 'origin', 'feat/h7-c1')
    return decide(root)


def cmd_launch(args) -> dict:
    root = Path(args.root)
    action = cmd_next(args)
    if action['action'] not in ('run', 'recover'):
        raise ValueError(f'nothing to launch: {action}')
    home = Path(args.home).expanduser()
    for name, text in extract_blocks(root).items():
        (home / name).write_text(text, encoding='utf-8')
    folder = root / LEDGER_DIR
    folder.mkdir(parents=True, exist_ok=True)
    n = len(ledger(root)) + 1
    out = f"{LEDGER_DIR}/{n:02d}-{action['step']}-{action['action']}.out"
    if (root / out).exists():
        raise ValueError(f'{out} exists')
    script = ' '.join([str(home / action['block']), *action['args']])
    entry = {'n': n, 'step': action['step'], 'kind': action['action'], 'block': action['block'],
             'args': action['args'], 'out': out, 'started': time.time(),
             'started_kst': datetime.now(KST).isoformat(timespec='seconds')}
    with open(folder / 'launches.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    shell = f'{{ bash {script}; echo "BLOCK EXIT $?"; }} > {root / out} 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', TMUX, '-c', str(root), shell], check=True)
    return {**entry, 'action': 'launched'}


def cmd_wait(args) -> int:
    root = Path(args.root)
    end = time.monotonic() + args.minutes * 60
    while True:
        launches = ledger(root)
        if not launches:
            print('no launch', flush=True)
            return 0
        last = launches[-1]
        if last['exit'] is not None or not tmux_alive() or time.monotonic() >= end:
            break
        time.sleep(30)
    text = (root / last['out']).read_text(encoding='utf-8', errors='replace') if (root / last['out']).exists() else ''
    print('\n'.join(text.splitlines()[-8:]), flush=True)
    print(json.dumps({'step': last['step'], 'exit': last['exit'], 'out': last['out']}), flush=True)
    return 0 if last['exit'] is not None or not tmux_alive() else 3


def cmd_approve(args) -> dict:
    import h7r_d1_d4 as dd
    root = Path(args.root)
    action = cmd_next(args)
    if action != {'action': 'approve', 'step': args.mode, 'mode': args.mode}:
        raise ValueError(f'next is not approve {args.mode}: {action}')
    path = root / CONTRACT
    text = path.read_text(encoding='utf-8')
    before = dd.experiment_sha256(json.loads(text))
    old = f'"{args.mode}": false'
    if text.count(old) != 1:
        raise ValueError(f'{CONTRACT}: approved.{args.mode} is not a single "{old}"')
    path.write_text(text.replace(old, f'"{args.mode}": true'), encoding='utf-8')
    c = json.loads(path.read_text(encoding='utf-8'))
    if dd.experiment_sha256(c) != before or c['approved'][args.mode] is not True:
        _git(root, 'checkout', '--', CONTRACT)
        raise ValueError('the approval changed more than the flag')
    evidence = {'smoke': 'D4 entry checks pass on the committed records (fit comparable, H-dev DEVELOPMENT_SIGNAL, '
                         'factory v2 time gate passed on the linked seed-31001 pair); eligible manifest committed',
                'full': 'the committed D4 smoke is a reproduced SMOKE of this experiment; '
                        f'{(DEADLINE - datetime.now(KST)).total_seconds() / 3600:.1f} h left of the delegation'}
    stamp = datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')
    with open(root / STATE, 'a', encoding='utf-8') as f:
        f.write(f'- {stamp}: D4 `approved.{args.mode}` (autopilot driver; 근거: {evidence[args.mode]}).\n')
    _git(root, 'add', CONTRACT, STATE)
    msg = (f'H7-R D1 D4: approve the {args.mode} stage under R2-D1-AUTOPILOT-1\n\n'
           f'Gate checked by scripts/h7r_d1_autopilot.py: {evidence[args.mode]}.\n'
           f'Delegation: {D}/autopilot/delegation-1.json.')
    _git(root, 'commit', '-q', '-m', msg)
    try:  # the loader reads only a committed contract; an approval it refuses is undone before any push
        dd.load_contract(path, root, args.mode)
    except (ValueError, KeyError) as exc:
        _git(root, 'reset', '-q', '--hard', 'HEAD~1')
        raise ValueError(f'the approved contract does not load ({exc}); the local approval commit was undone')
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        if subprocess.run(['git', 'push', '-q', 'origin', 'feat/h7-c1'], cwd=root).returncode == 0:
            return {'action': 'approved', 'mode': args.mode, 'commit': _git(root, 'rev-parse', 'HEAD')}
    raise ValueError('the approval is committed locally but the push failed: push it before any run')


def cmd_report(args) -> dict:
    root = Path(args.root)
    out = {}
    fit = committed_json(root, f'{D}/full/fit.json')
    if fit:
        out['fit'] = {'comparable': fit['comparable'], 'category': fit['reading']['category'] if fit['reading'] else None,
                      'wide_minus_narrow': fit['reading']['wide_minus_narrow'] if fit['reading'] else None}
    h = committed_json(root, DONE['hdev'])
    if h:
        out['hdev'] = h['hdev_reading']
    t = committed_json(root, DONE['timing'])
    if t:
        out['timing'] = {k: t['result'][k] for k in ('passed', 'positions', 'mean_ratio', 'p95_ratio',
                                                    'same_move_share')}
    for mode in ('smoke', 'full'):
        s = committed_json(root, DONE[mode])
        if s:
            body = {'reading': s['reading'], 'actor_reproduced': s['actor_check']['reproduced'],
                    'minutes': round(s['seconds'] / 60, 1), 'branches': s['branches']}
            if 'comparisons' in s:
                body['comparisons'] = s['comparisons']
                body['branch_outcomes'] = s.get('branch_outcomes')
            out[mode] = body
    out['launches'] = [{k: e[k] for k in ('n', 'step', 'kind', 'exit', 'recorded', 'out', 'started_kst')}
                       for e in ledger(root)]
    out['deadline'] = DEADLINE.isoformat()
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default=str(ROOT))
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('next', 'launch', 'approve'):
        s = sub.add_parser(name)
        s.add_argument('--no-pull', action='store_true')
        if name == 'launch':
            s.add_argument('--home', default='~')
        if name == 'approve':
            s.add_argument('mode', choices=('smoke', 'full'))
    w = sub.add_parser('wait')
    w.add_argument('--minutes', type=float, default=9.0)
    sub.add_parser('report')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == 'wait':
            return cmd_wait(args)
        fn = {'next': cmd_next, 'launch': cmd_launch, 'approve': cmd_approve, 'report': cmd_report}[args.command]
        print(json.dumps(fn(args), ensure_ascii=False, indent=1), flush=True)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'action': 'error', 'reason': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False),
              flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
