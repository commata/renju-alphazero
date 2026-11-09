"""R2-D1-AUTOPILOT-1 driver: the delegation's progression rules as code, for an agent on the desktop.

The agent never decides what comes next; it asks this tool, does exactly that, and reports when the tool says stop.

    next       the next action from the committed records and the launch ledger (JSON on stdout). While a block
               runs it neither pulls nor checks the tree (the block's HEAD must not move); otherwise it pulls on a
               clean tree (the agent itself never runs git pull). New files staged under the record folders (a block
               stopped between its git add and its commit) allow only wait / recover / stop; other changes refuse.
                 wait       a launched block has not finished (``wait``)
                 run        launch a block (``launch``)
                 recover    launch the record-preserving block of a failed step (``launch``), then stop
                 approve    commit a sub-approval after its gate (``approve MODE``)
                 interrupt  the delegation limit passed while a block runs (``interrupt``, then ``wait``)
                 stop       end the delegation run here and report (``final`` true: the delegation itself ends)
    launch     the action of ``next`` when it is run / recover: blocks extracted from the D1 draft into ``--home``,
               one tmux session ``d1auto``, output and exit code in the ledger folder; the ledger marks the launch
               ``started`` or, when tmux refuses, ``start_failed`` (not an attempt)
    wait       poll the running block for up to ``--minutes``; exit 0 when it finished, 3 while it still runs
    interrupt  Ctrl-C to the running block, only when ``next`` says interrupt; marked in the ledger
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

Time: every run, resume, retry and approval needs ``MIN_HOURS[step]`` left before the delegation limit; a
block still running at the limit is interrupted; recovery is allowed after the limit.

Failures: a step that failed stops the run (a block that ended without BLOCK EXIT counts as failed). The one retry is the delegation's: a D4 run (smoke or full) whose tool
log shows no ``STOP:`` refusal (a crash or kill, not a check) is resumed once with the same command and HEAD. A step
whose block did not reach its records commit gets its record-preserving block once (hdev: d1-full-3F.sh; D4:
d1-d4-F.sh), then the run stops. Past the delegation's 24 hours nothing is launched or approved.

Authorized retry: after the cloud session has identified the cause of a failure that is independent of any reading
and committed the fix, ``autopilot/retries.json`` names that failed launch (step and ledger number) with the cause and
the fix; that launch's step is then run once more. Only for ``AUTH_STEPS`` (smoke only when its failed run left no resume state; the tool refuses otherwise). A retried
D4 stage whose approval is not committed (an amendment reset it) is approved first, through its gate.
The eligible manifest's path is the contract's ``output.sample``.
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
MIN_HOURS = {'hdev': 0.5, 'timing': 0.5, 'eligible': 0.25, 'smoke': 0.5, 'full': MIN_HOURS_FOR_FULL}
RECORD_DIRS = (f'{D}/full/', f'{D}/factory-v2/', f'{D}/d4/')  # where an interrupted block may leave staged records

STEPS = ('hdev', 'timing', 'eligible', 'smoke', 'full')
BLOCKS = {'hdev': ('d1-full-3.sh', []), 'timing': ('d1-v2-1.sh', []), 'eligible': ('d1-d4-1.sh', []),
          'smoke': ('d1-d4-run.sh', ['smoke']), 'full': ('d1-d4-run.sh', ['full'])}
RECOVER = {'hdev': ('d1-full-3F.sh', []), 'smoke': ('d1-d4-F.sh', ['smoke']), 'full': ('d1-d4-F.sh', ['full'])}
RETRIES = f'{D}/autopilot/retries.json'
AUTH_STEPS = ('timing', 'eligible', 'smoke')
A1_SAMPLE = f'{D}/d4/eligible.json'  # the manifest of the A1 contract (amendment A2)
DONE = {'hdev': f'{D}/full/d1_eval.json', 'timing': f'{D}/factory-v2/timing.json',  # eligible: output.sample
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

def now_kst() -> datetime:
    return datetime.now(KST)


def ledger(root, *, all_entries: bool = False) -> list[dict]:
    """The launches, each with its outcome. A launch line is followed by a mark line: ``started`` once tmux took the
    block, ``start_failed`` when it did not, ``interrupted`` when the deadline stopped it. A launch whose block never
    started (start_failed, or no mark and no output file) is not an attempt and is left out unless ``all_entries``."""
    path = Path(root) / LEDGER_DIR / 'launches.jsonl'
    if not path.exists():
        return []
    entries, marks = [], {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if 'mark' in row:
                marks.setdefault(row['n'], []).append(row['mark'])
            else:
                entries.append(row)
    out = []
    for entry in entries:
        m = marks.get(entry['n'], [])
        started = 'started' in m or ('start_failed' not in m and (Path(root) / entry['out']).exists())
        row = {**entry, **outcome(root, entry), 'started_ok': started, 'interrupted': 'interrupted' in m}
        if started or all_entries:
            out.append(row)
    return out


def append_ledger(root, row: dict) -> None:
    folder = Path(root) / LEDGER_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / 'launches.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


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


def authorized(root) -> dict:
    """(step, failed launch number) -> cause, from the committed retry record."""
    rec = committed_json(root, RETRIES)
    if rec is None:
        return {}
    if rec.get('format') != 'h7r-autopilot-retries-v1':
        raise ValueError(f'{RETRIES}: unknown format')
    out = {}
    for r in rec['retries']:
        if r['step'] not in AUTH_STEPS or not r.get('cause') or not r.get('fix'):
            raise ValueError(f'{RETRIES}: not an authorizable retry: {r}')
        out[(r['step'], int(r['failed_launch']))] = f"{r['cause']} (fix: {r['fix']})"
    return out


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
    if step == 'eligible' and c['output']['sample'] != A1_SAMPLE:
        a1 = committed_json(root, A1_SAMPLE)
        strip = lambda m: {k: v for k, v in m.items() if k != 'contract_sha256'}  # noqa: E731
        if a1 is not None and strip(a1) != strip(rec):
            return act('stop', final=False, reason=f"{c['output']['sample']} lists other games / rows than the A1 "
                                                   f'manifest {A1_SAMPLE} (amendment A2 changes only the binding)')
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
        if committed_json(root, done_path('eligible', c)) is None:
            return 'the eligible manifest is not committed'
        return None
    left = (DEADLINE - now).total_seconds() / 3600
    if left < MIN_HOURS_FOR_FULL:
        return f'{left:.1f} h left of the delegation: below {MIN_HOURS_FOR_FULL} h for the D4 full run'
    return None


def done_path(step: str, c: dict) -> str:
    return c['output']['sample'] if step == 'eligible' else DONE[step]


def running(launches: list[dict], alive) -> dict | None:
    """The launch whose block is still running, if any."""
    if launches and launches[-1]['exit'] is None and alive():
        return launches[-1]
    return None


def decide(root=ROOT, now: datetime | None = None, alive=None) -> dict:
    now = now or now_kst()
    alive = tmux_alive if alive is None else alive
    launches = ledger(root)
    live = running(launches, alive)
    if live:
        if now >= DEADLINE:
            if live['interrupted']:
                return act('wait', step=live['step'], out=live['out'], reason='interrupted at the delegation limit')
            return act('interrupt', step=live['step'], out=live['out'],
                       reason=f'the delegation limit ({DEADLINE.isoformat()}) has passed while {live["step"]} runs')
        return act('wait', step=live['step'], out=live['out'])
    for e in launches:  # a block that ended without BLOCK EXIT (killed, interrupted, session lost): a failure
        if e['exit'] is None:
            e['exit'] = -1
    c = json.loads((Path(root) / CONTRACT).read_text(encoding='utf-8'))
    hours_left = (DEADLINE - now).total_seconds() / 3600
    blocked = []

    def timed(step, action):
        """``action`` (run / approve) only with the step's minimum time left before the delegation limit."""
        if hours_left < MIN_HOURS[step]:
            blocked.append(f'{step} needs {MIN_HOURS[step]} h, {max(hours_left, 0):.2f} h left before '
                           f'{DEADLINE.isoformat()}')
            return None
        return action
    for step in STEPS:
        rec = committed_json(root, done_path(step, c))
        if rec is not None:
            stop = gate(root, step, rec, c)
            if stop:
                return stop
            continue
        mine = [e for e in launches if e['step'] == step and e['kind'] == 'run']
        recovered = [e for e in launches if e['step'] == step and e['kind'] == 'recover']
        def approved_or_approve():
            if step in ('smoke', 'full') and c['approved'][step] is not True:
                why = approval_gate(root, step, c, now)
                if why:
                    return act('stop', final=True, reason=f'approved.{step} cannot be given: {why}')
                return act('approve', step=step, mode=step)
            return None

        def deadline_stop():
            return act('stop', final=True, reason=f'the delegation limit: {"; ".join(blocked)}')
        if not mine:
            block, args = BLOCKS[step]
            return timed(step, approved_or_approve() or act('run', step=step, block=block, args=args)) \
                or deadline_stop()
        last = mine[-1]
        tried = [e for e in recovered if e['n'] > last['n']]
        preserved = [e for e in tried if e['exit'] == 0]
        if tried and not preserved:
            return act('stop', final=now >= DEADLINE,
                       reason=f'{step} failed (exit {last["exit"]}, {last["out"]}) and its record preservation failed '
                              f'too (exit {tried[-1]["exit"]}, {tried[-1]["out"]}): records not committed; report')
        if last['exit'] == 0 and not last['recorded']:
            return act('stop', final=False, reason=f'{step}: exit 0 without a records commit ({last["out"]})')
        if len(mine) == 1 and not preserved and not last['interrupted'] and retryable(root, step, last):
            block, args = BLOCKS[step]
            resume = timed(step, act('run', step=step, block=block, args=args, retry=True,
                                     reason='the D4 run died without a tool refusal: one resume (delegation retry '
                                            'rule)'))
            if resume:
                return resume
        if step in RECOVER and not last['recorded'] and not tried:
            block, args = RECOVER[step]
            return act('recover', step=step, block=block, args=args,
                       reason=f'{step} failed (exit {last["exit"]}) before its records commit')
        why = authorized(root).get((step, last['n']))
        if why:
            block, args = BLOCKS[step]
            retry = timed(step, approved_or_approve() or act(
                'run', step=step, block=block, args=args, retry=True,
                reason=f'authorized retry of launch {last["n"]} ({RETRIES}): {why}'))
            if retry:
                return retry
        if blocked:
            return deadline_stop()
        if now >= DEADLINE:
            return act('stop', final=True, reason=f'the delegation limit ({DEADLINE.isoformat()}) has passed; {step} '
                                                  f'ended with exit {last["exit"]} ({last["out"]}); records '
                                                  f"{'preserved' if preserved or last['recorded'] else 'not committed'}")
        if preserved:
            return act('stop', final=False, reason=f'{step} failed (exit {last["exit"]}, {last["out"]}); its '
                                                   f'records were preserved ({preserved[-1]["out"]}); report')
        return act('stop', final=False, reason=f'{step} failed (exit {last["exit"]}); records '
                                               f"{'committed' if last['recorded'] else 'not committed'} "
                                               f'({last["out"]}); report')
    return act('stop', final=True, reason='every step is complete')


# --- commands --------------------------------------------------------------------------------------------------------

def tree_state(root) -> str:
    """``clean``; ``records`` when the only tracked changes are new files staged under ``RECORD_DIRS`` (a block
    stopped between its ``git add`` and its commit; its record-preserving block resets and recommits them); any other
    change is refused."""
    if _git(root, 'branch', '--show-current') != 'feat/h7-c1':
        raise ValueError('not on feat/h7-c1')
    lines = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                                    text=True).splitlines()
    if not lines:
        return 'clean'
    other = [line for line in lines if not (line[:2] == 'A ' and line[3:].startswith(RECORD_DIRS))]
    if other:
        raise ValueError(f'the tracked tree has changes that are not staged records: {other}')
    return 'records'


def require_clean(root):
    if tree_state(root) != 'clean':
        raise ValueError('the tracked tree is not clean')


def cmd_next(args) -> dict:
    """Never pulls or checks the tree while a block runs (its HEAD must not move); pulls only on a clean tree; with
    staged records left by an interrupted block only waiting, recovering or stopping is allowed."""
    root = Path(args.root)
    head = lambda: _git(root, 'rev-parse', '--short', 'HEAD')  # noqa: E731
    if running(ledger(root), tmux_alive):
        return {**decide(root), 'head': head(), 'pulled': False}
    state = tree_state(root)
    if state == 'clean':
        _git(root, 'pull', '--ff-only', '-q', 'origin', 'feat/h7-c1')  # the cloud session's fixes arrive here
    action = decide(root)
    if state == 'records' and action['action'] not in ('recover', 'stop', 'wait', 'interrupt'):
        action = act('stop', final=False, reason=f"staged records of an interrupted block are in the tree and the "
                                                 f"next action would be {action['action']} {action.get('step')}: "
                                                 'report (no pull, nothing launched)')
    return {**action, 'head': head(), 'pulled': state == 'clean', 'tree': state}


def cmd_launch(args) -> dict:
    root = Path(args.root)
    action = cmd_next(args)
    if action['action'] not in ('run', 'recover'):
        raise ValueError(f'nothing to launch: {action}')
    home = Path(args.home).expanduser()
    for name, text in extract_blocks(root).items():
        (home / name).write_text(text, encoding='utf-8')
    n = len(ledger(root, all_entries=True)) + 1
    out = f"{LEDGER_DIR}/{n:02d}-{action['step']}-{action['action']}.out"
    if (root / out).exists():
        raise ValueError(f'{out} exists')
    script = ' '.join([str(home / action['block']), *action['args']])
    entry = {'n': n, 'step': action['step'], 'kind': action['action'], 'block': action['block'],
             'args': action['args'], 'out': out, 'started': time.time(),
             'started_kst': now_kst().isoformat(timespec='seconds')}
    append_ledger(root, entry)
    shell = f'{{ bash {script}; echo "BLOCK EXIT $?"; }} > {root / out} 2>&1'
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', TMUX, '-c', str(root), shell], check=True,
                       capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, 'stderr', None) or str(exc)
        append_ledger(root, {'n': n, 'mark': 'start_failed', 'error': str(detail).strip()[:500]})
        raise ValueError(f'tmux did not start the block (recorded as start_failed, not an attempt): {detail}')
    append_ledger(root, {'n': n, 'mark': 'started'})
    return {**entry, 'action': 'launched'}


def cmd_interrupt(args) -> dict:
    """At the delegation limit: Ctrl-C to the running block (only when ``next`` says interrupt). The block then ends
    without its records commit; ``next`` gives its record-preserving block and stops."""
    root = Path(args.root)
    action = decide(root)
    if action['action'] != 'interrupt':
        raise ValueError(f'next is not interrupt: {action}')
    live = running(ledger(root), tmux_alive)
    subprocess.run(['tmux', 'send-keys', '-t', TMUX, 'C-c'], check=True, capture_output=True)
    append_ledger(root, {'n': live['n'], 'mark': 'interrupted', 'at': now_kst().isoformat(timespec='seconds')})
    return {'action': 'interrupted', 'step': live['step'], 'out': live['out']}


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
    action = {k: v for k, v in cmd_next(args).items() if k not in ('head', 'pulled', 'tree')}
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
                        f'{(DEADLINE - now_kst()).total_seconds() / 3600:.1f} h left of the delegation'}
    stamp = now_kst().strftime('%Y-%m-%d %H:%M KST')
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
    out['launches'] = [{k: e[k] for k in ('n', 'step', 'kind', 'exit', 'recorded', 'out', 'started_kst',
                                          'started_ok', 'interrupted')} for e in ledger(root, all_entries=True)]
    out['deadline'] = DEADLINE.isoformat()
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default=str(ROOT))
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('interrupt')
    for name in ('next', 'launch', 'approve'):
        s = sub.add_parser(name)
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
        fn = {'next': cmd_next, 'launch': cmd_launch, 'approve': cmd_approve, 'report': cmd_report,
              'interrupt': cmd_interrupt}[args.command]
        print(json.dumps(fn(args), ensure_ascii=False, indent=1), flush=True)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'action': 'error', 'reason': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False),
              flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
