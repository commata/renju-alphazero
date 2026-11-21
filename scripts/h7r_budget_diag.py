"""TB-V3-BUDGET-DIAG-1: the moves S400 played in E1-P that the depth-2 truth check could not prove
(docs/h7r/budget/TB-V3-BUDGET-DIAG-1-draft.md). A diagnosis: no games, no deployment decision; the END of
TB-V3-BUDGET-1 stays.

    prepare  (approved.prepare) the canonical model files, the committed E1 manifest and the TB-V3-BUDGET-1
             confirmation record are the pinned ones
    replay   (approved.replay) the 5 E1-P positions x their 3 seeds with S50 and with S400 through
             ``e1_evaluate.evaluate_position`` itself (its label rule: a truth of the manifest first, only a missing
             move computed at B0), each run with its route, applied simulations, root tactical score, first
             candidate, final move, check, root children and outcome; beside it, for diagnosis only, the first
             candidate and the final move checked again at B0 with the stop reasons (``lost_depth_info``:
             nodes_used, vcf_calls, vcf_cut_calls, budget_exhausted). The judgement is of the aggregate only:
             S50 TREE_AVOIDED 15 and S400 TREE_AVOIDED 12 + TRUTH_UNRESOLVED 3 is AGGREGATE_REPRODUCED (the original
             runs kept no per-run record: the same position, seed and move cannot be shown either way)
    deepen   (approved.deepen) every (position, move) labelled UNKNOWN in the replay, once, at depth 2 with B1, B2
             and B3 in that order (S400's unresolved runs first), each in its own process killed at 30 minutes
             (TIMEOUT, apart from a solver UNKNOWN), all of it within 2 hours (what is left: NOT_RUN); every move's
             B1 / B2 / B3 results kept with the stop reasons, and the reading of section 5
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402
from h7r_p1 import Guard, committed, digest, file_sha256, require  # noqa: E402

FORMAT = 'h7r-budget-diag-contract-v1'
CONTRACT_ID = 'TB-V3-BUDGET-DIAG-1'
D = 'docs/h7r/budget/TB-V3-BUDGET-DIAG-1'
CONTRACT = f'{D}/diag-contract.json'
RUNS = 'runs/h7r/TB-V3-BUDGET-DIAG-1'
STAGES = ('prepare', 'replay', 'deepen')
FACTORS = (1, 8)  # S50, S400
MODELS = {'policy': 'runs/h3_policy_64x4/best.pt', 'value': 'runs/h6_value_head_v3/best.pt'}
E1_MANIFEST = 'docs/mcts-v8-results/e1/e1_manifest.json'
BUDGET_CONFIRM = 'docs/h7r/budget/TB-V3-BUDGET-1/confirm.json'
EXPECTED = {'1': {'TREE_AVOIDED': 15}, '8': {'TREE_AVOIDED': 12, 'TRUTH_UNRESOLVED': 3}}
B0 = {'node_limit': 20_000, 'call_limit': 100_000, 'node_budget': 10_000_000}  # e1_build_suite.TRUTH_BUDGET
BUDGETS = {'B1': {'node_limit': 200_000, 'call_limit': 100_000, 'node_budget': 10_000_000},
           'B2': {'node_limit': 20_000, 'call_limit': 1_000_000, 'node_budget': 100_000_000},
           'B3': {'node_limit': 200_000, 'call_limit': 1_000_000, 'node_budget': 100_000_000}}
TASK_SECONDS = 1800
TOTAL_SECONDS = 7200
WORKERS = 6
SOURCES = ('scripts/h7r_budget_diag.py', 'scripts/e1_evaluate.py', 'scripts/e1_build_suite.py',
           'scripts/s1_loss_analysis.py', 'scripts/h7c_budget.py', 'scripts/h7r_budget.py', 'scripts/h7r_p1.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'draft', 'models', 'inputs', 'replay', 'deepen',
        'reading', 'stages', 'output', 'sources', 'scope'}
PROVEN_LOSS, CLEAR, UNKNOWN = 'PROVEN_LOSS', 'VCT2_CLEAR', 'UNKNOWN'


# --- contract -----------------------------------------------------------------------------------------------------

def contract_path(path, root=ROOT) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def contract_body(root=ROOT, sources=SOURCES) -> dict:
    from analysis import tb_v3

    root = Path(root)
    import e1_build_suite as eb
    require(dict(eb.TRUTH_BUDGET) == B0, 'B0 is not the E1 truth budget')
    return {
        'format': FORMAT, 'contract_id': CONTRACT_ID, 'status': 'fixed', 'approved': {s: False for s in STAGES},
        'purpose': 'diagnosis of the E1-P moves S400 played that the depth-2 truth check left UNKNOWN in '
                   'TB-V3-BUDGET-1; no games, no deployment decision; the END of TB-V3-BUDGET-1 stays',
        'draft': {'path': f'{D}-draft.md'},
        'models': {'policy': MODELS['policy'], 'value': MODELS['value'], 'identity': dict(tb_v3.MODELS)},
        'inputs': {'e1_manifest': {'path': E1_MANIFEST, 'sha256': file_sha256(root / E1_MANIFEST)},
                   'budget_confirm': {'path': BUDGET_CONFIRM, 'sha256': file_sha256(root / BUDGET_CONFIRM)}},
        'replay': {'class': 'E1-P', 'factors': list(FACTORS), 'seeds': 'each position\'s e1_seeds (3)',
                   'engine': 'h7c_budget.make_budget_agent (canonical models, cuda)', 'workers': WORKERS,
                   'label_rule': 'e1_evaluate.evaluate_position: a truth of the manifest first, a missing move at B0',
                   'diagnostic_recheck': 'the first candidate and the final move of every run at B0 again '
                                         '(lost_depth_info), kept apart from the labels',
                   'expected_aggregate': EXPECTED,
                   'judgement': 'AGGREGATE_REPRODUCED when the E1-P outcome counts equal the expected ones, else '
                                'NOT_REPRODUCED; never a claim about the same position, seed or move',
                   'not_reproduced': 'deepen goes on with this replay\'s UNKNOWN moves'},
        'deepen': {'b0': B0, 'budgets': BUDGETS, 'depth': 2, 'order': 'S400 unresolved runs first (first candidate, '
                   'then final move), then every other UNKNOWN move; per move B1, B2, B3',
                   'dedup': '(position, move) once', 'task_seconds': TASK_SECONDS, 'total_seconds': TOTAL_SECONDS,
                   'timeout': 'the task runs in its own process, killed at the limit: TIMEOUT (apart from a solver '
                              'UNKNOWN); past the total: NOT_RUN'},
        'reading': {'per_move': 'B1 / B2 / B3 each: PROVEN_LOSS, VCT2_CLEAR, UNKNOWN, TIMEOUT or NOT_RUN, with the '
                                'stop reasons',
                    'resolved_by': {'B1 only': 'raising the single VCF call limit was enough',
                                    'B2 only': 'raising the per-move totals was enough',
                                    'B3 only': 'only with both raised (of the tested combinations)',
                                    'B1 and B2': 'either raise was enough: no single cause',
                                    'none': 'no budget resolved it', 'incomplete': 'a TIMEOUT or NOT_RUN: no cause'},
                    'use': ['PROVEN_LOSS among S400\'s failure moves: a real loss within depth 2; with its run\'s '
                            'simulation mode (tactical 800 grounds a separate normal 400 / tactical 100 comparison)',
                            'all VCT2_CLEAR: the B0 limits made them UNKNOWN; no loss within depth 2 only',
                            'UNKNOWN left: open', 'mixed: every count reported'],
                    'scope': 'VCT2_CLEAR means no loss within depth 2, never a good move; the END of '
                             'TB-V3-BUDGET-1 stays; a deployment review is a new contract with a pre-fixed tactical '
                             'rule and a new tactical sample'},
        'stages': {'prepare': 'inputs', 'replay': 'the paired replay', 'deepen': 'the larger budgets and the report'},
        'output': {'prepare': f'{D}/prepare.json', 'replay': f'{D}/replay.json', 'deepen': f'{D}/deepen.json',
                   'runs': RUNS},
        'sources': [{'path': rel, 'sha256': file_sha256(root / rel)} for rel in sources],
        'scope': 'diagnosis only'}


def experiment_sha256(c: dict) -> str:
    return p1.experiment_sha256(c)


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    root = Path(root)
    path = contract_path(path, root)
    c = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the contract keys differ')
    require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
            f'{path}: not the fixed {CONTRACT_ID} contract')
    a = c['approved']
    require(isinstance(a, dict) and list(a) == list(STAGES) and all(type(v) is bool for v in a.values()),
            f'{path}: approved flags')
    for i, s in enumerate(STAGES):
        if a[s]:
            require(all(a[x] for x in STAGES[:i]), f'{path}: {s} approved before {list(STAGES[:i])}')
    dp = c['deepen']
    require(dp['b0'] == B0 and dp['budgets'] == BUDGETS and dp['task_seconds'] == TASK_SECONDS
            and dp['total_seconds'] == TOTAL_SECONDS and c['replay']['factors'] == list(FACTORS)
            and c['replay']['expected_aggregate'] == EXPECTED, f'{path}: the budgets, limits or expectation differ')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if file_sha256(root / r['path']) != r['sha256']]
        require(not changed, f'the checkout differs from the sources the contract was fixed for: {changed}')
    require(committed(path, root), f'{path}: not committed')
    if stage is not None:
        require(a[stage] is True, f'{path}: the {stage} stage is not approved')
    return c, file_sha256(path), experiment_sha256(c)


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def log_field(args, prefix: str) -> dict:
    name = getattr(args, 'log_name', None)
    if name is None:
        return {}
    require(re.fullmatch(rf'{prefix}-[A-Za-z0-9]+\.log', name) is not None, f'--log-name {name!r}')
    return {'log_name': name}


def committed_record(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def check_inputs(c: dict, root) -> dict:
    from analysis import tb_v3, tb_v3_net
    root = Path(root)
    for key, ref in c['inputs'].items():
        path = root / ref['path']
        require(path.is_file() and committed(path, root) and file_sha256(path) == ref['sha256'],
                f"{ref['path']}: not the pinned {key}")
    ident = tb_v3_net.model_identity(root / MODELS['policy'], root / MODELS['value'])
    require(ident == c['models']['identity'] == dict(tb_v3.MODELS), 'the model files are not the canonical pair')
    return ident


def positions(root) -> list[dict]:
    m = json.loads((Path(root) / E1_MANIFEST).read_text(encoding='utf-8'))
    out = [p for p in m['positions'] if p['class'] == 'E1-P']
    require(len(out) == 5 and all(len(p['e1_seeds']) == 3 for p in out), 'E1-P is not 5 positions x 3 seeds')
    return out


def check_device(device: str) -> str:
    require(device == 'cuda', f'--device {device!r}: {CONTRACT_ID} runs on cuda only')
    return device


# --- prepare ------------------------------------------------------------------------------------------------------

def prepare(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    out = root / c['output']['prepare']
    require(not out.exists(), f'{out} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [MODELS['policy'], MODELS['value'], E1_MANIFEST,
                                                                BUDGET_CONFIRM])
    models = check_inputs(c, root)
    confirm = json.loads((root / BUDGET_CONFIRM).read_text(encoding='utf-8'))
    require(confirm.get('verdict') == 'END' and confirm.get('failed') == ['E1 gate'],
            'the TB-V3-BUDGET-1 confirmation is not the END on the E1 gate this diagnosis is about')
    record = {'format': 'h7r-budget-diag-prepare-v1', **log_field(args, 'prepare'), **stamp(sha, eid, guard.check()),
              'passed': True, 'models': models, 'positions': [p['key'] for p in positions(root)],
              'confirm_tactical': confirm['tactical']['gate']['compare']['rescue_rate']}
    p1.publish(out, record)
    return record


# --- replay -------------------------------------------------------------------------------------------------------

def status_of(depth, status) -> str:
    return PROVEN_LOSS if depth is not None else CLEAR if status == 'SAFE' else UNKNOWN


def recheck(history, move, budget) -> dict:
    """A move's depth 0-2 class at ``budget`` with the stop reasons (``s1_loss_analysis.lost_depth_info``)."""
    from s1_loss_analysis import lost_depth_info
    started = time.perf_counter()
    depth, status, info = lost_depth_info([*history, list(move)], len(history), budget)
    return {'status': status_of(depth, status), 'lost_depth': depth, **info,
            'seconds': round(time.perf_counter() - started, 2)}


def replay_position(task) -> dict:
    """One E1-P position with one budget, its three seeds through ``e1_evaluate.evaluate_position`` (the agent of
    each run kept to read what the tree applied)."""
    import e1_evaluate as ev
    import h7c_budget as hb

    position, factor, policy, value, device = task
    agents = []

    def make(seed):
        agent = hb.make_budget_agent(factor, policy, value, seed, device)
        agents.append(agent)
        return agent
    result = ev.evaluate_position(position, make, f'S{50 * factor}')
    truths = {**position['truths'], **result['truths_added']}
    runs = []
    for run, agent in zip(result['runs'], agents):
        d = agent.diagnostics
        moves = {'pre': run['pre'], 'played': run['played']}
        runs.append({**run, 'factor': factor, 'simulation_mode': getattr(d, 'simulation_mode', None),
                     'selected_simulations': getattr(d, 'selected_simulations', None),
                     'tree_simulations': getattr(d, 'v8_tree_simulations', None),
                     'root_tactical_score': getattr(d, 'best_root_tactical_score', None),
                     'labels': {k: (truths.get(f'{m[0]},{m[1]}') or {}).get('status') for k, m in moves.items()},
                     'label_source': {k: 'manifest' if f'{m[0]},{m[1]}' in position['truths'] else 'computed_b0'
                                      for k, m in moves.items()},
                     'b0_recheck': {k: recheck(position['history'], m, B0) for k, m in moves.items()}})
    return {'key': position['key'], 'factor': factor, 'runs': runs, 'truths_added': result['truths_added']}


def outcome_counts(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        c = out.setdefault(str(r['factor']), Counter())
        c.update(run['outcome'] for run in r['runs'])
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def replay(args, root=ROOT, log=print) -> dict:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import get_context
    import h7c_arena as ar
    from hybrid.progress import Progress, run_chunks

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'replay')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    out = root / c['output']['replay']
    require(not out.exists(), f'{out} exists: never overwritten')
    device = check_device(args.device)
    guard = Guard(root, contract_path(args.contract, root), c, [MODELS['policy'], MODELS['value'], E1_MANIFEST])
    check_inputs(c, root)
    git = guard.start['git']
    ident = {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'git': git, 'device': device,
             'models': c['models']['identity']}
    sig = digest(ident)
    jsonl = root / RUNS / 'replay.jsonl'
    keys = [f"{p['key']}|S{50 * f}" for f in FACTORS for p in positions(root)]
    done = ar.load_finished(jsonl, set(keys), sig, log=log)
    todo = [(p, f) for f in FACTORS for p in positions(root) if f"{p['key']}|S{50 * f}" not in done]
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if getattr(args, 'threads_for_tests', False) else (
            lambda: ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn')))
        with jsonl.open('a', encoding='utf-8') as fh, pool_of() as pool:
            def keep(_, row):
                row = {**row, 'key': f"{row['key']}|S{50 * row['factor']}", 'position': row['key'], 'signature': sig}
                fh.write(json.dumps(row, separators=(',', ':')) + '\n')
                fh.flush()
                done[row['key']] = row
            run_chunks(pool, replay_position, [(p, f, str(root / MODELS['policy']), str(root / MODELS['value']),
                                                device) for p, f in todo],
                       Progress('budget-diag-replay', len(todo), 'positions'), lambda _: 1,
                       unit_name='positions', on_result=keep)
    rows = [done[k] for k in keys]
    record = {'format': 'h7r-budget-diag-replay-v1', **log_field(args, 'replay'), **stamp(sha, eid, guard.check()),
              'device': device, 'jsonl': jsonl.relative_to(root).as_posix(), 'jsonl_sha256': file_sha256(jsonl),
              'identity_sha256': sig, **replay_summary(rows)}
    p1.publish(out, record)
    return record


def replay_summary(rows: list[dict]) -> dict:
    counts = outcome_counts(rows)
    judged = 'AGGREGATE_REPRODUCED' if counts == EXPECTED else 'NOT_REPRODUCED'
    runs = [{'position': r['position'], 'factor': r['factor'], **{k: run[k] for k in (
        'seed', 'route', 'simulation_mode', 'selected_simulations', 'root_tactical_score', 'pre', 'played',
        'switched', 'outcome', 'labels', 'label_source')},
        'b0_recheck': {k: {x: v[x] for x in ('status', 'lost_depth', 'nodes_used', 'vcf_calls', 'vcf_cut_calls',
                                              'budget_exhausted')} for k, v in run['b0_recheck'].items()}}
        for r in rows for run in r['runs']]
    return {'outcomes': counts, 'expected': EXPECTED, 'judgement': judged,
            'note': 'an aggregate judgement only: the original runs kept no per-run record', 'runs': runs}


def replay_rows(c: dict, eid: str, rec: dict, root) -> list[dict]:
    root = Path(root)
    path = root / rec['jsonl']
    require(path.is_file() and file_sha256(path) == rec['jsonl_sha256'], f"{rec['jsonl']}: not the recorded replay")
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = [f"{p['key']}|S{50 * f}" for f in FACTORS for p in positions(root)]
    by = {r['key']: r for r in rows}
    require(set(by) == set(keys) and len(rows) == len(keys)
            and {r['signature'] for r in rows} == {rec['identity_sha256']},
            'the replay file is not one complete replay')
    return [by[k] for k in keys]


# --- deepen -------------------------------------------------------------------------------------------------------

def items_of(rows: list[dict], root) -> list[dict]:
    """Every (position, move) the replay labelled UNKNOWN, once, S400's unresolved runs first."""
    hist = {p['key']: p['history'] for p in positions(root)}
    first, rest, seen = [], [], set()
    runs = [(r, run) for r in rows for run in r['runs']]
    for r, run in sorted(runs, key=lambda x: (not (x[0]['factor'] == 8 and x[1]['outcome'] == 'TRUTH_UNRESOLVED'),
                                              -x[0]['factor'], x[0]['position'], x[1]['seed'])):
        for which in ('pre', 'played'):
            if run['labels'][which] != UNKNOWN:
                continue
            key = (r['position'], tuple(run[which]))
            if key in seen:
                continue
            seen.add(key)
            item = {'position': r['position'], 'move': list(run[which]), 'history': hist[r['position']],
                    'from': {'factor': r['factor'], 'seed': run['seed'], 'as': which, 'outcome': run['outcome'],
                             'simulation_mode': run['simulation_mode']}}
            (first if r['factor'] == 8 and run['outcome'] == 'TRUTH_UNRESOLVED' else rest).append(item)
    return first + rest


def _child(fn, args, queue):
    queue.put(fn(*args))


def run_with_timeout(fn, args: tuple, seconds: float):
    """``fn(*args)`` in its own process; killed at ``seconds`` (returns ('TIMEOUT', elapsed))."""
    from multiprocessing import get_context
    ctx = get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(fn, args, queue))
    started = time.perf_counter()
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return 'TIMEOUT', round(time.perf_counter() - started, 2)
    require(proc.exitcode == 0 and not queue.empty(), f'the task process failed (exit {proc.exitcode})')
    return queue.get(), round(time.perf_counter() - started, 2)


def resolved_by(per: dict) -> str:
    st = {b: per[b]['status'] for b in BUDGETS}
    if any(v in ('TIMEOUT', 'NOT_RUN') for v in st.values()):
        return 'incomplete'
    solved = {b for b, v in st.items() if v in (PROVEN_LOSS, CLEAR)}
    if {'B1', 'B2'} <= solved:
        return 'B1 and B2'
    if 'B1' in solved:
        return 'B1 only'
    if 'B2' in solved:
        return 'B2 only'
    return 'B3 only' if 'B3' in solved else 'none'


def final_status(per: dict) -> str:
    st = [per[b]['status'] for b in BUDGETS]
    for s in (PROVEN_LOSS, CLEAR):
        if s in st:
            return s
    return 'UNKNOWN' if all(s == UNKNOWN for s in st) else 'INCOMPLETE'


def report(items: list[dict]) -> dict:
    failure = [i for i in items if i['from']['factor'] == 8 and i['from']['outcome'] == 'TRUTH_UNRESOLVED']
    count = lambda xs: dict(sorted(Counter(final_status(i['results']) for i in xs).items()))  # noqa: E731
    lost = [i for i in failure if final_status(i['results']) == PROVEN_LOSS]
    use = []
    if lost:
        use.append(f"PROVEN_LOSS among S400's failure moves: {len(lost)} (simulation modes "
                   f"{sorted({i['from']['simulation_mode'] or 'unknown' for i in lost})})")
    c = count(failure)
    if failure and c.get(CLEAR) == len(failure):
        use.append('all VCT2_CLEAR: the B0 limits made them UNKNOWN; no loss within depth 2 only')
    if c.get('UNKNOWN') or c.get('INCOMPLETE'):
        use.append(f"left open: {c.get('UNKNOWN', 0)} UNKNOWN, {c.get('INCOMPLETE', 0)} INCOMPLETE")
    return {'failure_moves': len(failure), 'failure_final': c, 'all_moves_final': count(items),
            'resolved_by': dict(sorted(Counter(resolved_by(i['results']) for i in items).items())),
            'tactical_800_with_proven_loss': any(i['from']['simulation_mode'] == 'tactical' for i in lost),
            'use': use or ['no S400 failure move in this replay']}


def deepen(args, root=ROOT, log=print, runner=None) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'deepen')
    rrec = committed_record(root, c['output']['replay'], 'replay record', eid)
    out = root / c['output']['deepen']
    require(not out.exists(), f'{out} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [c['output']['replay'], rrec['jsonl'], E1_MANIFEST])
    rows = replay_rows(c, eid, rrec, root)
    items = items_of(rows, root)
    runner = runner or (lambda item, budget, seconds: run_with_timeout(recheck, (item['history'], item['move'],
                                                                                 BUDGETS[budget]), seconds))
    jsonl = root / RUNS / 'deepen.jsonl'
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                require(row.get('replay_sha256') == rrec['jsonl_sha256'], f'{jsonl}: rows of another replay')
                done[(row['position'], tuple(row['move']), row['budget'])] = row
    used = sum(r['seconds'] for r in done.values())
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open('a', encoding='utf-8') as fh:
        for item in items:
            for budget in BUDGETS:
                key = (item['position'], tuple(item['move']), budget)
                if key in done:
                    continue
                left = TOTAL_SECONDS - used
                if left <= 0:
                    row = {'status': 'NOT_RUN', 'seconds': 0.0, 'reason': 'the 2 hour total was reached'}
                else:
                    limit = min(TASK_SECONDS, left)
                    result, seconds = runner(item, budget, limit)
                    used += seconds
                    row = {'status': 'TIMEOUT', 'seconds': seconds,
                           'reason': 'killed at the task limit' if limit == TASK_SECONDS else
                           'killed at the remaining total'} if result == 'TIMEOUT' else {**result, 'seconds': seconds}
                row = {'position': item['position'], 'move': item['move'], 'budget': budget,
                       'replay_sha256': rrec['jsonl_sha256'], **row}
                if row['status'] != 'NOT_RUN':  # NOT_RUN is never journaled: a rerun after the cap tries nothing
                    fh.write(json.dumps(row, separators=(',', ':')) + '\n')
                    fh.flush()
                done[key] = row
                log(f"[budget-diag] {item['position']} {item['move']} {budget}: {row['status']}", flush=True)
    for item in items:
        item['results'] = {b: {k: v for k, v in done[(item['position'], tuple(item['move']), b)].items()
                               if k not in ('position', 'move', 'budget', 'replay_sha256')} for b in BUDGETS}
        item['b0'] = next(run['b0_recheck'][item['from']['as']] for r in rows for run in r['runs']
                          if r['position'] == item['position'] and list(run[item['from']['as']]) == item['move'])
        item['resolved_by'] = resolved_by(item['results'])
        item['final'] = final_status(item['results'])
    record = {'format': 'h7r-budget-diag-deepen-v1', **log_field(args, 'deepen'), **stamp(sha, eid, guard.check()),
              'replay_judgement': rrec['judgement'], 'jsonl': jsonl.relative_to(root).as_posix(),
              'jsonl_sha256': file_sha256(jsonl), 'seconds_used': round(used, 1),
              'items': [{k: v for k, v in i.items() if k != 'history'} for i in items], 'report': report(items),
              'scope': 'VCT2_CLEAR is no loss within depth 2 only; the END of TB-V3-BUDGET-1 stays'}
    p1.publish(out, record)
    return record


# --- verify / main ------------------------------------------------------------------------------------------------

def check_stamp(c: dict, eid: str, rec: dict, stage: str, root) -> None:
    import hashlib
    rel = c['output'][stage]
    git = rec.get('git') or {}
    require(rec.get('format') == f'h7r-budget-diag-{stage}-v1' and rec.get('experiment_sha256') == eid
            and git.get('dirty') is False and git.get('commit'), f'{rel}: not a {stage} record of this experiment')
    try:
        text = subprocess.check_output(['git', 'show', f"{git['commit']}:{CONTRACT}"], cwd=root, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{rel}: its commit {git['commit'][:12]} has no contract") from exc
    require(hashlib.sha256(text.encode('utf-8')).hexdigest() == rec.get('contract_file_sha256')
            and json.loads(text)['approved'].get(stage) is True, f'{rel}: not run under the approved contract')


def verify(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root)
    rel = c['output'][args.stage]
    rec = json.loads((root / rel).read_text(encoding='utf-8'))
    check_stamp(c, eid, rec, args.stage, root)
    if args.stage == 'prepare':
        require(rec.get('passed') is True and rec.get('models') == check_inputs(c, root),
                f'{rel}: not a passed prepare')
    elif args.stage == 'replay':
        rows = replay_rows(c, eid, rec, root)
        require({k: v for k, v in replay_summary(rows).items()} == {k: rec[k] for k in replay_summary(rows)},
                f'{rel}: not what its replay file gives')
    else:
        rrec = committed_record(root, c['output']['replay'], 'replay record', eid)
        rows = replay_rows(c, eid, rrec, root)
        items = items_of(rows, root)
        require([(i['position'], i['move']) for i in items] == [(i['position'], i['move']) for i in rec['items']],
                f'{rel}: not the UNKNOWN moves of the replay')
        path = root / rec['jsonl']
        require(file_sha256(path) == rec['jsonl_sha256'], f"{rec['jsonl']} changed")
        require(report(rec['items']) == rec['report'], f'{rel}: the report is not what its items give')
    return {'stage': args.stage, 'log_name': rec.get('log_name'), 'exit_code': 0, 'stage_files': [rel]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in STAGES:
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
        s.add_argument('--log-name', default=None)
    v = sub.add_parser('verify')
    v.add_argument('--stage', required=True, choices=STAGES)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            print(json.dumps(verify(args), separators=(',', ':')), flush=True)
            return 0
        rec = {'prepare': prepare, 'replay': replay, 'deepen': deepen}[args.command](args)
        keys = ('passed', 'outcomes', 'judgement', 'report', 'seconds_used')
        print(json.dumps({k: rec[k] for k in keys if k in rec}, ensure_ascii=False, indent=1), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
