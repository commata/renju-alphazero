"""R2-P1-POLICY-REPLAY-A1: the policy replay control experiment under amendment A1
(docs/h7r/r2/R2-P1-POLICY-REPLAY-A1-draft.md).

A1 replaces the reproduction gate of R2-P1-POLICY-REPLAY-1 (experiment e41841eaafa8, its FAIL record a1cd152 kept as
it is; diagnosis docs/h7r/r2/R2-P1-DIAG-1-draft.md): the past c2 values, eligibility and selected step become a
reference record; the inputs, model identities, training settings and step structure stay strict; a current-environment
equivalence gate under deterministic settings decides, by exact equality; the main runs use the same settings and the
same entry point. The arms, mu, seeds, training amount, Arena size and reading are the P1 contract's.

    prepare         (approved.prepare) every input against its committed expectation (``h7r_p1.check_inputs``)
    gate-run        (approved.gate) one gate run ``--run E1..E5`` in its own process: E1 the c2 trainer
                    (``h7c_train.train_policy``), E2 / E3 the P1 trainer as arm C (replay off, diagnostic on),
                    E4 / E5 as arm T (replay on, diagnostic on); seed 20261024
    gate            (approved.gate) the committed gate records judged: strict inputs, identities, settings and step
                    structure; G1 E2 = E3 and E4 = E5, G2 E1 = E2 (the diagnostic records left out), all exact; the
                    c2 comparison recorded only (a reference)
    train           (approved.train) one main run ``--arm C|T --seed S``: the committed gate re-judged first; the
                    run's numeric settings must be the gate's
    link            (approved.train) the committed seed records: the candidate seed's T and C both CANDIDATE
    arena-contract  (approved.arena) the h7r_eval Arena contract from the verified link
    judge           (approved.arena) as in R2-P1-POLICY-REPLAY-1: the exact Arena contract, complete arm records,
                    P and D (suite-stratified, 50:50, paired), the class-level sensitivity, the fixed interpretation

Every training run (gate and main) goes through ``execute``: the deterministic settings
(``torch.use_deterministic_algorithms(True)``, cuDNN deterministic, benchmark off; CUBLAS_WORKSPACE_CONFIG set before
the process starts), checked against the contract; passive per-step fingerprints (``h7r_p1_diag.Fingerprints``:
cycle batch, D4 symmetries, learning rate, training loss); the selected state's hash and its log-probabilities on
fixed positions. Invariance and identity as in R2-P1-POLICY-REPLAY-1: HEAD (clean), the contract file, the sources and
the read files at the start, checked again right before publishing; every record carries the experiment SHA-256 (the
contract without ``approved``) and its contract file SHA-256. Records are written once.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402  (unchanged; pinned by this contract's sources)
from h7r_p1 import Guard, committed, file_sha256, publish, require  # noqa: E402

FORMAT = 'h7r-p1a1-contract-v1'
CONTRACT_ID = 'R2-P1-POLICY-REPLAY-A1'
D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1'
CONTRACT = f'{D}/p1a1-contract.json'
SEED_RECORDS = f'{D}/seed-records'
GATE_RECORDS = f'{D}/gate-records'
RUNS = 'runs/h7r/R2-P1-POLICY-REPLAY-A1'
ARENA_CONTRACT = 'docs/h7r/contracts/R2-P1-A1-ARENA-1.json'
ARENA_ID = 'R2-P1-A1-ARENA-1'
ARMS = ('C', 'T')
STAGES = ('prepare', 'gate', 'train', 'arena')
GATE_RUNS = {'E1': 'old', 'E2': 'C', 'E3': 'C', 'E4': 'T', 'E5': 'T'}
G1 = (('E2', 'E3'), ('E4', 'E5'))
G2 = ('E1', 'E2')
SET_BY_THE_TOOL = ('deterministic_algorithms', 'cudnn_deterministic', 'cudnn_benchmark')
STRUCTURE = ('total_steps', 'steps_per_epoch', 'warmup_steps')


def experiment_sha256(c: dict) -> str:
    return p1.experiment_sha256(c)


def contract_path(path, root) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    """The fixed, committed A1 contract; its file SHA and experiment SHA. ``stage`` must be approved (in order)."""
    root = Path(root)
    path = contract_path(path, root)
    c = json.loads(path.read_text(encoding='utf-8'))
    require(c.get('format') == FORMAT and c.get('contract_id') == CONTRACT_ID, f'{path}: not the {CONTRACT_ID} contract')
    require(c.get('status') == 'fixed', f'{path}: the contract is not fixed')
    approved = c.get('approved')
    require(isinstance(approved, dict) and set(approved) == set(STAGES)
            and all(isinstance(v, bool) for v in approved.values()), f'{path}: approved flags')
    for i, s in enumerate(STAGES):
        if approved[s]:
            require(all(approved[x] for x in STAGES[:i]), f'{path}: {s} approved before {STAGES[:i]}')
    gate = c['stages']['gate']
    require(gate['runs'] == GATE_RUNS and gate['seed'] == c['seeds']['arena_candidate_training_seed'],
            f'{path}: the gate runs / seed are not the A1 ones')
    det = c['determinism']
    require(all(det['settings'].get(k) == v for k, v in (('deterministic_algorithms', True),
                                                          ('cudnn_deterministic', True), ('cudnn_benchmark', False)))
            and det['environment'].get('CUBLAS_WORKSPACE_CONFIG'), f'{path}: the deterministic settings')
    if stage is not None:
        require(approved[stage], f'{path}: the {stage} stage is not approved')
    if check_sources:
        for ref in c['sources']:
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: differs from the contract")
        for ref in c['evidence'].values():
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: evidence differs")
        require(committed(path, root), f'{path}: not committed')
    return c, file_sha256(path), experiment_sha256(c)


def committed_record(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def gate_record_path(run: str) -> str:
    return f'{GATE_RECORDS}/{run}/record.json'


def gate_record_paths() -> list[str]:
    return [gate_record_path(r) for r in GATE_RUNS]


def fresh_dir(out: Path):
    require(not (out / 'record.json').exists(), f'{out} already holds a run')
    require(not out.exists() or not any(out.iterdir()),
            f'{out} holds files of an interrupted run: report it (nothing is overwritten)')


# --- the deterministic settings -----------------------------------------------------------------------------------

def numeric_settings(c: dict, device: str) -> dict:
    from hybrid.h7r_d1 import numeric_settings as base
    return {**base(device), 'environment': {k: os.environ.get(k) for k in sorted(c['determinism']['environment'])}}


@contextmanager
def deterministic(c: dict, device: str):
    """The contract's deterministic settings for one run: the environment must already hold them (cuBLAS reads it
    when it starts); the tool sets the three torch switches, checks every contract value, and restores the switches
    afterwards. Yields the numeric settings (the full ``h7r_d1.numeric_settings`` and the environment)."""
    import torch

    det = c['determinism']
    for k, v in det['environment'].items():
        require(os.environ.get(k) == v, f'{k} must be {v!r} before the process starts (it is {os.environ.get(k)!r})')
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
              torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        now = numeric_settings(c, device)
        wrong = {k: now.get(k) for k, v in det['settings'].items() if now.get(k) != v}
        require(not wrong, f'numeric settings are not the contract\'s: {wrong}')
        yield now
        require(numeric_settings(c, device) == now, 'the numeric settings changed during the run')
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[1], before[2]


# --- the one entry point of every training run ---------------------------------------------------------------------

def execute(c: dict, kind: str, seed: int, device: str, checked: dict, root=ROOT, log=print,
            expect_settings: dict | None = None) -> dict:
    """One training run on the checked inputs (``h7r_p1.check_inputs``), for the gate and the main runs alike.
    ``kind``: 'old' (``h7c_train.train_policy``, the c2 trainer), 'C' or 'T' (``h7r_p1_train.train_policy`` as that
    arm: the P1 replay settings for T, none for C; the RenjuNet val diagnostic on for both, as
    ``h7r_p1.run_training`` runs them). ``expect_settings``: the numeric settings the run must start under."""
    from contextlib import ExitStack

    import torch

    from hybrid import h7c_train as tr, h7r_p1_train as pt
    from hybrid.h3_train import H3Config, evaluate
    from h7c_train_policy import load_policy_net
    import h7r_p1_diag as dg

    require(kind in ('old', *ARMS), f'kind {kind}')
    started = time.time()
    with deterministic(c, device) as settings, ExitStack() as stack:
        require(expect_settings is None or settings == expect_settings,
                f'numeric settings differ from the gate\'s: {settings} != {expect_settings}')
        cfg = p1.policy_config(c, seed, device)
        h3 = Path(root) / p1.PATHS['h3']
        parent, _ = load_policy_net(h3, device)
        teacher, _ = load_policy_net(h3, device)
        initial = {'parent': dg.state_sha256(parent.state_dict()), 'teacher': dg.state_sha256(teacher.state_dict())}
        rows = checked['rows']
        train, val = tr.policy_tensors(rows['train']), tr.policy_tensors(rows.get('val', []))
        guard = checked['cache']['val']
        h3_cfg = H3Config(seed=cfg.seed)
        ge = lambda m, d: evaluate(m, d, h3_cfg, torch.device(device))  # noqa: E731
        fp = dg.Fingerprints(train)
        fp.install(stack)
        if kind == 'old':
            result = {**tr.train_policy(cfg, parent, teacher, train, val, guard, ge, log=log), 'p1': None}
        else:
            replay = p1.replay_of(c, kind)
            result = pt.train_policy(cfg, parent, teacher, train, val, guard, ge, replay=replay,
                                     replay_data=checked['cache']['train'] if replay else None, diag=guard, log=log)
        stack.close()  # the fingerprint wrappers end with the training
        out = {'result': result, 'cfg': cfg, 'parent': parent, 'settings': settings, 'logits': None,
               'summary': {'initial_state_sha256': initial, 'fingerprints': fp.summary(),
                           'selected_state_sha256': None, 'fixed_logits': None}}
        if result['state'] is not None:
            from hybrid.h3_cache import tensor_bytes
            parent.load_state_dict(result['state'])
            logits = dg.fixed_logits(parent, checked, device)
            out['logits'] = logits
            out['summary']['selected_state_sha256'] = dg.state_sha256(result['state'])
            out['summary']['fixed_logits'] = {'positions': int(logits.shape[0]),
                                              'sha256': hashlib.sha256(tensor_bytes(logits)).hexdigest()}
    out['inputs'] = {'policy_rows': p1.PATHS['cycle_rows'], 'policy_rows_sha256': c['inputs']['cycle_rows']['sha256'],
                     'parent_checkpoint_sha256': checked['h3']['policy_checkpoint_sha256'],
                     'parent_metadata_sha256': checked['h3']['policy_metadata_sha256'],
                     'teacher_checkpoint_sha256': checked['h3']['policy_checkpoint_sha256'],
                     'h3_cache': p1.PATHS['cache'],
                     'cache_tensor_sha256': {s: c['inputs']['renjunet_cache']['expected'][s]['tensor_sha256']
                                             for s in ('train', 'val')},
                     'source': checked['source']}
    out['seconds'] = round(time.time() - started, 1)
    return out


def write_temporary(out: Path, files: dict, guard: Guard):
    """Writes ``files`` (name -> writer) as temporaries, checks the guard, then gives them their names; on any
    failure this run's temporaries are removed and nothing is published."""
    temporary = {name: out / f'{name}.tmp' for name in files}
    try:
        for name, write in files.items():
            write(temporary[name])
        guard.check()
    except BaseException:
        for f in temporary.values():
            f.unlink(missing_ok=True)
        raise
    for name, f in temporary.items():
        f.replace(out / name)


# --- prepare ------------------------------------------------------------------------------------------------------

def prepare(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    guard = Guard(root, contract_path(args.contract, root), c, p1.input_files())
    checked = p1.check_inputs(c, root)
    inp = c['inputs']
    record = {'format': 'h7r-p1a1-prepare-v1', **stamp(sha, eid, guard.check()), 'passed': True,
              'checks': {'cycle_rows': {'sha256': inp['cycle_rows']['sha256'], 'train': len(checked['rows']['train']),
                                        'val': len(checked['rows'].get('val', []))},
                         'h3': checked['h3'], 'c5_value': checked['c5_value'],
                         'cache': {s: {'states': int(checked['cache'][s]['target'].shape[0]),
                                       'tensor_sha256': inp['renjunet_cache']['expected'][s]['tensor_sha256']}
                                   for s in checked['cache']},
                         'cache_games_disjoint': True, 'evidence': sorted(c['evidence']),
                         'input_files_sha256': guard.start['files']}}
    publish(Path(root) / D / 'prepare.json', record)
    return record


# --- the gate -----------------------------------------------------------------------------------------------------

def gate_run(args, root=ROOT, log=print) -> dict:
    import torch
    from hybrid import h7c_train as tr

    c, sha, eid = load_contract(args.contract, root, 'gate')
    require(args.run in GATE_RUNS, f'run {args.run}: not a gate run')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    out = Path(root) / RUNS / 'gate' / args.run
    fresh_dir(out)
    guard = Guard(root, contract_path(args.contract, root), c, [*p1.input_files(), f'{D}/prepare.json'])
    checked = p1.check_inputs(c, root)
    kind, seed = GATE_RUNS[args.run], c['stages']['gate']['seed']
    run = execute(c, kind, seed, args.device, checked, root, log)
    result = run['result']
    record = {'format': 'h7r-p1a1-gate-run-v1', **stamp(sha, eid, guard.start['git']), 'run': args.run, 'kind': kind,
              'seed': seed, 'settings': run['settings'], 'sources': guard.start['sources'],
              'config': tr.config_dict(run['cfg']), 'inputs': run['inputs'], **run['summary'],
              'record': result['record'], 'p1': result['p1'], 'seconds': run['seconds']}
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    if result['state'] is not None:  # kept under runs/ only (never committed)
        files['state.pt'] = lambda f: torch.save({k: v.detach().cpu() for k, v in result['state'].items()}, f)
        files['logits.pt'] = lambda f: torch.save(run['logits'], f)
    write_temporary(out, files, guard)
    publish(out / 'record.json', record)
    return record


def differences(a, b, prefix='') -> list[str]:
    """Where two JSON values differ (dotted paths, two levels deep); empty when they are exactly equal."""
    if a == b:
        return []
    if isinstance(a, dict) and isinstance(b, dict) and prefix.count('.') < 2:
        return [p for k in sorted(set(a) | set(b)) for p in differences(a.get(k), b.get(k), f'{prefix}{k}.')] or \
            [prefix.rstrip('.')]
    return [prefix.rstrip('.') or '(value)']


def compared(rec: dict, with_p1: bool = True) -> dict:
    """What the gate compares exactly: the fingerprints (steps, batch, D4, learning rate, per-step loss), the whole
    trainer record (every evaluation's values, eligibility, selection, guards), the initial and selected state
    hashes, the fixed-position log-probability hash and, unless left out, the P1 records (replay and diagnostic)."""
    keys = ('fingerprints', 'record', 'initial_state_sha256', 'selected_state_sha256', 'fixed_logits')
    return {**{k: rec.get(k) for k in keys}, **({'p1': rec.get('p1')} if with_p1 else {})}


def strict_checks(c: dict, run: str, rec: dict, c2: dict) -> list[str]:
    """The checks that never depend on floating-point values: what ran, its inputs, the model identities, the
    training settings, the numeric settings and the step structure."""
    bad = []
    exp = c['stages']['gate']['expected']
    kind = GATE_RUNS[run]
    if (rec.get('format'), rec.get('run'), rec.get('kind'), rec.get('seed')) != (
            'h7r-p1a1-gate-run-v1', run, kind, c['stages']['gate']['seed']):
        bad.append('another run (format, run, kind or seed)')
        return bad
    inp, want = rec['inputs'], c['inputs']
    if inp['policy_rows_sha256'] != want['cycle_rows']['sha256']:
        bad.append('cycle rows')
    ids = {'policy_checkpoint_sha256': inp['parent_checkpoint_sha256'],
           'policy_metadata_sha256': inp['parent_metadata_sha256']}
    if ids != want['parent_and_teacher'] or inp['teacher_checkpoint_sha256'] != ids['policy_checkpoint_sha256']:
        bad.append('parent / teacher identity')
    if inp['cache_tensor_sha256'] != {s: want['renjunet_cache']['expected'][s]['tensor_sha256']
                                      for s in ('train', 'val')}:
        bad.append('RenjuNet cache tensors')
    for k in p1.REPRODUCED_INPUTS:
        if inp.get(k) != c2['inputs'][k]:
            bad.append(f'input {k} differs from the c2 record')
    if {k: v for k, v in rec['config'].items() if k != 'device'} != \
            {k: v for k, v in c2['config'].items() if k != 'device'}:
        bad.append('training settings differ from the c2 record')
    if rec['config'].get('device') != c['training']['device']:
        bad.append('device')
    det = c['determinism']
    if any(rec['settings'].get(k) != v for k, v in det['settings'].items()) or \
            rec['settings'].get('environment') != {k: det['environment'][k] for k in sorted(det['environment'])}:
        bad.append('numeric settings are not the contract\'s deterministic ones')
    r = rec['record']
    for k in STRUCTURE:
        if r[k] != exp[k]:
            bad.append(f'{k}: {r[k]} (expected {exp[k]})')
    if [e['step'] for e in r['evaluations']] != exp['evaluation_steps']:
        bad.append('evaluation steps')
    for k, n in (('train_rows', want['cycle_rows']['train_rows']), ('val_rows', want['cycle_rows']['val_rows'])):
        if r[k] != n:
            bad.append(f'{k}: {r[k]} (expected {n})')
    if rec['fingerprints']['steps'] != exp['total_steps']:
        bad.append('fingerprinted steps')
    p = rec['p1']
    if kind == 'old':
        if p is not None:
            bad.append('E1 is not the c2 trainer')
    else:
        from dataclasses import asdict
        arm = p1.replay_of(c, kind)  # what the trainer records for this arm's replay settings
        replay = None if arm is None else {**asdict(arm), 'generator_seed': rec['seed'] + arm.seed_offset}
        if p is None or p.get('replay') != replay:
            bad.append(f'the replay settings are not arm {kind}\'s')
        if p is None or p.get('parent_diag') is None or \
                any(e.get('renjunet_val_kl') is None for e in p.get('evaluations', [])):
            bad.append('the diagnostic is not on')
    return bad


def c2_reference(rec: dict, c2: dict) -> dict:
    """The run against the past c2 record: recorded only, never judged."""
    import h7r_p1_diag as dg
    r = rec['record']
    return {'evaluations': dg.eval_diffs(r['evaluations'], c2['evaluations']),
            'eligibility_equal': [e['eligible'] for e in r['evaluations']] == [e['eligible'] for e in c2['evaluations']],
            'selected_step': (r['best'] or {}).get('step'), 'c2_selected_step': (c2['best'] or {}).get('step'),
            'teacher_guard_equal': r['teacher_guard'] == c2['teacher_guard'],
            'parent_val_equal': r['parent_val'] == c2['parent_val'], 'verdict': r['verdict'],
            'c2_verdict': c2['verdict']}


def judge_gate(c: dict, eid: str, root=ROOT) -> dict:
    """The gate from the committed gate records alone (module docstring)."""
    recs = {run: committed_record(root, gate_record_path(run), f'gate run {run}', eid) for run in GATE_RUNS}
    c2 = json.loads((Path(root) / p1.C2_RECORD).read_text(encoding='utf-8'))
    exp = c['stages']['gate']['expected']
    c2_shape = {'total_steps': c2['total_steps'], 'steps_per_epoch': c2['steps_per_epoch'],
                'warmup_steps': c2['warmup_steps'], 'evaluation_steps': [e['step'] for e in c2['evaluations']]}
    strict = {run: strict_checks(c, run, rec, c2) for run, rec in recs.items()}
    if c2_shape != {k: exp[k] for k in c2_shape}:
        strict['c2'] = ['the c2 record does not show the contract step structure']
    first = recs['E1']
    same = {'settings': all(r['settings'] == first['settings'] for r in recs.values()),
            'sources': all(r['sources'] == first['sources'] for r in recs.values()),
            'initial_state': all(r['initial_state_sha256'] == first['initial_state_sha256'] for r in recs.values())}
    g1 = {f'{a}={b}': differences(compared(recs[a]), compared(recs[b])) for a, b in G1}
    g2 = {f'{G2[0]}={G2[1]}': differences(compared(recs[G2[0]], False), compared(recs[G2[1]], False))}
    passed = not any(strict.values()) and all(same.values()) and not any(g1.values()) and not any(g2.values())
    return {'passed': passed, 'strict': strict, 'same_across_runs': same, 'G1_determinism': g1,
            'G2_equivalence': g2, 'settings': first['settings'],
            'gate_records_sha256': {run: file_sha256(Path(root) / gate_record_path(run)) for run in GATE_RUNS},
            'c2_record_sha256': file_sha256(Path(root) / p1.C2_RECORD),
            'c2_reference': {run: c2_reference(rec, c2) for run, rec in recs.items()},
            'c2_reference_note': 'recorded only: the past c2 values, eligibility and selected step are not judged',
            'seconds': {run: rec['seconds'] for run, rec in recs.items()}}


def gate(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'gate')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    guard = Guard(root, contract_path(args.contract, root), c,
                  [p1.C2_RECORD, f'{D}/prepare.json', *gate_record_paths()])
    body = judge_gate(c, eid, root)
    record = {'format': 'h7r-p1a1-gate-v1', **stamp(sha, eid, guard.check()), **body}
    publish(Path(root) / D / 'gate.json', record)
    return record


def verified_gate(c: dict, eid: str, root=ROOT) -> dict:
    """The committed gate record, judged again from the committed gate records (never its ``passed`` alone)."""
    rec = committed_record(root, f'{D}/gate.json', 'gate record', eid)
    again = judge_gate(c, eid, root)
    require({k: rec.get(k) for k in again} == again, 'the gate record is not what the committed gate records give')
    require(again['passed'] is True, f"the gate did not pass: {again['strict']} {again['G1_determinism']} "
                                     f"{again['G2_equivalence']} {again['same_across_runs']}")
    return rec


# --- main runs ----------------------------------------------------------------------------------------------------

def train(args, root=ROOT, log=print) -> dict:
    import torch
    from hybrid import h7c_train as tr
    from model.checkpoint import save_checkpoint

    c, sha, eid = load_contract(args.contract, root, 'train')
    require(args.arm in ARMS and args.seed in c['seeds']['training'], 'arm / seed not in the contract')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    g = verified_gate(c, eid, root)
    out = Path(root) / RUNS / f'{args.arm}-{args.seed}'
    fresh_dir(out)
    guard = Guard(root, contract_path(args.contract, root), c,
                  [*p1.input_files(), p1.C2_RECORD, f'{D}/prepare.json', f'{D}/gate.json', *gate_record_paths()])
    checked = p1.check_inputs(c, root)
    run = execute(c, args.arm, args.seed, args.device, checked, root, log, expect_settings=g['settings'])
    result, cfg, parent, inputs = run['result'], run['cfg'], run['parent'], run['inputs']
    exp = c['stages']['gate']['expected']
    r = result['record']
    require([r[k] for k in STRUCTURE] == [exp[k] for k in STRUCTURE]
            and [e['step'] for e in r['evaluations']] == exp['evaluation_steps'],
            'the step structure is not the contract\'s; nothing is published')
    git = guard.start['git']
    record = {'format': 'h7r-p1a1-train-v1', **stamp(sha, eid, git), 'arm': args.arm, 'seed': args.seed,
              'config': tr.config_dict(cfg), 'inputs': inputs, 'settings': run['settings'],
              'gate_sha256': file_sha256(Path(root) / D / 'gate.json'), **run['summary'], **r, 'p1': result['p1'],
              'seconds': run['seconds']}
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    if result['state'] is not None:  # ``parent`` already holds the selected state (``execute``)
        parent_meta = json.loads((Path(root) / p1.h3_meta_path()).read_text(encoding='utf-8'))
        meta = {'stage': tr.POLICY_STAGE, 'track': 'B', 'cycle': 2, 'policy_trained': True, 'value_trained': False,
                'config': {'channels': parent_meta['config']['channels'], 'blocks': parent_meta['config']['blocks'],
                           **tr.config_dict(cfg)},
                'parent_checkpoint_sha256': inputs['parent_checkpoint_sha256'],
                'teacher_checkpoint_sha256': inputs['teacher_checkpoint_sha256'],
                'policy_rows_sha256': inputs['policy_rows_sha256'], 'source_models': inputs['source']['models'],
                'best': r['best'], 'teacher_guard_top1': r['teacher_guard']['top1'],
                'p1': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'arm': args.arm, 'seed': args.seed,
                       'replay': result['p1']['replay']},
                'git': git}
        files['best.pt'] = lambda f: save_checkpoint(f, parent)
        files['best.json'] = lambda f: f.write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        files['logits.pt'] = lambda f: torch.save(run['logits'], f)
    write_temporary(out, files, guard)  # after every temporary is written, right before anything takes its name
    if result['state'] is not None:
        record['candidate'] = {'checkpoint_sha256': file_sha256(out / 'best.pt'),
                               'metadata_sha256': file_sha256(out / 'best.json')}
    publish(out / 'record.json', record)
    return record


# --- link and Arena contract --------------------------------------------------------------------------------------

def run_record_path(arm, seed) -> str:
    return f'{SEED_RECORDS}/{arm}-{seed}/record.json'


def seed_record_paths(c) -> list[str]:
    return [run_record_path(arm, seed) for seed in c['seeds']['training'] for arm in ARMS]


def linked_models(c: dict, eid: str, root=ROOT) -> dict:
    """The candidates of the committed seed records (None for a run without one)."""
    models = {}
    for seed in c['seeds']['training']:
        for arm in ARMS:
            rec = committed_record(root, run_record_path(arm, seed), f'{arm} {seed} run record', eid)
            require(rec['arm'] == arm and rec['seed'] == seed and rec['format'] == 'h7r-p1a1-train-v1',
                    f'{arm} {seed}: another run')
            require((rec['verdict'] == 'CANDIDATE') == ('candidate' in rec), f'{arm} {seed}: verdict / candidate')
            models[f'{arm}{seed}'] = None if rec['verdict'] != 'CANDIDATE' else {
                'policy': f'{RUNS}/{arm}-{seed}/best.pt', 'value': p1.PATHS['c5_value'],
                'identity': {'policy_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                             'policy_metadata_sha256': rec['candidate']['metadata_sha256'],
                             **c['inputs']['value_c5']}}
    return models


def link(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'train')
    guard = Guard(root, contract_path(args.contract, root), c, seed_record_paths(c))
    models = linked_models(c, eid, root)
    s = c['seeds']['arena_candidate_training_seed']
    require(models[f'T{s}'] is not None and models[f'C{s}'] is not None,
            f'the candidate seed {s}: T or C has no candidate (no other seed is substituted)')
    record = {'format': 'h7r-p1a1-link-v1', **stamp(sha, eid, guard.check()), 'candidate_seed': s, 'models': models}
    publish(Path(root) / D / 'link.json', record)
    return record


def verified_link(c: dict, eid: str, root=ROOT) -> tuple[dict, str]:
    rec = committed_record(root, f'{D}/link.json', 'link record', eid)
    require(rec.get('candidate_seed') == c['seeds']['arena_candidate_training_seed']
            and rec.get('models') == linked_models(c, eid, root), 'the link record is not what the seed records give')
    return rec, file_sha256(Path(root) / D / 'link.json')


def arena_contract_body(c: dict, eid: str, link_rec: dict, link_sha: str, root=ROOT) -> dict:
    s = c['seeds']['arena_candidate_training_seed']
    champion = {'policy': p1.PATHS['h3'], 'value': p1.PATHS['c5_value'],
                'identity': {**c['inputs']['parent_and_teacher'], **c['inputs']['value_c5']}}
    models = {'champion': champion, 'T': link_rec['models'][f'T{s}'], 'C': link_rec['models'][f'C{s}']}
    suite = {x['path']: x for x in c['arena']['suites']}
    arms = []
    for spec in c['arena']['arms']:
        arm_label = spec['name'].split('_')[1]
        arms.append({'name': spec['name'], 'tested': {'model': arm_label, 'budget': 1},
                     'opponent': {'model': 'champion', 'budget': 1}, 'pairs': spec['pairs'], 'seed': spec['seed'],
                     'pairing': spec['pairing'], 'suite': {'path': spec['suite'], 'sha256': suite[spec['suite']]['sha256']}})
    o2 = {'path': 'docs/h7r/suites/O2.json', 'sha256': file_sha256(Path(root) / 'docs/h7r/suites/O2.json')}
    for spec in c['arena']['auxiliary']['arms']:
        name = spec['name'].split('_')[1]  # e.g. T20261111
        if link_rec['models'].get(name) is None:
            continue  # no candidate for this auxiliary run: its arm is left out (auxiliary arms are never read)
        models[name] = link_rec['models'][name]
        arms.append({'name': spec['name'], 'tested': {'model': name, 'budget': 1},
                     'opponent': {'model': 'champion', 'budget': 1}, 'pairs': spec['pairs'], 'seed': spec['seed'],
                     'pairing': spec['pairing'], 'suite': o2})
    st = c['reading']['statistics']
    return {'format': 'h7r-eval-contract-v1', 'contract_id': ARENA_ID, 'purpose': 'pilot', 'status': 'fixed',
            'meaning': f'the Arena of {CONTRACT_ID} (experiment {eid[:12]}): development reading by '
                       'scripts/h7r_p1a1.py judge; never a promotion or champion change',
            'models': models, 'arms': arms,
            'stats': {'alpha_one_sided': st['alpha_one_sided'], 'bootstrap': st['bootstrap']},
            'p1': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'link_sha256': link_sha}}


def arena_contract(args, root=ROOT) -> dict:
    from hybrid import h7r_eval as ev

    c, sha, eid = load_contract(args.contract, root, 'arena')
    guard = Guard(root, contract_path(args.contract, root), c, [f'{D}/link.json', *seed_record_paths(c)])
    link_rec, link_sha = verified_link(c, eid, root)
    body = arena_contract_body(c, eid, link_rec, link_sha, root)
    path = Path(root) / ARENA_CONTRACT
    ev.check_contract(body, path, root, check_files=False)
    guard.check()
    publish(path, body)
    ev.load_contract(path, root, check_files=False)  # its seeds are used by no other contract
    return body


# --- judge --------------------------------------------------------------------------------------------------------

def judge(args, root=ROOT) -> dict:
    from hybrid import h7r_eval as ev
    import h7r_eval as cli

    c, sha, eid = load_contract(args.contract, root, 'arena')
    apath = Path(root) / ARENA_CONTRACT
    require(committed(apath, root), 'the Arena contract is not committed')
    arena, asha = ev.load_contract(apath, root, check_files=False)
    link_rec, link_sha = verified_link(c, eid, root)
    expected = json.loads(json.dumps(arena_contract_body(c, eid, link_rec, link_sha, root)))
    require(arena == expected, 'the Arena contract is not the one the A1 contract and the verified link give '
                               '(models, arms, suites, pairs, seeds, pairings, budgets or statistics differ)')
    arm_files = {a['name']: Path(args.arms_dir) / f"{a['name']}.json" for a in arena['arms']}
    guard = Guard(root, contract_path(args.contract, root), c,
                  [f'{D}/link.json', *seed_record_paths(c), ARENA_CONTRACT])
    arm_sha = {name: file_sha256(path) for name, path in arm_files.items()}
    arms = {}
    for a in arena['arms']:
        loaded = cli.load_arm(arm_files[a['name']], arena, asha, Path(root))
        require(loaded['table']['valid'], f"{a['name']}: invalid pairs")
        require(len(loaded['table']['pairs']) == a['pairs'], f"{a['name']}: {len(loaded['table']['pairs'])} of "
                                                             f"{a['pairs']} pairs (the sample is never reduced)")
        arms[a['name']] = loaded
    tables = {k: v['table'] for k, v in arms.items()}
    strata = p1.paired_strata(tables)
    st = c['reading']['statistics']
    b = st['bootstrap']
    main = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'])
    sens = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'], unit='class')
    lab = p1.labels(main)
    git = guard.check()
    require({name: file_sha256(path) for name, path in arm_files.items()} == arm_sha,
            'an arm record changed during the judge; nothing is published')
    record = {'format': 'h7r-p1a1-judge-v1', **stamp(sha, eid, git), 'arena_contract_sha256': asha,
              'arm_records_sha256': arm_sha, 'readings': lab, 'interpretation': p1.interpret(lab), 'bounds': main,
              'sensitivity_class_level': {'bounds': sens, 'readings': p1.labels(sens),
                                          'same_as_main': p1.labels(sens) == lab},
              'per_arm': {k: v['summary'] for k, v in arms.items()},
              'auxiliary_note': 'auxiliary arms (O2aux_*) are descriptive only',
              'end': None if lab['P'] == 'STRONGER' else 'P is not STRONGER: the contract ends'}
    publish(Path(root) / D / 'judge.json', record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    g = sub.add_parser('gate-run')
    g.add_argument('--run', required=True, choices=sorted(GATE_RUNS))
    g.add_argument('--device', default='cuda')
    sub.add_parser('gate')
    t = sub.add_parser('train')
    t.add_argument('--arm', required=True, choices=ARMS)
    t.add_argument('--seed', required=True, type=int)
    t.add_argument('--device', default='cuda')
    sub.add_parser('link')
    sub.add_parser('arena-contract')
    j = sub.add_parser('judge')
    j.add_argument('--arms-dir', required=True)
    args = parser.parse_args(argv)
    fn = {'prepare': prepare, 'gate-run': gate_run, 'gate': gate, 'train': train, 'link': link,
          'arena-contract': arena_contract, 'judge': judge}[args.command]
    try:
        out = fn(args)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    keys = ('passed', 'strict', 'same_across_runs', 'G1_determinism', 'G2_equivalence', 'run', 'verdict',
            'selected_state_sha256', 'seconds', 'readings', 'interpretation', 'candidate_seed')
    print(json.dumps({k: out[k] for k in keys if k in out} or {'written': True}, ensure_ascii=False, indent=1),
          flush=True)
    if args.command == 'gate' and not out['passed']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
