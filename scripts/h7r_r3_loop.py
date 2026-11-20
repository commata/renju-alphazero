"""R3-POLICY-LOOP-1: when the updated policy plays the next self-play, does the strength recover or rise?
(docs/h7r/r3/R3-POLICY-LOOP-1-draft.md)

Start: H3 policy + c5 value. Up to three cycles; in cycle k the policy selected in cycle k - 1 (H3 for k = 1) plays
the self-play, is the prior of the S400 re-search and is the initial model of the training; the KL / replay teacher
and the guard stay the original H3 (mu 0.5, every training setting of R2-S400-TRAIN, 2 epochs); the value stays c5.
Each cycle ends with a development Arena against the fixed champion (H3 + c5) and a pre-fixed decision.

    prepare      (approved.prepare) every input against the contract: H3 / c5 files, the R2-S400-TRAIN contract and its
                 gate, rows and reference seed record (the gate's comparison), the RenjuNet cache, the committed
                 suites (recomputed), no record using the contract's seeds, an empty runs folder; the prepare record
    gate         (approved.gate) the new training path (initial model and teacher loaded separately) with both set to
                 H3 trains S(20261024) on the R2-S400-TRAIN rows again and must equal its committed record exactly;
                 a FAIL exits 2 and no cycle starts
    next         (approved.loop) the next step of the loop as JSON: run, record (an uncommitted result) or done
    step         --cycle K --name STEP: run the next step (it must be the one ``next`` names)
                   selfplay        ``h7c_selfplay.run``, 1,000 games, the cycle's run seed, the actor's files
                   search          S50 reproduction (the first 5 games, every eligible ply equal to the actor's own
                                   record: ``h7c_retarget.compare_to_actor``), then S400 (factor 8) on every eligible
                                   ply, both through ``h7c_retarget._execute`` with the actor's models
                                   (``check_engine``); not reproduced: S50_NOT_REPRODUCED, exit 2, the loop stops
                   rows            ``h7c_labels.build`` (the c2 label settings; game-level split) gives the S50 rows;
                                   every visit row, train and val, takes its S400 target; no S400 mass keeps the S50
                                   target (counted per split); no row is added; checked again
                   train           initial model = the actor, teacher = H3, ``h7r_p1_train.train_policy`` with the
                                   R2-S400-TRAIN settings, the gate's numeric settings; NO_CANDIDATE ends the loop
                   arena-contract  the cycle's v2 Arena contract (O1-dev-c / d, 300 pairs, the cycle's seeds)
                   arena           models checked, the two arms played (resumable), the judge and the decision
    verify       --cycle K --name STEP: a step record against its files (before it is staged, and again later)
    files        --cycle K --name STEP: the paths a step's record commit stages

The chain (section 3): every step names the policy it used by SHA-256; selfplay, search and train of cycle k must
name the candidate of train k - 1 (its checkpoint, metadata and selected state), and only after judge k - 1 decided
CONTINUE. Decisions (section 7): NO_CANDIDATE -> END_NO_CANDIDATE; lower > 0 -> END_STRONGER; upper < -2%p ->
STOP_REGRESSION; otherwise CONTINUE, END_BUDGET after cycle 3. The change against the previous cycle is described
only, paired by opening id.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_eval as cli  # noqa: E402
import h7r_p1 as p1  # noqa: E402
import h7r_p1a1 as a1  # noqa: E402
import h7r_s400_arena as sa  # noqa: E402
import h7r_s400_train as st  # noqa: E402
from h7r_p1 import Guard, committed, digest, file_sha256, require  # noqa: E402

FORMAT = 'h7r-r3-loop-contract-v1'
CONTRACT_ID = 'R3-POLICY-LOOP-1'
D = 'docs/h7r/r3/R3-POLICY-LOOP-1'
CONTRACT = f'{D}/loop-contract.json'
RUNS = 'runs/h7r/R3-POLICY-LOOP-1'
STAGES = ('prepare', 'gate', 'loop')
STEPS = ('selfplay', 'search', 'rows', 'train', 'arena-contract', 'arena')
MAX_CYCLES = 3
DELTA = 0.02
GAMES = 1000
GATE_SEED = 20261024
SELFPLAY_SEEDS = (8809, 8810, 8811)
TRAINING_SEEDS = (20261201, 20261202, 20261203)
SUITES = (('O1-dev-c', 'docs/h7r/suites/O1-dev-c.json'), ('O1-dev-d', 'docs/h7r/suites/O1-dev-d.json'))
SUITE_RECORD = 'docs/h7r/suites/O1-dev-cd-record.json'
ARENA_SEEDS = {1: (9105, 9106), 2: (9107, 9108), 3: (9109, 9110)}
PAIRS = 300
STATS = {'alpha_one_sided': 0.025,
         'bootstrap': {'samples': 20000, 'seed': 20261211, 'method': 'percentile', 'unit': 'opening pair'}}
SEARCH = {'factor': 8, 'mode': 'restored', 'workers': 6, 'reproduction_games': 5, 'reproduction_factor': 1}
LABELS = {'from_ply': 5, 'outcome_weight': 0.1, 'proof_weight': 1.0, 'prove_losses': False, 'vct1': False,
          'probe_game_min_ply': 4, 'fast': 'fast4', 'workers': 12}
SELFPLAY_WORKERS = 6
EPOCHS = 2.0
PURPOSE = 'r3-policy-loop'  # search records: never a label-step input (it accepts purpose retarget only)
SOURCES = ('scripts/h7r_r3_loop.py', 'scripts/h7r_s400_train.py', 'scripts/h7r_s400_arena.py', 'scripts/h7r_eval.py',
           'scripts/h7r_p1a1.py', 'scripts/h7r_p1.py', 'scripts/h7r_p1_diag.py', 'src/hybrid/h7r_p1_train.py',
           'src/hybrid/h7c_train.py', 'scripts/h7c_selfplay.py', 'scripts/h7c_retarget.py',
           'src/hybrid/h7c_retarget.py',
           'scripts/h7c_labels.py', 'scripts/h7r_s400_pre.py', 'scripts/h7r_dev_suites_2.py',
           'src/hybrid/h7r_eval_v2.py', 'src/hybrid/h7r_openings.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'draft', 'start', 's400_train', 'training', 'cycles',
        'seeds', 'suites', 'arena', 'rules', 'stages', 'output', 'sources', 'scope'}


# --- contract -----------------------------------------------------------------------------------------------------

def contract_path(path, root=ROOT) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def arena_id(k: int) -> str:
    return f'R3-POLICY-LOOP-ARENA-{k}'


def arena_contract_path(k: int) -> str:
    return f'docs/h7r/contracts-v2/{arena_id(k)}.json'


def contract_body(root=ROOT, sources=SOURCES) -> dict:
    """The fixed contract (every approval false) from the committed R2-S400-TRAIN contract, its records and the
    committed suites."""
    root = Path(root)
    s4, s4sha, s4eid = st.load_contract(None, root, check_sources=False)
    ref = st.run_record_path(GATE_SEED)
    sha = lambda rel: file_sha256(root / rel)  # noqa: E731
    return {
        'format': FORMAT, 'contract_id': CONTRACT_ID, 'status': 'fixed', 'approved': {s: False for s in STAGES},
        'purpose': 'R3 development experiment: from H3 + c5, update the policy only and let it play the next '
                   'self-play for at most three cycles; is the development Arena score against the fixed champion '
                   'recovered or above it? Never a promotion',
        'draft': {'path': f'{D}-draft.md'},
        'start': {'policy': {'path': p1.PATHS['h3'], **s4['inputs']['parent_and_teacher']},
                  'value': {'path': p1.PATHS['c5_value'], **s4['inputs']['value_c5']},
                  'champion': 'H3 policy + c5 value (also the fixed Arena opponent of every cycle)'},
        's400_train': {'contract': st.CONTRACT, 'contract_file_sha256': s4sha, 'experiment_sha256': s4eid,
                       'gate': f'{st.D}/gate.json', 'gate_sha256': sha(f'{st.D}/gate.json'),
                       'rows_record': f'{st.D}/rows.json', 'rows_record_sha256': sha(f'{st.D}/rows.json'),
                       'gate_reference': {'seed': GATE_SEED, 'record': ref, 'record_sha256': sha(ref)},
                       'use': 'the training settings (training, arms.T.replay, determinism) and the gate comparison'},
        'training': {'trainer': 'hybrid.h7r_p1_train.train_policy', 'settings': 'R2-S400-TRAIN training and '
                     'arms.T.replay unchanged (batch 512, lr 2e-4, wd 1e-4, min lr 0.05, kl_lambda 0.5, D4, 10 '
                     'evaluations, guard_drop 0.02, BN eval; replay RenjuNet train split, batch 512, mu 0.5)',
                     'epochs': EPOCHS, 'steps': 'round(2 x ceil(train rows / 512)), recorded per cycle',
                     'initial_model': 'the policy selected in cycle k - 1 (cycle 1: H3)',
                     'teacher': 'H3 (KL, replay and guard), every cycle', 'value': 'c5, never trained',
                     'selection': 'guard (RenjuNet val top-1 within 0.02 of H3), lowest cycle val objective; none: '
                                  'NO_CANDIDATE', 'val_targets': 'S400, as the train targets'},
        'cycles': {'max': MAX_CYCLES, 'games': GAMES, 'selfplay_workers': SELFPLAY_WORKERS, 'search': SEARCH,
                   'labels': LABELS, 'data': 'each cycle trains on its own self-play only',
                   'rows': 'every visit row (train and val) takes its S400 target (h7c_retarget.retarget_target); no '
                           'S400 mass keeps the S50 target, counted per split; no row added; the split is per game '
                           '(h7c_labels split_for, val 5%), later-split duplicates masked'},
        'seeds': {'selfplay_run': list(SELFPLAY_SEEDS), 'training': list(TRAINING_SEEDS),
                  'arena': {str(k): {name: s for (name, _), s in zip(SUITES, v)} for k, v in ARENA_SEEDS.items()},
                  'bootstrap': STATS['bootstrap']['seed'], 'gate': GATE_SEED},
        'suites': {'record': SUITE_RECORD, 'record_sha256': sha(SUITE_RECORD),
                   'specs': [{'name': n, 'path': path, 'sha256': sha(path), 'pairs': PAIRS} for n, path in SUITES],
                   'use': 'development only (already read by R2-S400-PRESERVE-1); never a confirmation'},
        'arena': {'models': 'L<k> = the cycle-k candidate + c5 value against champion = H3 + c5, S50', 'stats': STATS,
                  'arms': 'one per suite and cycle, 300 pairs', 'reading': "P_k = score - 0.5, suites 50:50, "
                  'stratified opening bootstrap; STRONGER (lower > 0) / WEAKER (upper < 0) / NOT_ESTABLISHED; the '
                  'class level reported', 'change': 'P_k - P_(k-1) paired by opening id, described only'},
        'rules': {'delta': DELTA, 'max_cycles': MAX_CYCLES,
                  'table': ['train NO_CANDIDATE: END_NO_CANDIDATE', 'search S50_NOT_REPRODUCED: '
                            'STOP_S50_NOT_REPRODUCED', 'lower > 0: END_STRONGER (a separate confirmation contract)',
                            'upper < -0.02: STOP_REGRESSION (no self-play with that model)',
                            'otherwise, k < 3: CONTINUE', 'otherwise, k = 3: END_BUDGET'],
                  'multiplicity': 'at most three STRONGER readings, each at a nominal one-sided 2.5%: at most about '
                                  '7.5% together (an upper bound, assuming each test holds its level); END_STRONGER '
                                  'only picks a candidate for an independent confirmation'},
        'stages': {'prepare': 'every input checked', 'gate': 'the new training path reproduces S(20261024) exactly',
                   'loop': 'one approval for at most three cycles under the rules; every step recorded and committed '
                           'on its own; the next self-play only after a committed CONTINUE and the candidate files '
                           'checked; an interrupted step resumes from its records'},
        'output': {'prepare': f'{D}/prepare.json', 'gate': f'{D}/gate.json', 'cycle': D + '/c{k}/<step>.json',
                   'judge': D + '/c{k}/judge.json', 'runs': RUNS + '/c{k}',
                   'arena_contract': 'docs/h7r/contracts-v2/R3-POLICY-LOOP-ARENA-{k}.json'},
        'sources': [{'path': rel, 'sha256': file_sha256(root / rel)} for rel in sources],
        'scope': 'development only: no champion change, confirmation, promotion or formal R4'}


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
    seeds = c['seeds']
    require(c['rules']['delta'] == DELTA and c['rules']['max_cycles'] == MAX_CYCLES and c['cycles']['max'] == MAX_CYCLES
            and c['cycles']['games'] == GAMES and c['cycles']['search'] == SEARCH
            and seeds['selfplay_run'] == list(SELFPLAY_SEEDS) and seeds['training'] == list(TRAINING_SEEDS)
            and seeds['arena'] == {str(k): {n: s for (n, _), s in zip(SUITES, v)} for k, v in ARENA_SEEDS.items()}
            and c['arena']['stats'] == STATS and c['training']['epochs'] == EPOCHS,
            f'{path}: the rules, seeds, budgets or statistics differ')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if file_sha256(root / r['path']) != r['sha256']]
        require(not changed, f'the checkout differs from the sources the contract was fixed for: {changed}')
    require(committed(path, root), f'{path}: not committed')
    if stage is not None:
        require(a[stage] is True, f'{path}: the {stage} stage is not approved')
    return c, file_sha256(path), experiment_sha256(c)


def s400(c: dict, root=ROOT) -> tuple[dict, str]:
    """The R2-S400-TRAIN contract, exactly the pinned file and experiment (its sources checked)."""
    s4, sha, eid = st.load_contract(None, root)
    ref = c['s400_train']
    require(sha == ref['contract_file_sha256'] and eid == ref['experiment_sha256'],
            'the R2-S400-TRAIN contract is not the pinned one')
    for path, want in ((ref['gate'], ref['gate_sha256']), (ref['rows_record'], ref['rows_record_sha256'])):
        require(file_sha256(Path(root) / path) == want, f'{path}: not the pinned R2-S400-TRAIN record')
    return s4, eid


def gate_reference(c: dict, s4eid: str, root=ROOT) -> dict:
    ref = c['s400_train']['gate_reference']
    rec = st.committed_record(root, ref['record'], f"S {ref['seed']} run record", s4eid)
    require(file_sha256(Path(root) / ref['record']) == ref['record_sha256'], f"{ref['record']}: not the pinned record")
    require(rec.get('arm') == 'S' and rec.get('seed') == GATE_SEED and rec['p1']['replay']['mu'] == 0.5,
            f"{ref['record']}: not the S run of seed {GATE_SEED}")
    return rec


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def log_field(args, prefix: str) -> dict:
    name = getattr(args, 'log_name', None)
    if name is None:
        return {}
    require(re.fullmatch(rf'{re.escape(prefix)}-[A-Za-z0-9]+\.log', name) is not None, f'--log-name {name!r}')
    return {'log_name': name}


def committed_record(root, rel, what, eid) -> dict:
    return st.committed_record(root, rel, what, eid)


def h3_files(c: dict) -> dict:
    s = c['start']['policy']
    return {'path': s['path'], 'checkpoint_sha256': s['policy_checkpoint_sha256'],
            'metadata_sha256': s['policy_metadata_sha256']}


def value_files(c: dict) -> dict:
    v = c['start']['value']
    return {'path': v['path'], 'checkpoint_sha256': v['value_checkpoint_sha256'],
            'metadata_sha256': v['value_metadata_sha256']}


def meta_of(rel: str) -> str:
    return Path(rel).with_suffix('.json').as_posix()


def value_meta(rel: str) -> str:
    return Path(rel).with_name('best.json').as_posix()


def models_of(policy: dict, value: dict) -> dict:
    """The four SHA-256 as ``analysis.tb_v3_net.model_identity`` names them."""
    return {'policy_checkpoint_sha256': policy['checkpoint_sha256'],
            'policy_metadata_sha256': policy['metadata_sha256'],
            'value_checkpoint_sha256': value['checkpoint_sha256'], 'value_metadata_sha256': value['metadata_sha256']}


def check_files(root, files: dict, what: str) -> None:
    for key, rel in (('checkpoint_sha256', files['path']), ('metadata_sha256', meta_of(files['path']))):
        path = Path(root) / rel
        require(path.is_file(), f'{what}: {rel} is missing')
        require(file_sha256(path) == files[key], f'{what}: {rel} is not the recorded file')


def check_value(root, value: dict) -> None:
    for key, rel in (('checkpoint_sha256', value['path']), ('metadata_sha256', value_meta(value['path']))):
        path = Path(root) / rel
        require(path.is_file() and file_sha256(path) == value[key], f'c5 value: {rel} is not the contract file')


def no_cublas(step: str) -> None:
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None, f'{step}: CUBLAS_WORKSPACE_CONFIG is set; the engine '
                                                               'runs (self-play, search, Arena) keep the conditions of '
                                                               'every earlier engine run (only gate and train set it)')


# --- paths --------------------------------------------------------------------------------------------------------

def cdir(k: int) -> str:
    return f'{RUNS}/c{k}'


def record_path(k: int, step: str) -> str:
    return f"{D}/c{k}/{'judge' if step == 'arena' else step}.json"


def model_dir(k: int) -> str:
    return f'{cdir(k)}/model'


def run_files(k: int, step: str) -> list[str]:
    """What a step writes under runs/ (an interrupted step leaves some of them)."""
    base = cdir(k)
    return {'selfplay': [f'{base}/selfplay.jsonl'], 'search': [f'{base}/s50-check.jsonl', f'{base}/s400.jsonl'],
            'rows': [f'{base}/policy_rows_s50.jsonl.gz', f'{base}/policy_rows_s400.jsonl.gz'],
            'train': [f'{model_dir(k)}/{n}' for n in ('best.pt', 'best.json', 'logits.pt')] +
                     [f'{model_dir(k)}/{n}.tmp' for n in ('best.pt', 'best.json', 'logits.pt')],
            'arena-contract': [arena_contract_path(k)], 'arena': []}[step]


def arena_dir(k: int) -> str:
    return f'{cdir(k)}/arena'


def partial(root, k: int, step: str) -> bool:
    root = Path(root)
    if step == 'arena':
        d = root / arena_dir(k)
        return d.is_dir() and any(d.iterdir())
    return any((root / rel).exists() for rel in run_files(k, step))


# --- prepare ------------------------------------------------------------------------------------------------------

def used_seeds(root) -> dict:
    """Where the contract's run and Arena seeds already appear as a seed in a committed docs JSON (other than this
    contract's own files): any hit refuses the prepare."""
    root = Path(root)
    want = {*SELFPLAY_SEEDS, *(s for v in ARENA_SEEDS.values() for s in v)}
    hits = {}
    for path in (root / 'docs').rglob('*.json'):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(D + '/') or rel in {arena_contract_path(k) for k in ARENA_SEEDS}:
            continue
        found = {int(x) for x in re.findall(r'"(?:seed|run_seed|base_seed)": (\d+)', path.read_text(errors='ignore'))}
        for s in sorted(found & want):
            hits.setdefault(s, []).append(rel)
    return hits


def committed_suites(c: dict, root=ROOT) -> dict:
    import h7r_dev_suites_2 as ds2
    root = Path(root)
    for rel, sha in [(c['suites']['record'], c['suites']['record_sha256']),
                     *((s['path'], s['sha256']) for s in c['suites']['specs'])]:
        require((root / rel).is_file() and committed(root / rel, root) and file_sha256(root / rel) == sha,
                f'{rel}: not the committed suite file of the contract')
    return ds2.check(None, root, log=lambda *a, **k: None)


def prepare(args, root=ROOT) -> dict:
    from analysis import tb_v3

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    out = root / c['output']['prepare']
    require(not out.exists(), f'{out} exists: never overwritten')
    s4, s4eid = s400(c, root)
    h3, value = h3_files(c), value_files(c)
    guard = Guard(root, contract_path(args.contract, root), c, [*st.input_files(s4), c['s400_train']['gate_reference']
                                                                ['record']])
    require((h3['checkpoint_sha256'], h3['metadata_sha256']) ==
            (tb_v3.MODELS['policy_checkpoint_sha256'], tb_v3.MODELS['policy_metadata_sha256']),
            'the start policy is not the frozen H3')
    check_files(root, h3, 'H3')
    check_value(root, value)
    gref = st.verified_gate(s4, s4eid, root)
    rows_rec, cand = st.verified_rows(s4, s4eid, root)
    base = gate_reference(c, s4eid, root)
    checked = p1.check_inputs(s4, root)  # H3, c5, the RenjuNet cache (and the c2 rows of the gate's origin)
    suites = committed_suites(c, root)
    hits = used_seeds(root)
    require(not hits, f'the contract seeds are already used: {hits}')
    left = [str(p.relative_to(root)) for p in (root / RUNS).rglob('*') if p.is_file() and 'logs' not in p.parts] \
        if (root / RUNS).exists() else []
    require(not left, f'{RUNS} holds files of another attempt: {left[:5]} (nothing is overwritten)')
    record = {'format': 'h7r-r3-loop-prepare-v1', **log_field(args, 'prepare'), **stamp(sha, eid, guard.check()),
              'passed': True, 's400_train': c['s400_train'], 'start': {'policy': h3, 'value': value},
              'gate_reference': {'selected_state_sha256': base['selected_state_sha256'],
                                 'initial_state_sha256': base['initial_state_sha256'],
                                 'gate_settings': gref['settings'], 'rows': len(cand),
                                 'candidate_rows_sha256': rows_rec['candidate_rows_sha256']},
              'renjunet_cache': {s: int(checked['cache'][s]['target'].shape[0]) for s in ('train', 'val')},
              'suites': suites['suites'], 'seeds_unused': True}
    p1.publish(out, record)
    return record


# --- training -----------------------------------------------------------------------------------------------------

def run_training(s4: dict, seed: int, device: str, checked: dict, init_path, teacher_path, log=print,
                 expect_settings: dict | None = None) -> dict:
    """``h7r_p1a1.execute`` (kind T) with the initial model and the teacher loaded from their own files: the same
    deterministic settings, configuration, replay, fingerprints, trainer and fixed-position log-probabilities."""
    import torch

    from hybrid import h7c_train as tr, h7r_p1_train as pt
    from hybrid.h3_train import H3Config, evaluate
    from h7c_train_policy import load_policy_net
    import h7r_p1_diag as dg

    started = time.time()
    with a1.deterministic(s4, device) as settings, ExitStack() as stack:
        require(expect_settings is None or settings == expect_settings,
                f'numeric settings differ from the gate\'s: {settings} != {expect_settings}')
        cfg = p1.policy_config(s4, seed, device)
        parent, _ = load_policy_net(Path(init_path), device)
        teacher, _ = load_policy_net(Path(teacher_path), device)
        initial = {'parent': dg.state_sha256(parent.state_dict()), 'teacher': dg.state_sha256(teacher.state_dict())}
        rows = checked['rows']
        train, val = tr.policy_tensors(rows['train']), tr.policy_tensors(rows.get('val', []))
        guard = checked['cache']['val']
        h3_cfg = H3Config(seed=cfg.seed)
        ge = lambda m, d: evaluate(m, d, h3_cfg, torch.device(device))  # noqa: E731
        fp = dg.Fingerprints(train)
        fp.install(stack)
        replay = p1.replay_of(s4, 'T')
        result = pt.train_policy(cfg, parent, teacher, train, val, guard, ge, replay=replay,
                                 replay_data=checked['cache']['train'], diag=guard, log=log)
        stack.close()
        out = {'result': result, 'cfg': cfg, 'parent': parent, 'settings': settings, 'logits': None,
               'summary': {'initial_state_sha256': initial, 'fingerprints': fp.summary(),
                           'selected_state_sha256': None, 'fixed_logits': None}}
        if result['state'] is not None:
            from hybrid.h3_cache import tensor_bytes
            import hashlib
            parent.load_state_dict(result['state'])
            logits = dg.fixed_logits(parent, checked, device)
            out['logits'] = logits
            out['summary']['selected_state_sha256'] = dg.state_sha256(result['state'])
            out['summary']['fixed_logits'] = {'positions': int(logits.shape[0]),
                                              'sha256': hashlib.sha256(tensor_bytes(logits)).hexdigest()}
    out['seconds'] = round(time.time() - started, 1)
    return out


def run_view(run: dict) -> dict:
    from hybrid import h7c_train as tr
    result = run['result']
    return st.gate_view({**run['summary'], **result['record'], 'p1': result['p1'], 'config': tr.config_dict(run['cfg']),
                         'settings': run['settings']})


def gate(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'gate')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    out = root / c['output']['gate']
    require(not out.exists(), f'{out} exists: never overwritten')
    s4, s4eid = s400(c, root)
    base = gate_reference(c, s4eid, root)
    h3 = h3_files(c)
    guard = Guard(root, contract_path(args.contract, root), c, [*st.input_files(s4), c['s400_train']['rows_record'],
                                                                c['s400_train']['gate_reference']['record']])
    settings = st.verified_gate(s4, s4eid, root)['settings']
    rows_rec, cand = st.verified_rows(s4, s4eid, root)
    checked = p1.check_inputs(s4, root)
    checked['rows'] = st.rows_by_split(cand)
    run = run_training(s4, GATE_SEED, args.device, checked, root / h3['path'], root / h3['path'], log, settings)
    mine = run_view(run)
    diffs = a1.differences(mine, st.gate_view(base))
    record = {'format': 'h7r-r3-loop-gate-v1', **log_field(args, 'gate'), **stamp(sha, eid, guard.check()),
              'seed': GATE_SEED, 'passed': not diffs, 'differences': diffs, 'settings': run['settings'],
              'reference': c['s400_train']['gate_reference'], 'run': mine,
              'initial_model': h3, 'teacher': h3, 'rows_sha256': rows_rec['candidate_rows_sha256'],
              'seconds': run['seconds'], 'note': 'a training-path check: its checkpoint is never used'}
    p1.publish(out, record)
    return record


def verified_gate(c: dict, eid: str, root=ROOT) -> dict:
    rec = committed_record(root, c['output']['gate'], 'gate record', eid)
    s4, s4eid = s400(c, root)
    again = a1.differences(rec['run'], st.gate_view(gate_reference(c, s4eid, root)))
    require(rec.get('passed') is True and rec.get('differences') == [] and not again,
            f"the training-path gate did not pass: {again or rec.get('differences')}")
    return rec


# --- the loop: state ----------------------------------------------------------------------------------------------

def outcome(step: str, rec: dict) -> str | None:
    """The end a committed step record means (None: the loop goes on)."""
    if step == 'search' and rec.get('status') != 'SEARCHED':
        return 'STOP_S50_NOT_REPRODUCED'
    if step == 'train' and rec.get('verdict') != 'CANDIDATE':
        return 'END_NO_CANDIDATE'
    if step == 'arena' and rec.get('decision') != 'CONTINUE':
        return rec.get('decision')
    return None


def state(c: dict, eid: str, root=ROOT) -> dict:
    """The next step of the loop: ``run`` (with ``partial`` when an interrupted attempt left files), ``record`` (a
    result written but not committed) or ``done`` with the decision."""
    root = Path(root)
    for k in range(1, MAX_CYCLES + 1):
        for step in STEPS:
            rel = record_path(k, step)
            if (root / rel).is_file() and committed(root / rel, root):
                end = outcome(step, committed_record(root, rel, f'c{k} {step} record', eid))
                if end is not None:
                    return {'action': 'done', 'cycle': k, 'step': step, 'decision': end}
            elif (root / rel).exists():
                return {'action': 'record', 'cycle': k, 'step': step, 'record': rel}
            else:
                return {'action': 'run', 'cycle': k, 'step': step, 'record': rel, 'partial': partial(root, k, step)}
    return {'action': 'done', 'cycle': MAX_CYCLES, 'step': 'arena', 'decision': 'END_BUDGET'}


def loop_context(args, root) -> dict:
    c, sha, eid = load_contract(getattr(args, 'contract', None), root, 'loop')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    g = verified_gate(c, eid, root)
    return {'c': c, 'sha': sha, 'eid': eid, 'gate': g}


def actor(c: dict, eid: str, k: int, root=ROOT) -> dict:
    """The policy of cycle k (section 3): H3 for k = 1, else the candidate of train k - 1 after a committed CONTINUE
    of judge k - 1; its files are the recorded ones now. ``state_sha256``: its parameters' hash (the training's
    initial state must be it)."""
    root = Path(root)
    if k == 1:
        h3 = h3_files(c)
        s4, s4eid = s400(c, root)
        check_files(root, h3, 'H3')
        return {**h3, 'state_sha256': gate_reference(c, s4eid, root)['initial_state_sha256']['parent'], 'cycle': 0}
    judge = committed_record(root, record_path(k - 1, 'arena'), f'c{k - 1} judge record', eid)
    require(judge.get('decision') == 'CONTINUE', f"cycle {k - 1} decided {judge.get('decision')}: no cycle {k}")
    rec = committed_record(root, record_path(k - 1, 'train'), f'c{k - 1} train record', eid)
    require(rec.get('verdict') == 'CANDIDATE' and rec.get('candidate'), f'cycle {k - 1} has no candidate')
    files = {'path': f'{model_dir(k - 1)}/best.pt', 'checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
             'metadata_sha256': rec['candidate']['metadata_sha256']}
    check_files(root, files, f'the cycle {k - 1} candidate')
    require(judge['models']['L']['policy_checkpoint_sha256'] == files['checkpoint_sha256'],
            f'judge {k - 1} judged another policy than the cycle {k - 1} candidate')
    return {**files, 'state_sha256': rec['selected_state_sha256'], 'cycle': k - 1}


def expect_next(ctx: dict, root, k: int, step: str) -> None:
    now = state(ctx['c'], ctx['eid'], root)
    require(now['action'] == 'run' and (now['cycle'], now['step']) == (k, step),
            f'the next step is {now}, not run cycle {k} {step}')


# --- the loop: steps ----------------------------------------------------------------------------------------------

def step_selfplay(ctx, args, root, log) -> tuple[dict, int]:
    import h7c_selfplay as sp

    c, k = ctx['c'], args.cycle
    no_cublas('selfplay')
    act, value = actor(c, ctx['eid'], k, root), value_files(c)
    check_value(root, value)
    jsonl = root / cdir(k) / 'selfplay.jsonl'
    guard = Guard(root, contract_path(args.contract, root), c, [act['path'], meta_of(act['path']), value['path'],
                                                                value_meta(value['path'])])
    ns = SimpleNamespace(cycle=k, run_seed=SELFPLAY_SEEDS[k - 1], games=GAMES, start=0, workers=SELFPLAY_WORKERS,
                         policy_checkpoint=root / act['path'], value_checkpoint=root / value['path'],
                         device=args.device, jsonl=jsonl, allow_dirty=False)
    summary = sp.run(ns, log=log)
    ident = summary['identity']
    require(ident['models'] == models_of(act, value), 'the self-play was not played by the actor of this cycle')
    require((ident['cycle'], ident['run_seed']) == (k, SELFPLAY_SEEDS[k - 1]), 'the self-play cycle / run seed')
    require(summary['summary']['games'] == GAMES, f"{summary['summary']['games']} games, the contract has {GAMES}")
    record = {'format': 'h7r-r3-loop-selfplay-v1', **log_field(args, f'c{k}-selfplay'),
              **stamp(ctx['sha'], ctx['eid'], guard.check()), 'cycle': k, 'actor': act, 'value': value,
              'run_seed': SELFPLAY_SEEDS[k - 1], 'jsonl': cdir(k) + '/selfplay.jsonl',
              'jsonl_sha256': file_sha256(jsonl),
              'identity_sha256': digest(ident), 'models': ident['models'], 'games': GAMES,
              'summary': summary['summary'], 'wall_seconds': summary['wall_seconds']}
    p1.publish(root / record_path(k, 'selfplay'), record)
    return record, 0


def load_cycle_games(ctx, root, k: int) -> tuple[list[dict], dict, dict]:
    import h7c_retarget as rt
    from hybrid import h7c_retarget as hr

    rec = committed_record(root, record_path(k, 'selfplay'), f'c{k} selfplay record', ctx['eid'])
    path = root / rec['jsonl']
    require(path.is_file() and file_sha256(path) == rec['jsonl_sha256'], f"{rec['jsonl']}: not the recorded games")
    games, ident = rt.load_games(path)
    require(hr.selfplay_identity(games) == ident and ident['models'] == rec['models'],
            f"{rec['jsonl']}: not the recorded self-play")
    return games, ident, rec


def search_identity(ctx, k: int, factor: int, selfplay: dict, jsonl_sha: str, scope: dict) -> dict:
    from hybrid import h7c_retarget as hr
    return {'format': hr.FORMAT, 'purpose': PURPOSE, 'cycle': k, 'factor': factor, 'budget': hr.budget(factor),
            'mode': SEARCH['mode'], 'selfplay_jsonl_sha256': jsonl_sha, 'selfplay': selfplay,
            'experiment_sha256': ctx['eid'], 'scope': scope}


def reproduction(games: list[dict], finished: dict) -> dict:
    """Every eligible ply of ``games`` at factor 1 against the actor's own record (``compare_to_actor``)."""
    from hybrid import h7c_retarget as hr

    counts, examples, compared = {}, [], 0
    expected = sum(len(hr.eligible_plies(g)) for g in games)
    for g in games:
        rows = {r['ply']: r for r in (finished.get(g['game_index']) or {}).get('rows', [])}
        for p in hr.eligible_plies(g):
            row = rows.get(p['ply'])
            if row is None:
                continue
            compared += 1
            for field in hr.compare_to_actor(p, row):
                counts[field] = counts.get(field, 0) + 1
                if len(examples) < 20:
                    examples.append({'game_index': g['game_index'], 'ply': p['ply'], 'field': field})
    if compared != expected:
        counts['not_compared'] = expected - compared
    return {'expected': expected, 'compared': compared, 'differing_fields': counts, 'examples': examples,
            'reproduced': not counts and compared == expected and expected > 0}


def execute_search(root, act, value, games, identity, jsonl, factor, device, log) -> dict:
    import h7c_retarget as rt
    import h7r_s400_pre as pre

    ns = SimpleNamespace(jsonl=jsonl, workers=SEARCH['workers'], device=device, policy_checkpoint=root / act['path'],
                         value_checkpoint=root / value['path'])
    finished = rt._execute(ns, games, identity, (SEARCH['mode'],), factor, log, validate=True)
    for g in games:  # resumed records too
        pre.check_rows(g, (finished.get(g['game_index']) or {}).get('rows'))
    return finished


def step_search(ctx, args, root, log) -> tuple[dict, int]:
    import h7c_retarget as rt
    from hybrid import h7c_retarget as hr

    c, k = ctx['c'], args.cycle
    no_cublas('search')
    games, ident, sp_rec = load_cycle_games(ctx, root, k)
    act, value = actor(c, ctx['eid'], k, root), value_files(c)
    require(sp_rec['actor']['checkpoint_sha256'] == act['checkpoint_sha256'], 'the self-play actor is not the actor')
    rt.check_engine(ident, root / act['path'], root / value['path'], args.device)  # the prior is the actor
    guard = Guard(root, contract_path(args.contract, root), c, [sp_rec['jsonl'], act['path'], meta_of(act['path']),
                                                                value['path'], record_path(k, 'selfplay')])
    early = sorted(games, key=lambda g: g['game_index'])[:SEARCH['reproduction_games']]
    s50 = root / cdir(k) / 's50-check.jsonl'
    ident50 = search_identity(ctx, k, SEARCH['reproduction_factor'], ident, sp_rec['jsonl_sha256'],
                              {'games': [g['game_index'] for g in early]})
    log(f'[r3-loop] c{k} S50 reproduction: {len(early)} games', flush=True)
    repro = reproduction(early, execute_search(root, act, value, early, ident50, s50, 1, args.device, log))
    base = {'format': 'h7r-r3-loop-search-v1', **log_field(args, f'c{k}-search'), 'cycle': k, 'actor': act,
            's50_reproduction': {**repro, 'records_sha256': file_sha256(s50), 'identity_sha256': digest(ident50)}}
    if not repro['reproduced']:
        record = {**base, **stamp(ctx['sha'], ctx['eid'], guard.check()), 'status': 'S50_NOT_REPRODUCED'}
        p1.publish(root / record_path(k, 'search'), record)
        return record, 2
    s400p = root / cdir(k) / 's400.jsonl'
    cov = hr.coverage(games)
    ident400 = search_identity(ctx, k, SEARCH['factor'], ident, sp_rec['jsonl_sha256'], cov)
    log(f"[r3-loop] c{k} S400: {len(games)} games, {cov['eligible']:,} positions", flush=True)
    started = time.perf_counter()
    finished = execute_search(root, act, value, games, ident400, s400p, SEARCH['factor'], args.device, log)
    rows = [r for g in games for r in finished[g['game_index']]['rows']]
    record = {**base, **stamp(ctx['sha'], ctx['eid'], guard.check()), 'status': 'SEARCHED',
              's400': {'records_sha256': file_sha256(s400p), 'identity_sha256': digest(ident400), 'rows': len(rows),
                       'coverage': cov, 'search_seconds': round(sum(r['seconds'] for r in rows), 1),
                       'wall_seconds_this_session': round(time.perf_counter() - started, 1)}}
    p1.publish(root / record_path(k, 'search'), record)
    return record, 0


def s400_index(path, games: list[dict], identity_sha: str, want_sha256: str) -> dict:
    """The S400 rows by (game index, ply): the file is ``want_sha256`` (hashed from the bytes that are parsed: no
    window between check and read); every game once, no other, each game's rows its eligible plies, checked."""
    import hashlib
    import h7r_s400_pre as pre

    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == want_sha256, f'{path}: not the recorded S400 records')
    scope = {g['game_index'] for g in games}
    by = {}
    for line in data.decode('utf-8').splitlines():
        if line.strip():
            rec = json.loads(line)
            require(digest(rec['identity']) == identity_sha, 'an S400 record of another search identity')
            gi = rec['game_index']
            require(gi in scope and gi not in by, f'S400 record of game {gi}: not a game of the cycle, or twice')
            by[gi] = rec['rows']
    missing = sorted(scope - set(by))
    require(not missing, f'S400 records missing for {len(missing)} games (first {missing[:3]})')
    for g in games:
        pre.check_rows(g, by[g['game_index']])
    return {(gi, r['ply']): r for gi, rows in by.items() for r in rows}


def replace_targets(base: list[dict], by_id: dict, s400: dict) -> tuple[list[dict], dict]:
    """Every visit row (train and val) with its S400 target; no S400 mass keeps the S50 target (counted per split)."""
    from hybrid import h7c_retarget as hr

    out, counts = [], {}
    for row in base:
        if row['kind'] == 'visits':
            g = by_id[row['game_id']]
            ply = next(p for p in g['plies'] if p['ply'] == row['ply'])
            target, _, _ = hr.retarget_target(ply, s400[(g['game_index'], row['ply'])])
            n = counts.setdefault(row['split'], {'visit_rows': 0, 's400': 0, 's50_kept_no_s400_mass': 0,
                                                 'unmasked_visit_rows': 0})
            n['visit_rows'] += 1
            n['unmasked_visit_rows'] += not row['mask']
            if target['kind'] == 'visits':
                out.append({**row, 'target': target['target'], 's400': {'target': 's400'}})
                n['s400'] += 1
            else:
                out.append({**row, 's400': {'target': 's50_kept', 'reason': 'no S400 mass'}})
                n['s50_kept_no_s400_mass'] += 1
        else:
            out.append(row)
    return out, counts


def check_replacement(base: list[dict], cand: list[dict], by_id: dict, s400: dict) -> list[str]:
    """Why ``cand`` is not ``base`` with only the visit targets replaced by the rule (empty when it is)."""
    want, _ = replace_targets(base, by_id, s400)
    if len(cand) != len(want):
        return [f'{len(cand)} rows, the rule gives {len(want)}']
    bad = []
    for i, (a, b) in enumerate(zip(want, cand)):
        if a != b:
            bad.append(f"row {i} ({a.get('game_id')} ply {a.get('ply')}): not the rule's row")
            if len(bad) >= 20:
                break
    return bad


def step_rows(ctx, args, root, log) -> tuple[dict, int]:
    import h7c_labels as lab
    from hybrid import h7c_train as tr

    c, k = ctx['c'], args.cycle
    srec = committed_record(root, record_path(k, 'search'), f'c{k} search record', ctx['eid'])
    require(srec['status'] == 'SEARCHED', f"the search ended {srec['status']}")
    games, ident, sp_rec = load_cycle_games(ctx, root, k)
    s400p = cdir(k) + '/s400.jsonl'
    base_p, cand_p = root / cdir(k) / 'policy_rows_s50.jsonl.gz', root / cdir(k) / 'policy_rows_s400.jsonl.gz'
    guard = Guard(root, contract_path(args.contract, root), c, [sp_rec['jsonl'], s400p, record_path(k, 'search')])
    s400 = s400_index(root / s400p, games, srec['s400']['identity_sha256'], srec['s400']['records_sha256'])
    labels = c['cycles']['labels']
    probes = lab.probe_positions(lab.default_probe_files(root), root)
    _, base, summary = lab.build(games, probes=probes, **{k2: v for k2, v in labels.items()})
    by_id = {lab.game_id(g): g for g in games}
    cand, counts = replace_targets(base, by_id, s400)
    bad = check_replacement(base, cand, by_id, s400)
    require(not bad, f'the S400 rows fail their check: {bad}')
    recovered = {}
    for path, rows in ((base_p, base), (cand_p, cand)):
        recovered[path.name] = keep_or_write_rows(path, rows)
    source = tr.policy_rows_source(cand_p, k)
    act = actor(c, ctx['eid'], k, root)
    require(source['models'] == models_of(act, value_files(c)), 'the rows are not of the actor\'s self-play')
    by_split = st.rows_by_split(cand)
    record = {'format': 'h7r-r3-loop-rows-v1', **log_field(args, f'c{k}-rows'), **stamp(ctx['sha'], ctx['eid'],
                                                                                         guard.check()),
              'cycle': k, 'labels': labels, 'labels_summary': summary,
              's50_rows': {'path': cdir(k) + '/policy_rows_s50.jsonl.gz', 'sha256': file_sha256(base_p),
                           'rows': len(base)},
              's400_rows': {'path': cdir(k) + '/policy_rows_s400.jsonl.gz', 'sha256': file_sha256(cand_p),
                            'rows': len(cand)},
              's400_records_sha256': srec['s400']['records_sha256'], 'replacement_by_split': counts,
              'files': recovered,
              'unmasked': {s: len(v) for s, v in by_split.items()}, 'source_models': source['models'],
              'probe_files': [{'path': Path(f).resolve().relative_to(root.resolve()).as_posix(), 'sha256': file_sha256(f)}
                              for f in lab.default_probe_files(root)], 'check': 'passed'}
    p1.publish(root / record_path(k, 'rows'), record)
    return record, 0


def keep_or_write_rows(path: Path, rows: list[dict]) -> str:
    """Write ``rows``; a file an interrupted attempt left is kept only when it holds exactly these rows (refused
    otherwise: nothing is overwritten); a leftover temporary file is never final and is removed."""
    path.with_name(path.name + '.tmp').unlink(missing_ok=True)
    if path.exists():
        found, _ = st.read_rows_hashed(path)
        require(found == rows, f'{path}: an earlier attempt left other rows than the recomputed ones: report it '
                               '(nothing is overwritten)')
        return 'kept (equal to the recomputed rows)'
    st.write_rows(path, rows)
    return 'written'


def verified_rows(ctx, root, k: int, rec: dict | None = None) -> tuple[dict, list[dict]]:
    """The committed rows record (or ``rec``, before its commit); both files are their SHA-256 and the S400 file
    passes the rule again."""
    import h7c_labels as lab

    rec = rec or committed_record(root, record_path(k, 'rows'), f'c{k} rows record', ctx['eid'])
    base, base_sha = st.read_rows_hashed(root / rec['s50_rows']['path'])
    cand, cand_sha = st.read_rows_hashed(root / rec['s400_rows']['path'])
    require(base_sha == rec['s50_rows']['sha256'] and cand_sha == rec['s400_rows']['sha256'],
            f'c{k}: the rows files are not the recorded ones')
    srec = committed_record(root, record_path(k, 'search'), f'c{k} search record', ctx['eid'])
    require(rec['s400_records_sha256'] == srec['s400']['records_sha256'], f'c{k}: the rows record names other S400 '
                                                                        'records than the search record')
    games, _, _ = load_cycle_games(ctx, root, k)
    s400 = s400_index(root / cdir(k) / 's400.jsonl', games, srec['s400']['identity_sha256'],
                      srec['s400']['records_sha256'])
    bad = check_replacement(base, cand, {lab.game_id(g): g for g in games}, s400)
    require(not bad, f'c{k}: the S400 rows fail their check: {bad}')
    return rec, cand


TRAIN_FILES = ('best.pt', 'best.json', 'logits.pt')


def steps_rule(train_rows: int, batch: int) -> int:
    return max(1, round(EPOCHS * math.ceil(train_rows / batch)))


def step_train(ctx, args, root, log) -> tuple[dict, int]:
    import torch
    from hybrid import h7c_train as tr
    from model.checkpoint import save_checkpoint

    c, k = ctx['c'], args.cycle
    s4, s4eid = s400(c, root)
    act, value, h3 = actor(c, ctx['eid'], k, root), value_files(c), h3_files(c)
    rrec, cand = verified_rows(ctx, root, k)
    out = root / model_dir(k)
    for tmp in out.glob('*.tmp') if out.is_dir() else []:  # never final: an interrupted write
        tmp.unlink()
    leftover = sorted(n for n in TRAIN_FILES if (out / n).exists())
    others = sorted(p.name for p in out.iterdir() if p.name not in TRAIN_FILES) if out.is_dir() else []
    require(not others, f'{model_dir(k)} holds other files: {others} (nothing is overwritten)')
    guard = Guard(root, contract_path(args.contract, root), c,
                  [*st.input_files(s4), rrec['s400_rows']['path'], act['path'], meta_of(act['path']), h3['path'],
                   record_path(k, 'rows'), c['output']['gate']])
    checked = p1.check_inputs(s4, root)
    checked['rows'] = st.rows_by_split(cand)
    checked['source'] = tr.policy_rows_source(root / rrec['s400_rows']['path'], k)
    require(checked['source']['models'] == models_of(act, value), 'the rows are not of the actor\'s self-play')
    require(guard.start['files'][rrec['s400_rows']['path']] == rrec['s400_rows']['sha256'],
            'the S400 rows changed between the start of the run and their check')
    seed = TRAINING_SEEDS[k - 1]
    run = run_training(s4, seed, args.device, checked, root / act['path'], root / h3['path'], log,
                       ctx['gate']['settings'])
    result, cfg, parent = run['result'], run['cfg'], run['parent']
    r = result['record']
    init = run['summary']['initial_state_sha256']
    gref = gate_reference(c, s4eid, root)
    require(init['parent'] == act['state_sha256'], f"the initial state {init['parent'][:12]} is not the cycle "
                                                   f"{k - 1} policy {act['state_sha256'][:12]}; nothing is published")
    require(init['teacher'] == gref['initial_state_sha256']['teacher'], 'the teacher is not H3; nothing is published')
    want = steps_rule(len(checked['rows']['train']), cfg.batch_size)
    require(r['total_steps'] == want and r['train_rows'] == len(checked['rows']['train']),
            f"{r['total_steps']} steps, the 2 epoch rule gives {want}; nothing is published")
    require(result['p1']['replay']['mu'] == s4['arms']['T']['replay']['weight_mu'] == 0.5, 'the replay mu')
    require((result['state'] is not None) == (r['verdict'] == 'CANDIDATE'),
            f"verdict {r['verdict']} with{'out' if result['state'] is None else ''} a selected state")
    git = guard.start['git']
    record = {'format': 'h7r-r3-loop-train-v1', **log_field(args, f'c{k}-train'), **stamp(ctx['sha'], ctx['eid'], git),
              'cycle': k, 'seed': seed, 'initial_model': act, 'teacher': h3, 'config': tr.config_dict(cfg),
              'rows_sha256': rrec['s400_rows']['sha256'], 'source_models': checked['source']['models'],
              'settings': run['settings'], 'gate_sha256': file_sha256(root / c['output']['gate']),
              'steps_rule': {'epochs': EPOCHS, 'train_rows': r['train_rows'], 'total_steps': r['total_steps']},
              **run['summary'], **r, 'p1': result['p1'], 'seconds': run['seconds']}
    files = {}
    if result['state'] is not None:
        parent_meta = json.loads((root / meta_of(h3['path'])).read_text(encoding='utf-8'))
        meta = {'stage': tr.POLICY_STAGE, 'track': 'B', 'cycle': k, 'policy_trained': True, 'value_trained': False,
                'config': {'channels': parent_meta['config']['channels'], 'blocks': parent_meta['config']['blocks'],
                           **tr.config_dict(cfg)},
                'parent_checkpoint_sha256': act['checkpoint_sha256'],
                'teacher_checkpoint_sha256': h3['checkpoint_sha256'],
                'policy_rows_sha256': rrec['s400_rows']['sha256'], 'source_models': checked['source']['models'],
                'best': r['best'], 'teacher_guard_top1': r['teacher_guard']['top1'],
                'r3_loop': {'contract_id': CONTRACT_ID, 'experiment_sha256': ctx['eid'], 'cycle': k, 'seed': seed},
                'git': git}
        files['best.pt'] = lambda f: save_checkpoint(f, parent)
        files['best.json'] = lambda f: f.write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        files['logits.pt'] = lambda f: torch.save(run['logits'], f)
    out.mkdir(parents=True, exist_ok=True)
    if leftover:  # an attempt died between its files and its record: the same run must give the same files
        require(result['state'] is not None, f'{model_dir(k)} holds {leftover}, the run has no candidate')
        again = out / 'recheck'
        shutil.rmtree(again, ignore_errors=True)
        again.mkdir()
        a1.write_temporary(again, files, guard)
        differ = [n for n in leftover if file_sha256(out / n) != file_sha256(again / n)]
        if differ:
            shutil.rmtree(again)
            raise ValueError(f'{model_dir(k)}: {differ} of an earlier attempt differ from this run: report it '
                             '(nothing is overwritten)')
        for n in TRAIN_FILES:
            if not (out / n).exists():
                (again / n).replace(out / n)
        shutil.rmtree(again)
        record['recovered_files'] = leftover
    else:
        a1.write_temporary(out, files, guard)
    if result['state'] is not None:
        record['candidate'] = {'checkpoint_sha256': file_sha256(out / 'best.pt'),
                               'metadata_sha256': file_sha256(out / 'best.json')}
    p1.publish(root / record_path(k, 'train'), record)
    return record, 0


# --- the Arena ----------------------------------------------------------------------------------------------------

def arm_name(suite: str, k: int) -> str:
    return f'{suite}_L{k}_vs_champion'


def arena_contract_body(ctx, root, k: int) -> dict:
    from hybrid import h7r_seeds as seeds

    c, eid = ctx['c'], ctx['eid']
    rec = committed_record(root, record_path(k, 'train'), f'c{k} train record', eid)
    require(rec.get('verdict') == 'CANDIDATE' and rec.get('candidate'), f'cycle {k} has no candidate')
    value = value_files(c)
    v = {'value_checkpoint_sha256': value['checkpoint_sha256'], 'value_metadata_sha256': value['metadata_sha256']}
    h3 = h3_files(c)
    models = {'champion': {'policy': h3['path'], 'value': value['path'],
                           'identity': {'policy_checkpoint_sha256': h3['checkpoint_sha256'],
                                        'policy_metadata_sha256': h3['metadata_sha256'], **v}},
              'L': {'policy': f'{model_dir(k)}/best.pt', 'value': value['path'],
                    'identity': {'policy_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                                 'policy_metadata_sha256': rec['candidate']['metadata_sha256'], **v}}}
    arms = []
    for spec, seed in zip(c['suites']['specs'], ARENA_SEEDS[k]):
        arms.append({'name': arm_name(spec['name'], k), 'tested': {'model': 'L', 'budget': 1},
                     'opponent': {'model': 'champion', 'budget': 1}, 'pairs': spec['pairs'], 'seed': seed,
                     'pairing': spec['name'], 'suite': {'path': spec['path'], 'sha256': spec['sha256']}})
    return {'format': 'h7r-eval-contract-v2', 'seed_policy': seeds.POLICY, 'contract_id': arena_id(k),
            'purpose': 'pilot',
            'status': 'fixed', 'meaning': f'the cycle {k} Arena of {CONTRACT_ID} (experiment {eid[:12]}): development '
                                          'reading; never a promotion',
            'models': models, 'arms': arms, 'stats': STATS,
            'r3_loop': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'cycle': k,
                        'train_record_sha256': file_sha256(root / record_path(k, 'train'))}}


def step_arena_contract(ctx, args, root, log) -> tuple[dict, int]:
    from hybrid import h7r_eval_v2 as ev2

    c, k = ctx['c'], args.cycle
    committed_suites(c, root)
    path = root / arena_contract_path(k)
    require(not (path.exists() and committed(path, root)), f'{path} is committed without its step record: report it')
    guard = Guard(root, contract_path(args.contract, root), c, [record_path(k, 'train'), c['suites']['record']])
    body = arena_contract_body(ctx, root, k)
    ev2.check_contract(body, path, root, check_files=False)
    ev2.check_unique(body, path, root)
    git = guard.check()
    kept = path.exists()
    if kept:  # written by an attempt that died before its record
        require(json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(body)),
                f'{path}: an earlier attempt wrote another contract: report it (nothing is overwritten)')
    else:
        p1.publish(path, body)
    record = {'format': 'h7r-r3-loop-arena-contract-v1', **log_field(args, f'c{k}-arena-contract'),
              **stamp(ctx['sha'], ctx['eid'], git), 'cycle': k, 'arena_contract': arena_contract_path(k),
              'arena_contract_sha256': file_sha256(path), 'arms': [a['name'] for a in body['arms']],
              'contract_file': 'kept (equal to the body the records give)' if kept else 'written'}
    p1.publish(root / record_path(k, 'arena-contract'), record)
    return record, 0


def load_arena(ctx, root, k: int) -> dict:
    from hybrid import h7r_eval_v2 as ev2

    rec = committed_record(root, record_path(k, 'arena-contract'), f'c{k} arena-contract record', ctx['eid'])
    apath = root / arena_contract_path(k)
    require(apath.is_file() and committed(apath, root) and file_sha256(apath) == rec['arena_contract_sha256'],
            f'{arena_contract_path(k)}: not the committed Arena contract of the record')
    arena, asha = ev2.load_contract(apath, root, check_files=False)
    require(arena == json.loads(json.dumps(arena_contract_body(ctx, root, k))),
            f'{arena_contract_path(k)}: not the one this contract and the committed records give')
    return {'arena': arena, 'asha': asha}


def play(arena: dict, asha: str, arm_name_: str, jsonl: Path, device: str, workers: int, root, log,
         threads_for_tests: bool = False) -> dict:
    """``scripts/h7r_eval.py play`` on a v2 Arena contract (the loop of ``h7r_s400_arena.play``)."""
    import h7c_arena as ar
    from hybrid import h7r_eval as ev
    from hybrid import h7r_openings as op
    from hybrid.progress import Progress, run_chunks
    from time import perf_counter

    git = cli._git(False)
    arm = cli._arm(arena, arm_name_)
    openings = op.pick(cli._suite(arm, root), arm['pairs'], arm['seed'])
    identity = cli.arm_identity(arena, asha, arm, device, git)
    sig = cli._sha_json(identity)
    tasks = cli.build_tasks(arm, openings)
    finished = ar.load_finished(jsonl, {t['key'] for t in tasks}, sig, log=log)
    cli.check_games(list(finished.values()), tasks, arm['name'], sig)
    side = lambda s: {'policy': str(root / arena['models'][arm[s]['model']]['policy']),  # noqa: E731
                      'value': str(root / arena['models'][arm[s]['model']]['value']), 'budget': arm[s]['budget']}
    sides = {'tested': side('tested'), 'opponent': side('opponent')}
    context = {'models': {'candidate': sides['tested'], 'parent': sides['opponent']}, 'sides': sides,
               'device': device, 'signature': sig, 'arm': arm['name']}
    todo = [{**t, 'context': context} for t in tasks if t['key'] not in finished]
    log(f"[r3-loop] {arena['contract_id']} {arm['name']}: {len(tasks)} games ({len(finished)} done), workers "
        f"{workers}, signature {sig[:12]}", flush=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from multiprocessing import get_context

        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if threads_for_tests else (
            lambda: ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')))
        with jsonl.open('a', encoding='utf-8') as out, pool_of() as pool:
            def keep(_, rows):
                cli.check_games(rows, tasks, arm['name'], sig)
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, cli._play_chunk, [[t] for t in todo], Progress('r3-arena', len(todo), 'games'), len,
                       unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    record = {'format': f'{cli.ARENA_FORMAT}-arm', 'identity': identity, 'signature': sig, 'openings': openings,
              'wall_seconds': round(perf_counter() - started, 1), 'workers': workers,
              'games_played_now': len(todo), 'games': games}
    record['summary'] = ev.arm_summary(ev.pair_table(games, openings), arena['stats'])
    return record


def label(b: dict) -> str:
    return 'STRONGER' if b['lower_one_sided'] > 0 else 'WEAKER' if b['upper_one_sided'] < 0 else 'NOT_ESTABLISHED'


def decide(bounds: dict, k: int) -> str:
    """Section 7 (fixed before any result)."""
    if bounds['lower_one_sided'] > 0:
        return 'END_STRONGER'
    if bounds['upper_one_sided'] < -DELTA:
        return 'STOP_REGRESSION'
    return 'CONTINUE' if k < MAX_CYCLES else 'END_BUDGET'


def arm_tables(arena: dict, asha: str, folder: Path, root) -> tuple[dict, dict, dict]:
    files = {arm['name']: folder / f"{arm['name']}.json" for arm in arena['arms']}
    sha = {name: file_sha256(path) for name, path in files.items()}
    gits = {name: (json.loads(path.read_text(encoding='utf-8')).get('identity') or {}).get('git') or {}
            for name, path in files.items()}
    commits = {g.get('commit') for g in gits.values()}
    require(len(commits) == 1 and None not in commits and all(g.get('dirty') is False for g in gits.values()),
            f'the arms were not played from one clean code commit: {gits}')
    loaded = {}
    for arm in arena['arms']:
        got = cli.load_arm(files[arm['name']], arena, asha, Path(root))
        require(got['table']['valid'], f"{arm['name']}: invalid pairs (an opponent illegal move)")
        require(len(got['table']['pairs']) == arm['pairs'], f"{arm['name']}: not the full sample")
        loaded[arm['name']] = got
    return loaded, sha, {'commit': commits.pop(), 'dirty': False}


def strata_of(arena: dict, loaded: dict) -> dict:
    return {arm['pairing']: [(p['pair_score'], 0.5, p['class_id']) for p in loaded[arm['name']]['table']['pairs']]
            for arm in arena['arms']}


def change_strata(arena: dict, loaded: dict, prev_arena: dict, prev_loaded: dict) -> dict:
    """suite -> [(cycle k pair score, cycle k - 1 pair score, class id)], paired by opening id (the arms of two
    cycles draw the same 300 openings in different orders)."""
    prev = {a['pairing']: {p['opening_id']: p for p in prev_loaded[a['name']]['table']['pairs']}
            for a in prev_arena['arms']}
    out = {}
    for arm in arena['arms']:
        mine = loaded[arm['name']]['table']['pairs']
        theirs = prev[arm['pairing']]
        require({p['opening_id'] for p in mine} == set(theirs) and len(mine) == len(theirs),
                f"{arm['pairing']}: the two cycles did not play the same openings")
        rows = []
        for p in mine:
            q = theirs[p['opening_id']]
            require(q['class_id'] == p['class_id'], f"{p['opening_id']}: another class in the previous cycle")
            rows.append((p['pair_score'], q['pair_score'], p['class_id']))
        out[arm['pairing']] = rows
    return out


def bounds_of(strata: dict, unit: str) -> dict:
    return p1.stratified_bounds(strata, STATS['bootstrap']['samples'], STATS['bootstrap']['seed'],
                                STATS['alpha_one_sided'], unit=unit)


def compute_judge(ctx, root, k: int) -> dict:
    a = load_arena(ctx, root, k)
    arena, asha = a['arena'], a['asha']
    folder = root / arena_dir(k)
    loaded, arm_sha, git = arm_tables(arena, asha, folder, root)
    strata = strata_of(arena, loaded)
    main, sens = bounds_of(strata, 'opening'), bounds_of(strata, 'class')
    p, ps = main['P'], sens['P']
    change = None
    if k > 1:
        prev = load_arena(ctx, root, k - 1)
        pj = committed_record(root, record_path(k - 1, 'arena'), f'c{k - 1} judge record', ctx['eid'])
        pl, psha, _ = arm_tables(prev['arena'], prev['asha'], root / arena_dir(k - 1), root)
        require(psha == pj['arm_records_sha256'], f'cycle {k - 1}: its arm records are not the judged ones')
        cb = bounds_of(change_strata(arena, loaded, prev['arena'], pl), 'opening')
        change = {'estimate': cb['D']['estimate'], 'lower_one_sided': cb['D']['lower_one_sided'],
                  'upper_one_sided': cb['D']['upper_one_sided'], 'paired_by': 'opening id', 'units': cb['units'],
                  'note': 'described only, never a decision'}
    require({n: file_sha256(folder / f'{n}.json') for n in arm_sha} == arm_sha,
            'an arm record changed during the judge')
    decision = decide(p, k)
    return {'format': 'h7r-r3-loop-judge-v1', 'contract_id': CONTRACT_ID, 'experiment_sha256': ctx['eid'],
            'contract_file_sha256': ctx['sha'], 'cycle': k, 'arena_contract_id': arena['contract_id'],
            'arena_contract_sha256': asha, 'arm_records_sha256': arm_sha, 'arms_commit': git['commit'],
            'models': {m: arena['models'][m]['identity'] for m in ('L', 'champion')},
            'bounds': {'P': p, 'unit': main['unit'], 'units': main['units']}, 'reading': label(p),
            'sensitivity_class_level': {'P': ps, 'reading': label(ps), 'same_as_main': label(ps) == label(p)},
            'per_suite': {arm['pairing']: loaded[arm['name']]['summary'] for arm in arena['arms']},
            'change_from_previous_cycle': change, 'decision': decision, 'rule': c_rule(k),
            'scope': 'development reading; END_STRONGER only names a candidate for an independent confirmation'}


def c_rule(k: int) -> str:
    return (f'lower > 0: END_STRONGER; upper < -{DELTA}: STOP_REGRESSION; otherwise '
            f"{'CONTINUE' if k < MAX_CYCLES else 'END_BUDGET'} (cycle {k} of {MAX_CYCLES})")


def step_arena(ctx, args, root, log) -> tuple[dict, int]:
    from hybrid.h7r_ledger import publish_json

    c, k = ctx['c'], args.cycle
    no_cublas('arena')
    a = load_arena(ctx, root, k)
    arena, asha = a['arena'], a['asha']
    sa.check_models(arena, root)
    folder = root / arena_dir(k)
    head = cli._git(False)
    for arm in arena['arms']:
        out = folder / f"{arm['name']}.json"
        if out.exists():
            got = json.loads(out.read_text(encoding='utf-8'))
            require(got.get('identity', {}).get('git') == head, f'{out} was played at another commit than HEAD '
                                                                f'{head}: report it (nothing is overwritten)')
            continue
        rec = play(arena, asha, arm['name'], folder / f"{arm['name']}.jsonl", args.device,
                   getattr(args, 'workers', 6) or 6, root, log, getattr(args, 'threads_for_tests', False))
        publish_json(out, rec)
    guard = Guard(root, contract_path(args.contract, root), c, [arena_contract_path(k), record_path(k, 'train')])
    body = compute_judge(ctx, root, k)
    git = guard.check()
    require(git.get('commit') == body['arms_commit'], f"the judge runs at {git.get('commit')}, the arms were played at "
                                                      f"{body['arms_commit']}")
    record = {**body, **log_field(args, f'c{k}-arena'), 'git': git}
    p1.publish(root / record_path(k, 'arena'), record)
    return record, 0


# --- verify -------------------------------------------------------------------------------------------------------

def read_record(root, k: int, step: str, eid: str) -> dict:
    path = Path(root) / record_path(k, step)
    require(path.is_file(), f'{record_path(k, step)}: no record')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid and rec.get('cycle') == k, f'{record_path(k, step)}: another '
                                                                           'experiment or cycle')
    return rec


def exit_code(step: str, rec: dict) -> int:
    return 2 if step == 'search' and rec.get('status') != 'SEARCHED' else 0


def verify(args, root=ROOT) -> dict:
    """A step record (committed or not) against what it names now (section 3 chain included)."""
    root = Path(root)
    ctx = loop_context(args, root)
    k, step = args.cycle, args.name
    require(step in STEPS and 1 <= k <= MAX_CYCLES, f'{k} {step}')
    rec = read_record(root, k, step, ctx['eid'])
    c = ctx['c']
    if step == 'selfplay':
        act = actor(c, ctx['eid'], k, root)
        require(rec['actor']['checkpoint_sha256'] == act['checkpoint_sha256']
                and rec['models'] == models_of(act, value_files(c)),
                'the self-play record names another actor')
        path = root / rec['jsonl']
        require(path.is_file() and file_sha256(path) == rec['jsonl_sha256'], f"{rec['jsonl']}: not the recorded games")
    elif step == 'search':
        require(file_sha256(root / cdir(k) / 's50-check.jsonl') == rec['s50_reproduction']['records_sha256'],
                'the S50 check records changed')
        if rec['status'] == 'SEARCHED':
            require(file_sha256(root / cdir(k) / 's400.jsonl') == rec['s400']['records_sha256'],
                    'the S400 records changed')
    elif step == 'rows':
        verified_rows(ctx, root, k, rec)
    elif step == 'train':
        out = root / model_dir(k)
        if rec.get('verdict') == 'CANDIDATE':
            for name, key in (('best.pt', 'checkpoint_sha256'), ('best.json', 'metadata_sha256')):
                f = out / name
                require(f.is_file(), f'{model_dir(k)}/{name} is missing')
                require(file_sha256(f) == rec['candidate'][key], f'{model_dir(k)}/{name}: its SHA-256 is not the one '
                                                                 f'the record names')
        else:
            left = [n for n in ('best.pt', 'best.json', 'logits.pt') if (out / n).exists()]
            require(not left, f'the record is {rec.get("verdict")}, yet {left} exist')
    elif step == 'arena-contract':
        path = root / arena_contract_path(k)
        require(path.is_file() and file_sha256(path) == rec['arena_contract_sha256'], 'the Arena contract changed')
        require(json.loads(path.read_text(encoding='utf-8')) ==
                json.loads(json.dumps(arena_contract_body(ctx, root, k))),
                'the Arena contract is not the one the records give')
    else:
        again = compute_judge(ctx, root, k)
        found = {kk: v for kk, v in rec.items() if kk not in ('git', 'log_name')}
        require(found == again, f'{record_path(k, step)}: not the judgment its arm records give (recomputed)')
        require((rec.get('git') or {}).get('commit') == again['arms_commit'] and rec['git'].get('dirty') is False,
                'the judge did not run at the arms commit')
    return {'cycle': k, 'step': step, 'log_name': rec.get('log_name'), 'exit_code': exit_code(step, rec),
            'stage': files_of(k, step)}


def files_of(k: int, step: str) -> list[str]:
    return [record_path(k, step), *([arena_contract_path(k)] if step == 'arena-contract' else [])]


# --- main ---------------------------------------------------------------------------------------------------------

STEP_FUNCTIONS = {'selfplay': step_selfplay, 'search': step_search, 'rows': step_rows, 'train': step_train,
                  'arena-contract': step_arena_contract, 'arena': step_arena}


def step(args, root=ROOT, log=print) -> tuple[dict, int]:
    root = Path(root)
    ctx = loop_context(args, root)
    expect_next(ctx, root, args.cycle, args.name)
    return STEP_FUNCTIONS[args.name](ctx, args, root, log)


def next_step(args, root=ROOT) -> dict:
    root = Path(root)
    ctx = loop_context(args, root)
    return state(ctx['c'], ctx['eid'], root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'gate'):
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
        s.add_argument('--log-name', default=None)
    sub.add_parser('next')
    s = sub.add_parser('step')
    s.add_argument('--cycle', type=int, required=True)
    s.add_argument('--name', required=True, choices=STEPS)
    s.add_argument('--device', default='cuda')
    s.add_argument('--workers', type=int, default=6, help='Arena workers')
    s.add_argument('--log-name', default=None)
    for name in ('verify', 'files'):
        s = sub.add_parser(name)
        s.add_argument('--cycle', type=int, required=True)
        s.add_argument('--name', required=True, choices=STEPS)
    args = parser.parse_args(argv)
    code = 0
    try:
        if args.command == 'next':
            print(json.dumps(next_step(args), separators=(',', ':')), flush=True)
            return 0
        if args.command == 'files':
            print('\n'.join(files_of(args.cycle, args.name)), flush=True)
            return 0
        if args.command == 'verify':
            print(json.dumps(verify(args), separators=(',', ':')), flush=True)
            return 0
        if args.command == 'step':
            rec, code = step(args)
        else:
            rec = {'prepare': prepare, 'gate': gate}[args.command](args)
            code = 0 if rec.get('passed') else 2
        keys = ('passed', 'differences', 'status', 'verdict', 'best', 'selected_state_sha256', 'total_steps',
                'replacement_by_split', 'reading', 'bounds', 'decision', 'change_from_previous_cycle', 'seconds',
                'wall_seconds', 'arena_contract_sha256')
        print(json.dumps({k: rec[k] for k in keys if k in rec} or {'written': True}, ensure_ascii=False, indent=1,
                         default=str), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())
