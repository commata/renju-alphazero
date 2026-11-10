"""R2-P1-POLICY-REPLAY-1: the policy replay control experiment (docs/h7r/r2/R2-P1-POLICY-REPLAY-1-draft.md).

    prepare         (approved.prepare) every input against its committed expectation: the c2 cycle rows, H3 and
                    c5 checkpoints and metadata, the RenjuNet H3 cache (manifest identity, train / val tensor hashes,
                    states, games, disjoint game ids), the evidence records
    reproduce       (approved.train) the new trainer with replay off on the c2 inputs and seed: the c2 record back
                    (the contract's exact steps, evaluation steps, eligibility, selected step; inputs; settings;
                    floats within 1e-6)
    train           (approved.train) one run: ``--arm C|T --seed S``; the committed reproduction is re-judged first
    link            (approved.train) the committed seed records: the candidate seed's T and C both CANDIDATE
    arena-contract  (approved.arena) the h7r_eval Arena contract from the verified link
    judge           (approved.arena) the Arena contract must be exactly the one the contract and the verified link
                    give; the arm records complete; P (T vs champion) and D (T - C), suite-stratified, 50:50, paired;
                    the class-level sensitivity; the pre-fixed interpretation

Inputs: reproduce and train check every input they consume against the contract's expectations (never only the
cache's own manifest) and train on exactly the checked objects. Invariance: every command records HEAD (clean),
the contract file, the sources and the files it reads when it starts, and checks them again right before it
publishes; a change publishes nothing. Every record carries the experiment SHA-256 (the contract without
``approved``) and the contract file SHA-256 of its run, so an approval commit never breaks the link between stages.
Records are written once; committed copies of the run records live in ``seed-records/`` (``runs/`` is git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FORMAT = 'h7r-p1-contract-v1'
CONTRACT_ID = 'R2-P1-POLICY-REPLAY-1'
D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-1'
CONTRACT = f'{D}/p1-contract.json'
SEED_RECORDS = f'{D}/seed-records'
RUNS = 'runs/h7r/R2-P1-POLICY-REPLAY-1'
ARENA_CONTRACT = 'docs/h7r/contracts/R2-P1-ARENA-1.json'
ARENA_ID = 'R2-P1-ARENA-1'
ARMS = ('C', 'T')
STAGES = ('prepare', 'train', 'arena')
PATHS = {'cycle_rows': 'runs/h7c/c2/policy_rows.jsonl.gz', 'h3': 'runs/h3_policy_64x4/best.pt',
         'c5_value': 'runs/h7c/c5/value/best.pt', 'cache': 'data/external/renjunet/h3_cache'}
CACHE_FILES = ('manifest.json', 'train.pt', 'val.pt')
C2_RECORD = 'docs/h7c/c2/policy_record.json'
FLOAT_TOL = 1e-6
SUITES = ('O0', 'O1')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def experiment_sha256(c: dict) -> str:
    return digest({k: v for k, v in c.items() if k != 'approved'})


def committed(path, root) -> bool:
    from hybrid.h7c_nmask import committed as _committed
    return _committed(Path(path), Path(root))


def contract_path(path, root) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    """The fixed, committed contract; its file SHA and experiment SHA. ``stage`` must be approved (in order)."""
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
    if stage is not None:
        require(approved[stage], f'{path}: the {stage} stage is not approved')
    if check_sources:
        for ref in c['sources']:
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: differs from the contract")
        for ref in c['evidence'].values():
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: evidence differs")
        require(committed(path, root), f'{path}: not committed')
    return c, file_sha256(path), experiment_sha256(c)


def git_state(root=ROOT) -> dict:
    run = lambda *a: subprocess.check_output(['git', *a], cwd=root, text=True).strip()  # noqa: E731
    state = {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain', '--untracked-files=no'))}
    require(not state['dirty'], 'the tracked tree is not clean')
    return state


class Guard:
    """What a command started from: HEAD (clean), the contract file, the sources and the files it reads. ``check``
    before any publication: a change publishes nothing."""

    def __init__(self, root, path, c: dict, files):
        self.root, self.path, self.sources = Path(root), Path(path), [r['path'] for r in c['sources']]
        self.files = sorted(set(files))
        self.start = self.state()

    def state(self) -> dict:
        return {'git': git_state(self.root), 'contract_file_sha256': file_sha256(self.path),
                'sources': {p: file_sha256(self.root / p) for p in self.sources},
                'files': {p: file_sha256(self.root / p) for p in self.files}}

    def check(self) -> dict:
        try:
            now = self.state()
        except ValueError as exc:  # the tree became dirty
            raise ValueError(f'changed during the run: {exc}; nothing is published')
        changed = [k for k in ('git', 'contract_file_sha256') if now[k] != self.start[k]]
        changed += [p for part in ('sources', 'files') for p in self.start[part] if now[part][p] != self.start[part][p]]
        require(not changed, f'changed during the run: {changed}; nothing is published')
        return self.start['git']


def publish(path, body):
    from hybrid.h7r_ledger import publish_json
    path = Path(path)
    require(not path.exists(), f'{path} exists: never overwritten')
    publish_json(path, body)


def committed_record(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


# --- inputs -----------------------------------------------------------------------------------------------------

def h3_meta_path() -> str:
    return Path(PATHS['h3']).with_suffix('.json').as_posix()


def c5_meta_path() -> str:
    return Path(PATHS['c5_value']).with_name('best.json').as_posix()


def input_files() -> list[str]:
    """Every file prepare, reproduce and train read (invariance is checked on all of them)."""
    return [PATHS['cycle_rows'], PATHS['h3'], h3_meta_path(), PATHS['c5_value'], c5_meta_path(),
            *(f"{PATHS['cache']}/{f}" for f in CACHE_FILES)]


def check_policy_meta(path) -> dict:
    meta = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    require(meta.get('policy_trained') is True and meta.get('value_trained') is False
            and {'channels', 'blocks'} <= set(meta.get('config', {})), f'{path}: not a policy-only checkpoint')
    return meta


def check_cache(cache_dir, expected: dict) -> dict:
    """The H3 cache against the committed expectation (h3_cache_report.json): manifest identity, the train / val
    tensor hashes (recomputed from the loaded tensors), states, games and disjoint game ids. The loaded splits,
    which are exactly what was checked, are returned."""
    from hybrid.h3_cache import load_cache, tensor_sha256

    manifest = json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('format', 'h2_output_sha256', 'rules_sha256', 'encoder_version', 'action_index_version'):
        require(manifest.get(key) == expected[key], f'cache {key}: {manifest.get(key)} != {expected[key]}')
    out, ids = {}, {}
    for split in ('train', 'val'):
        want = expected[split]
        require(manifest['splits'][split]['sha256'] == want['tensor_sha256'], f'cache {split}: manifest hash differs')
        data, _ = load_cache(Path(cache_dir), split)
        require(tensor_sha256(data) == want['tensor_sha256'], f'cache {split}: the tensors are not the contract\'s')
        ids[split] = {int(x) for x in data['game_id'].tolist()}
        require(int(data['target'].shape[0]) == want['states'] and len(ids[split]) == want['games'],
                f'cache {split}: {data["target"].shape[0]} states / {len(ids[split])} games')
        out[split] = data
    require(not ids['train'] & ids['val'], 'cache: a game in both train and val')
    return out


def check_inputs(c: dict, root=ROOT) -> dict:
    """Every input against the contract's expectation; the loaded objects (what training then consumes)."""
    from analysis import tb_v3
    from hybrid import h7c_train as tr

    root = Path(root)
    inp = c['inputs']
    rows_path = root / PATHS['cycle_rows']
    require(file_sha256(rows_path) == inp['cycle_rows']['sha256'], 'the c2 cycle rows differ')
    rows = tr.load_policy_rows(rows_path)
    require((len(rows['train']), len(rows.get('val', []))) == (inp['cycle_rows']['train_rows'],
                                                               inp['cycle_rows']['val_rows']), 'c2 row counts differ')
    source = tr.policy_rows_source(rows_path, 2)
    h3_ids = {'policy_checkpoint_sha256': file_sha256(root / PATHS['h3']),
              'policy_metadata_sha256': file_sha256(root / h3_meta_path())}
    require(h3_ids == inp['parent_and_teacher'], 'H3 checkpoint / metadata differ')
    require(h3_ids == {k: tb_v3.MODELS[k] for k in h3_ids}, 'H3 is not the frozen TB-V3 policy')
    require({k: source['models'][k] for k in h3_ids} == h3_ids, 'the c2 self-play was not played by H3')
    check_policy_meta(root / PATHS['h3'])
    v_ids = {'value_checkpoint_sha256': file_sha256(root / PATHS['c5_value']),
             'value_metadata_sha256': file_sha256(root / c5_meta_path())}
    require(v_ids == inp['value_c5'], 'c5 value checkpoint / metadata differ')
    meta = json.loads((root / c5_meta_path()).read_text(encoding='utf-8'))
    require(meta.get('stage') == 'H6-2a' and meta.get('trunk_frozen') is True
            and meta.get('parent_h3_checkpoint_sha256') == h3_ids['policy_checkpoint_sha256'], 'c5 metadata')
    cache = check_cache(root / PATHS['cache'], inp['renjunet_cache']['expected'])
    return {'rows': rows, 'source': source, 'h3': h3_ids, 'c5_value': v_ids, 'cache': cache}


def prepare(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    guard = Guard(root, contract_path(args.contract, root), c, input_files())
    checked = check_inputs(c, root)
    inp = c['inputs']
    record = {'format': 'h7r-p1-prepare-v1', **stamp(sha, eid, guard.check()), 'passed': True,
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


# --- training ---------------------------------------------------------------------------------------------------

def policy_config(c: dict, seed: int, device: str):
    from hybrid import h7c_train as tr
    t = c['training']
    return tr.PolicyConfig(seed=seed, batch_size=t['batch_size'], epochs=t['epochs'], learning_rate=t['learning_rate'],
                           weight_decay=t['weight_decay'], min_lr_ratio=t['min_lr_ratio'], kl_lambda=t['kl_lambda'],
                           augment_d4=t['augment_d4'], evaluations=t['evaluations'], guard_drop=t['guard_drop'],
                           device=device)


def replay_of(c: dict, arm: str):
    from hybrid import h7r_p1_train as pt
    if arm == 'C':
        return None
    r = c['arms']['T']['replay']
    return pt.Replay(mu=r['weight_mu'], batch=r['batch'], seed_offset=pt.REPLAY_SEED_OFFSET)


def run_training(c, arm: str, seed: int, device: str, checked: dict, root=ROOT, log=print) -> tuple:
    """One run of the P1 trainer on the checked inputs (``check_inputs``): (trainer result, inputs block, config,
    the parent network)."""
    import torch

    from hybrid import h7c_train as tr, h7r_p1_train as pt
    from hybrid.h3_train import H3Config, evaluate
    from h7c_train_policy import load_policy_net

    cfg = policy_config(c, seed, device)
    h3 = Path(root) / PATHS['h3']
    parent, _ = load_policy_net(h3, device)
    teacher, _ = load_policy_net(h3, device)
    rows = checked['rows']
    train, val = tr.policy_tensors(rows['train']), tr.policy_tensors(rows.get('val', []))
    guard = checked['cache']['val']
    replay = replay_of(c, arm)
    h3_cfg = H3Config(seed=cfg.seed)
    result = pt.train_policy(cfg, parent, teacher, train, val, guard,
                             lambda m, d: evaluate(m, d, h3_cfg, torch.device(device)), replay=replay,
                             replay_data=checked['cache']['train'] if replay else None, diag=guard, log=log)
    inputs = {'policy_rows': PATHS['cycle_rows'], 'policy_rows_sha256': c['inputs']['cycle_rows']['sha256'],
              'parent_checkpoint_sha256': checked['h3']['policy_checkpoint_sha256'],
              'parent_metadata_sha256': checked['h3']['policy_metadata_sha256'],
              'teacher_checkpoint_sha256': checked['h3']['policy_checkpoint_sha256'], 'h3_cache': PATHS['cache'],
              'source': checked['source']}
    return result, inputs, cfg, parent


def close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, (str, list)):
        return a == b
    return abs(float(a) - float(b)) <= FLOAT_TOL


def reproduction_check(new: dict, c2: dict, expected: dict) -> list[str]:
    """Why the new run is not the c2 run (empty when it is): the contract's exact structure and selection, which the
    c2 record must also show; floats within 1e-6."""
    bad = []
    steps = lambda r: [e['step'] for e in r['evaluations']]  # noqa: E731
    elig = lambda r: [e['eligible'] for e in r['evaluations']]  # noqa: E731
    sel = lambda r: (r['best'] or {}).get('step')  # noqa: E731
    shown = {'total_steps': lambda r: r['total_steps'], 'steps_per_epoch': lambda r: r['steps_per_epoch'],
             'warmup_steps': lambda r: r['warmup_steps'], 'evaluation_steps': steps, 'eligibility': elig,
             'selected_step': sel}
    for k, get in shown.items():
        if get(c2) != expected[k]:
            bad.append(f'the c2 record does not show the contract {k} {expected[k]}')
    for k in ('total_steps', 'steps_per_epoch', 'warmup_steps'):
        if new[k] != expected[k]:
            bad.append(f'{k}: {new[k]} (expected {expected[k]})')
    for k in ('train_rows', 'val_rows', 'verdict'):
        if new[k] != c2[k]:
            bad.append(f'{k}: {new[k]} != {c2[k]}')
    if steps(new) != expected['evaluation_steps']:
        bad.append(f"evaluation steps {steps(new)} != {expected['evaluation_steps']}")
    if elig(new) != expected['eligibility']:
        bad.append(f"eligibility {elig(new)} != {expected['eligibility']}")
    if sel(new) != expected['selected_step']:
        bad.append(f"selected step {sel(new)} != {expected['selected_step']}")
    if not bad:
        for a, b in zip(new['evaluations'], c2['evaluations']):
            if not close({k: a[k] for k in ('val', 'guard_top1', 'guard_drop')},
                         {k: b[k] for k in ('val', 'guard_top1', 'guard_drop')}):
                bad.append(f"evaluation {a['step']}: values beyond {FLOAT_TOL}")
        for k in ('parent_guard_top1', 'parent_val', 'teacher_guard'):
            if not close(new[k], c2[k]):
                bad.append(f'{k}: beyond {FLOAT_TOL}')
    return bad


REPRODUCED_INPUTS = ('policy_rows_sha256', 'parent_checkpoint_sha256', 'parent_metadata_sha256',
                     'teacher_checkpoint_sha256')


def reproduction_mismatches(run: dict, settings: dict, inputs: dict, c2: dict, c: dict) -> list[str]:
    bad = reproduction_check(run, c2, c['stages']['reproduce']['expected'])
    if {k: v for k, v in settings.items() if k != 'device'} != {k: v for k, v in c2['config'].items() if k != 'device'}:
        bad.append('settings differ from the c2 record')
    for k in REPRODUCED_INPUTS:
        if inputs.get(k) != c2['inputs'][k]:
            bad.append(f'input {k} differs')
    return bad


def reproduce(args, root=ROOT, log=print) -> dict:
    from hybrid import h7c_train as tr

    c, sha, eid = load_contract(args.contract, root, 'train')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    guard = Guard(root, contract_path(args.contract, root), c, [*input_files(), C2_RECORD, f'{D}/prepare.json'])
    checked = check_inputs(c, root)
    c2 = json.loads((Path(root) / C2_RECORD).read_text(encoding='utf-8'))
    seed = c['seeds']['arena_candidate_training_seed']
    result, inputs, cfg, _ = run_training(c, 'C', seed, args.device, checked, root, log)
    settings = tr.config_dict(cfg)
    bad = reproduction_mismatches(result['record'], settings, inputs, c2, c)
    record = {'format': 'h7r-p1-reproduce-v1', **stamp(sha, eid, guard.check()), 'seed': seed, 'passed': not bad,
              'mismatches': bad, 'settings': settings, 'inputs': inputs, 'run': result['record'], 'p1': result['p1'],
              'c2_record_sha256': file_sha256(Path(root) / C2_RECORD)}
    publish(Path(root) / D / 'reproduce.json', record)
    return record


def verified_reproduction(c: dict, eid: str, root=ROOT) -> dict:
    """The committed reproduction record, re-judged from its content (never its ``passed`` alone): the same seed, the
    same c2 record (by SHA), no mismatch when the rules are applied again."""
    rep = committed_record(root, f'{D}/reproduce.json', 'reproduction record', eid)
    c2_path = Path(root) / C2_RECORD
    require(rep.get('c2_record_sha256') == file_sha256(c2_path), 'the reproduction was judged against another c2 record')
    require(rep.get('seed') == c['seeds']['arena_candidate_training_seed'], 'the reproduction used another seed')
    c2 = json.loads(c2_path.read_text(encoding='utf-8'))
    again = reproduction_mismatches(rep['run'], rep['settings'], rep['inputs'], c2, c)
    require(not again and rep.get('mismatches') == [] and rep.get('passed') is True,
            f'the reproduction gate did not pass: {again or rep.get("mismatches")}')
    return rep


def train(args, root=ROOT, log=print) -> dict:
    from hybrid import h7c_train as tr
    from model.checkpoint import save_checkpoint

    c, sha, eid = load_contract(args.contract, root, 'train')
    require(args.arm in ARMS and args.seed in c['seeds']['training'], 'arm / seed not in the contract')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    verified_reproduction(c, eid, root)
    out = Path(root) / RUNS / f'{args.arm}-{args.seed}'
    require(not (out / 'record.json').exists(), f'{out} already holds a run')
    require(not out.exists() or not any(out.iterdir()),
            f'{out} holds files of an interrupted run: report it (nothing is overwritten)')
    guard = Guard(root, contract_path(args.contract, root), c,
                  [*input_files(), C2_RECORD, f'{D}/prepare.json', f'{D}/reproduce.json'])
    checked = check_inputs(c, root)
    result, inputs, cfg, parent = run_training(c, args.arm, args.seed, args.device, checked, root, log)
    git = guard.check()  # before the checkpoint and the record
    record = {'format': 'h7r-p1-train-v1', **stamp(sha, eid, git), 'arm': args.arm, 'seed': args.seed,
              'config': tr.config_dict(cfg), 'inputs': inputs, **result['record'], 'p1': result['p1']}
    out.mkdir(parents=True, exist_ok=True)
    if result['state'] is not None:
        parent_meta = json.loads((Path(root) / h3_meta_path()).read_text(encoding='utf-8'))
        parent.load_state_dict(result['state'])
        save_checkpoint(out / 'best.pt.tmp', parent)
        meta = {'stage': tr.POLICY_STAGE, 'track': 'B', 'cycle': 2, 'policy_trained': True, 'value_trained': False,
                'config': {'channels': parent_meta['config']['channels'], 'blocks': parent_meta['config']['blocks'],
                           **tr.config_dict(cfg)},
                'parent_checkpoint_sha256': inputs['parent_checkpoint_sha256'],
                'teacher_checkpoint_sha256': inputs['teacher_checkpoint_sha256'],
                'policy_rows_sha256': inputs['policy_rows_sha256'], 'source_models': inputs['source']['models'],
                'best': result['record']['best'], 'teacher_guard_top1': result['record']['teacher_guard']['top1'],
                'p1': {'experiment_sha256': eid, 'arm': args.arm, 'seed': args.seed, 'replay': result['p1']['replay']},
                'git': git}
        (out / 'best.json.tmp').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        (out / 'best.pt.tmp').replace(out / 'best.pt')
        (out / 'best.json.tmp').replace(out / 'best.json')
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
            require(rec['arm'] == arm and rec['seed'] == seed and rec['format'] == 'h7r-p1-train-v1',
                    f'{arm} {seed}: another run')
            require((rec['verdict'] == 'CANDIDATE') == ('candidate' in rec), f'{arm} {seed}: verdict / candidate')
            models[f'{arm}{seed}'] = None if rec['verdict'] != 'CANDIDATE' else {
                'policy': f'{RUNS}/{arm}-{seed}/best.pt', 'value': PATHS['c5_value'],
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
    record = {'format': 'h7r-p1-link-v1', **stamp(sha, eid, guard.check()), 'candidate_seed': s, 'models': models}
    publish(Path(root) / D / 'link.json', record)
    return record


def verified_link(c: dict, eid: str, root=ROOT) -> tuple[dict, str]:
    """The committed link record, equal to what the committed seed records give; its file SHA."""
    rec = committed_record(root, f'{D}/link.json', 'link record', eid)
    require(rec.get('candidate_seed') == c['seeds']['arena_candidate_training_seed']
            and rec.get('models') == linked_models(c, eid, root), 'the link record is not what the seed records give')
    return rec, file_sha256(Path(root) / D / 'link.json')


def arena_contract_body(c: dict, eid: str, link_rec: dict, link_sha: str, root=ROOT) -> dict:
    s = c['seeds']['arena_candidate_training_seed']
    champion = {'policy': PATHS['h3'], 'value': PATHS['c5_value'],
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
                       'scripts/h7r_p1.py judge; never a promotion or champion change',
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

def stratified_bounds(strata: dict, samples: int, seed: int, alpha: float, unit: str = 'opening') -> dict:
    """``strata``: suite -> list of (T pair score, C pair score, class id) per opening (paired). Resampling within
    each suite (openings, or whole classes for ``unit='class'``); the combined statistic weights the suites
    equally. Returns the point values and one-sided bounds of P = T - 0.5 and D = T - C."""
    from hybrid.h7r_eval import _q  # the R1 tools' percentile rule

    rng = Random(seed)
    suites = sorted(strata)
    groups = {}
    for s in suites:
        if unit == 'class':
            by = {}
            for row in strata[s]:
                by.setdefault(row[2], []).append(row)
            groups[s] = list(by.values())
        else:
            groups[s] = [[row] for row in strata[s]]

    def stat(chosen):
        p = d = 0.0
        for s in suites:
            rows = [r for g in chosen[s] for r in g]
            p += (sum(r[0] for r in rows) / len(rows)) / len(suites)
            d += (sum(r[0] - r[1] for r in rows) / len(rows)) / len(suites)
        return p - 0.5, d
    point = stat(groups)
    draws = []
    for _ in range(samples):
        chosen = {s: [groups[s][rng.randrange(len(groups[s]))] for _ in range(len(groups[s]))] for s in suites}
        draws.append(stat(chosen))
    out = {}
    for i, name in enumerate(('P', 'D')):
        vals = sorted(x[i] for x in draws)
        out[name] = {'estimate': round(point[i], 6), 'lower_one_sided': round(_q(vals, alpha), 6),
                     'upper_one_sided': round(_q(vals, 1 - alpha), 6)}
    out['unit'] = unit
    out['units'] = {s: len(groups[s]) for s in suites}
    return out


def labels(bounds: dict) -> dict:
    p, d = bounds['P'], bounds['D']
    lab = lambda b, hi, lo: hi if b['lower_one_sided'] > 0 else lo if b['upper_one_sided'] < 0 \
        else 'NOT_ESTABLISHED'  # noqa: E731
    return {'P': lab(p, 'STRONGER', 'WEAKER'), 'D': lab(d, 'BETTER_THAN_CONTROL', 'WORSE_THAN_CONTROL')}


INTERPRETATION = {
    'STRONGER': 'a development candidate for the separate confirmation stage; no champion change',
    'BETTER_THAN_CONTROL': 'the policy-preserving term helped over the old method, consistent with the knowledge-loss '
                           'hypothesis; not stronger than the champion (regression mitigated only)',
    'NOT_ESTABLISHED': 'the effect of this treatment is not established; the hypothesis is not ruled out (weight, '
                       'sample or precision may miss it)',
    'WORSE_THAN_CONTROL': 'this treatment did worse than the old method; not a ruling on the hypothesis'}


def interpret(lab: dict) -> str:
    return INTERPRETATION['STRONGER'] if lab['P'] == 'STRONGER' else INTERPRETATION[lab['D']]


def paired_strata(tables: dict) -> dict:
    """suite -> [(T, C, class id)] over the openings both arms played (they must be the same)."""
    out = {}
    for s in SUITES:
        t = {p['opening_id']: p for p in tables[f'{s}_T_vs_champion']['pairs']}
        cc = {p['opening_id']: p for p in tables[f'{s}_C_vs_champion']['pairs']}
        require(set(t) == set(cc) and t, f'{s}: T and C were not played on the same openings')
        out[s] = [(t[k]['pair_score'], cc[k]['pair_score'], t[k]['class_id']) for k in sorted(t)]
    return out


def judge(args, root=ROOT) -> dict:
    from hybrid import h7r_eval as ev
    import h7r_eval as cli

    c, sha, eid = load_contract(args.contract, root, 'arena')
    apath = Path(root) / ARENA_CONTRACT
    require(committed(apath, root), 'the Arena contract is not committed')
    arena, asha = ev.load_contract(apath, root, check_files=False)
    link_rec, link_sha = verified_link(c, eid, root)
    expected = json.loads(json.dumps(arena_contract_body(c, eid, link_rec, link_sha, root)))
    require(arena == expected, 'the Arena contract is not the one the P1 contract and the verified link give '
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
    strata = paired_strata(tables)
    st = c['reading']['statistics']
    b = st['bootstrap']
    main = stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'])
    sens = stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'], unit='class')
    lab = labels(main)
    git = guard.check()
    require({name: file_sha256(path) for name, path in arm_files.items()} == arm_sha,
            'an arm record changed during the judge; nothing is published')
    record = {'format': 'h7r-p1-judge-v1', **stamp(sha, eid, git), 'arena_contract_sha256': asha,
              'arm_records_sha256': arm_sha, 'readings': lab, 'interpretation': interpret(lab), 'bounds': main,
              'sensitivity_class_level': {'bounds': sens, 'readings': labels(sens),
                                          'same_as_main': labels(sens) == lab},
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
    r = sub.add_parser('reproduce')
    r.add_argument('--device', default='cuda')
    t = sub.add_parser('train')
    t.add_argument('--arm', required=True, choices=ARMS)
    t.add_argument('--seed', required=True, type=int)
    t.add_argument('--device', default='cuda')
    sub.add_parser('link')
    sub.add_parser('arena-contract')
    j = sub.add_parser('judge')
    j.add_argument('--arms-dir', required=True)
    args = parser.parse_args(argv)
    fn = {'prepare': prepare, 'reproduce': reproduce, 'train': train, 'link': link, 'arena-contract': arena_contract,
          'judge': judge}[args.command]
    try:
        out = fn(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    keys = ('passed', 'mismatches', 'verdict', 'best', 'readings', 'interpretation', 'candidate_seed')
    print(json.dumps({k: out[k] for k in keys if k in out} or {'written': True}, ensure_ascii=False, indent=1),
          flush=True)
    if args.command == 'reproduce' and not out['passed']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
