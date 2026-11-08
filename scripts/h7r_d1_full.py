"""R2-D1-HEAD-WIDTH-1, the full-scale stage: the existing value head (value_channels 1) against one widened to 4 on
the whole c8 T-train split (35,816 rows), 210 steps for both arms (the c8 config, 3.0 epochs), seeds 31001-31003,
both arms trained anew.

The earlier tools stay as they are (``scripts/h7r_d1.py``, ``scripts/h7r_d1_evaluate.py``, ``hybrid/h7r_d1.py``
pinned by the 1,000-row records, ``scripts/h7r_d1_5k.py`` by the 5,000-row ones). This entry point reads its own
contract (``h7r-d1-contract-full-v1``, ``d1-full-contract.json``) and runs the same procedure with the same
stage-independent parts, taken unchanged from those files. What it adds:

    the contract pins the 5,000-row contract, experiment id and records (the 1,000-row ones through it); the
    5,000-row train reading must be WIDTH_EFFECT (section 7.3);
    the data is the whole cycle train split: its rows, split and label hashes and its outcome / proof rows must be
    the contract's; prepare records the game count, the game-id SHA-256, the composition and the floors;
    every widened parent's state SHA-256 must be the one the 5,000-row (and so the 1,000-row) prepare recorded;
    every run records its runtime environment (``hybrid.h7r_d1.numeric_settings``); the six runs are comparable only
    if every field the contract names is equal across them and to the prepare record, the arms of a seed have the
    same data assignment, step count and evaluation steps, and all ran on one clean commit; otherwise fit records
    the reasons and no reading, and evaluate refuses;
    the R2-0B runs of the same seeds are a pinned reference only: their T-val curve, selected step and objective
    (fit) and their H-dev primary (evaluate) are reported beside the new narrow runs, never ruled on;
    evaluate writes in A1's order, re-checks HEAD right before the result and resumes without a new inference.

    prepare / train --arm --seed / fit / evaluate [--resume]   (each stage needs its own approval, in order)
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_d1 as cli  # noqa: E402
import h7r_d1_5k as k5  # noqa: E402
import h7r_d1_evaluate as ev  # noqa: E402
from hybrid import h7r_d1 as d1  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d1-contract-full-v1'
STAGE = 'full'
PREP_FORMAT = 'h7r-d1-prepare-full-v1'
TOOL_PATH = 'scripts/h7r_d1_full.py'
STAGES, SEEDS, ARMS = cli.STAGES, cli.SEEDS, cli.ARMS
SOURCES = (*k5.SOURCES, TOOL_PATH)
ENV_FIELDS = ('torch', 'cuda', 'cudnn', 'device_name', 'cudnn_enabled', 'cudnn_benchmark', 'cudnn_deterministic',
              'cudnn_allow_tf32', 'matmul_allow_tf32', 'float32_matmul_precision', 'deterministic_algorithms',
              'threads')
KEYS = {'format', 'contract_id', 'stage', 'status', 'approved', 'predecessor', 'r2_contract', 'amendment', 'epsilon',
        'arms', 'control', 'widen', 'data', 'budget', 'pairing', 'environment', 'seeds', 'fit', 'ruling',
        'reference', 'evaluate', 'next', 'sources', 'output', 'note'}
PREDECESSOR_KEYS = {'contract', 'experiment_sha256', 'prepare', 'fit', 'result', 'readings', 'entry_rule'}
DOCS = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/full/'


def load_d1_full(path, root: Path, stage: str, *, check_sources: bool = True) -> tuple[dict, str, str]:
    """The fixed, committed full-scale contract, ``stage`` and the earlier stages approved, every pin checked;
    (contract, contract file SHA-256, experiment SHA-256)."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no full-scale D1 contract')
    c = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(c, dict) and set(c) == KEYS,
               f'{path}: the full-scale contract keys differ from {sorted(KEYS)}')
    r2.require(c['format'] == FORMAT and c['contract_id'] == cli.CONTRACT_ID and c['stage'] == STAGE,
               f'{path}: not the full-scale D1 contract')
    r2.require(c['status'] == 'fixed', f'{path}: not fixed (a draft is never run)')
    r2.require(isinstance(c['approved'], dict) and set(c['approved']) == set(STAGES)
               and all(type(v) is bool for v in c['approved'].values()), 'D1 full approvals: one bool per stage')
    r2.require(c['approved'][stage] is True, f'the D1 full {stage} stage is not approved')
    earlier = STAGES[:STAGES.index(stage)]
    r2.require(all(c['approved'][e] for e in earlier), f'the D1 full {stage} stage needs {earlier} approved first')
    r2.require(committed(path, root), f'{path}: not committed')
    for key in ('r2_contract', 'amendment', 'epsilon'):
        r2.require(cli.is_ref(c[key]), f'D1 full {key}: a path and a full SHA-256 are required')
        cli.pinned(root, c[key], key)
    eps = cli.pinned(root, c['epsilon'], 'epsilon record')
    r2.require(c['ruling'].get('epsilon') == eps['epsilon']['value'], 'the ruling epsilon is not the R2-0B epsilon')
    pre = c['predecessor']
    r2.require(isinstance(pre, dict) and set(pre) == PREDECESSOR_KEYS
               and all(cli.is_ref(pre[k]) for k in ('contract', 'prepare', 'fit', 'result')),
               'D1 full predecessor: the 5,000-row contract and records, pinned')
    c5k = cli.pinned(root, pre['contract'], '5,000-row contract')
    eid5 = cli.experiment_sha256(c5k)
    r2.require(c5k.get('format') == k5.FORMAT and c5k.get('stage') == k5.STAGE and pre['experiment_sha256'] == eid5,
               'D1 full predecessor: not the 5,000-row experiment')
    prep5 = cli.pinned(root, pre['prepare'], '5,000-row prepare record')
    fit5 = cli.pinned(root, pre['fit'], '5,000-row train-fit record')
    res5 = cli.pinned(root, pre['result'], '5,000-row H-dev record')
    r2.require(prep5.get('format') == k5.PREP_FORMAT and prep5.get('passed') is True
               and prep5.get('experiment_sha256') == eid5, 'D1 full predecessor: not the passed 5,000-row prepare')
    r2.require(fit5.get('format') == cli.FIT_FORMAT and fit5.get('experiment_sha256') == eid5
               and res5.get('format') == cli.EVAL_FORMAT and res5.get('experiment_sha256') == eid5,
               'D1 full predecessor: the fit and H-dev records are not of the 5,000-row experiment')
    r2.require(pre['readings'] == {'train_fit': fit5['reading']['category'], 'hdev': res5['hdev_reading']['label']},
               'D1 full predecessor: the readings differ from the 5,000-row records')
    r2.require(fit5['reading']['category'] == 'WIDTH_EFFECT',
               'the 5,000-row train width effect did not hold in all three seeds: no full-scale stage (section 7.3)')
    for key in ('arms', 'seeds', 'fit', 'r2_contract', 'amendment', 'epsilon'):
        r2.require(c[key] == c5k[key], f'D1 full {key}: not the 5,000-row {key}')
    r2.require({k: v for k, v in c['ruling'].items() if k != 'scope'}
               == {k: v for k, v in c5k['ruling'].items() if k != 'scope'}, 'D1 full ruling: not the earlier rules')
    r2.require({k: v for k, v in c['reference'].items() if k != 'scale'}
               == {k: v for k, v in c5k['reference'].items() if k != 'scale'}, 'D1 full reference differs')
    w = c['widen']
    r2.require(set(w) == {'init', 'init_seeds', 'exact', 'device_check', 'parents'}
               and all(w[k] == c5k['widen'][k] for k in ('init', 'init_seeds', 'exact', 'device_check')),
               'D1 full widen: the earlier construction, init seeds and checks')
    d = c['data']
    r2.require(set(d) == {'source', 'split', 'rows', 'split_sha256', 'labels_sha256', 'composition_known',
                          'recorded_at_prepare'} and d['split'] == 'train' and d['rows'] == 35816
               and type(d['rows']) is int and r2.is_sha(d['split_sha256']) and r2.is_sha(d['labels_sha256'])
               and set(d['composition_known']) == {'outcome_rows', 'proof_rows'}
               and sum(d['composition_known'].values()) == d['rows'],
               'D1 full data: the whole c8 T-train split (35,816 rows) with its hashes and composition')
    b = c['budget']
    r2.require(set(b) == {'config', 'total_steps', 'note'} and b['config'] == c5k['budget']['config']
               and b['total_steps'] == 210 and type(b['total_steps']) is int,
               'D1 full budget: the c8 config, 210 steps')
    ctl = c['control']
    r2.require(isinstance(ctl, dict) and set(ctl) == {'kind', 'reason', 'r2_0b_reference', 'use'}
               and ctl['kind'] == 'trained anew', 'D1 full control: both arms trained anew')
    ref = ctl['r2_0b_reference']
    r2.require(set(ref) == {*SEEDS, 'epsilon_record'} and ref['epsilon_record'] == c['epsilon']
               and all(set(ref[s]) == {'record', 'r2_train'} and cli.is_ref(ref[s]['record'])
                       and cli.is_ref(ref[s]['r2_train']) for s in SEEDS),
               'D1 full control: the R2-0B reference runs of the three seeds, pinned')
    for s in SEEDS:
        cli.pinned(root, ref[s]['record'], f'R2-0B {s} record')
        cli.pinned(root, ref[s]['r2_train'], f'R2-0B {s} run record')
    env = c['environment']
    r2.require(isinstance(env, dict) and set(env) == {'fields', 'rule'} and tuple(env['fields']) == ENV_FIELDS,
               f'D1 full environment: the fields {ENV_FIELDS} and the equality rule')
    r2.require(isinstance(c['pairing'], dict) and set(c['pairing']) == {'per_seed', 'rule'},
               'D1 full pairing: the per-seed pairing and its rule')
    e = c['evaluate']
    r2.require(isinstance(e, dict) and set(e) == {'write_order', 'resume', 'predicted_record', 'head_check'}
               and e['resume'] is True and isinstance(e['predicted_record'], str)
               and e['predicted_record'].startswith('runs/') and e['predicted_record'].endswith('.json'),
               'D1 full evaluate: a predictions record under runs/')
    sources = c['sources']
    r2.require(isinstance(sources, dict) and set(sources) == set(SOURCES)
               and all(r2.is_sha(v) for v in sources.values()), 'D1 full sources: every file pinned by SHA-256')
    r2.require({p: sources[p] for p in k5.SOURCES} == c5k['sources'],
               'D1 full sources: the earlier tools are not the ones the 5,000-row contract pins')
    if check_sources:
        changed = [p for p in SOURCES if r2.file_sha256(root / p) != sources[p]]
        r2.require(not changed, f'the checkout differs from the sources D1 full was fixed for: {changed}')
    out = c['output']
    r2.require(set(out) == {'prepare', 'parents', 'run_dirs', 'fit', 'result', 'predictions'}
               and all(out[k].startswith(DOCS) for k in ('prepare', 'fit', 'result'))
               and out['parents'].startswith('runs/') and out['predictions'].startswith('runs/')
               and set(out['run_dirs']) == set(ARMS) and all(set(v) == set(SEEDS) and all(
                   x.startswith('runs/') for x in v.values()) for v in out['run_dirs'].values()),
               'D1 full outputs: records under docs/.../full/, parents, runs and predictions under runs/')
    return c, r2.file_sha256(path), cli.experiment_sha256(c)


def predecessor_prepare(c: dict, root: Path) -> dict:
    return cli.pinned(root, c['predecessor']['prepare'], '5,000-row prepare record')


def check_reference_inputs(c: dict, ref: dict) -> None:
    """The A1 reference record names the split and labels the contract pins."""
    r2.require(ref['inputs']['cycle_cache_splits']['train'] == c['data']['split_sha256']
               and ref['inputs']['cycle_labels_sha256'] == c['data']['labels_sha256'],
               'the A1 reference cycle cache is not the split and labels the full-scale contract pins')


def full_data(c: dict, cycle: dict, cycle_cache) -> tuple[dict, dict]:
    """The whole cycle train split (no selection) and its description: rows, game count, game-id SHA-256."""
    from hybrid.h7r_d4 import digest

    train = cycle['train']
    ids = json.loads((Path(cycle_cache) / 'train_games.json').read_text(encoding='utf-8'))
    present = sorted({ids[g] for g in train['game'].tolist()})
    desc = {'games': len(present), 'rows': int(train['value'].shape[0]), 'games_sha256': digest(present)}
    r2.require(desc['rows'] == c['data']['rows'], f"the train split has {desc['rows']} rows, the contract "
                                                  f"{c['data']['rows']}")
    known = c['data']['composition_known']
    kind = train['kind']
    rows = {'outcome_rows': int((kind == 0).sum()), 'proof_rows': int((kind == 1).sum())}
    r2.require(rows == known, f'the train split composition {rows} is not the contract\'s {known}')
    return train, desc


# --- prepare ---------------------------------------------------------------------------------------------------------

def prepare(args, log=print) -> dict:
    """The 5,000-row prepare on the whole split, gated the same way (bytes, the parent state of the earlier stages,
    CPU float64, finite device outputs, reload); every step's elapsed time is printed and recorded."""
    import torch
    from hybrid.h6_value_train import state_sha256
    from hybrid.h7r_d4 import digest
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1_full(args.d1_contract, root, 'prepare')
    out = root / c['output']['prepare']
    r2.require(not out.exists() and not (root / c['output']['parents']).exists(),
               'a D1 full prepare output exists: never overwritten')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'prepare needs a clean tracked tree')
    started = time.monotonic()
    timings = {}

    def mark(name):
        timings[name] = round(time.monotonic() - started, 1)
        log(f'[d1-full-prepare] {name} ({timings[name]:.0f} s)', flush=True)
    prep5 = predecessor_prepare(c, root)
    r2c, ref, _ = cli.reference_inputs(c, root)
    check_reference_inputs(c, ref)
    cycle, replay = cli.checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache, ('val',))
    subset, desc = full_data(c, cycle, args.cycle_cache)
    mark(f"data: the whole train split, {desc['games']} games / {desc['rows']} rows")
    fl = d1.floors(subset)
    mark('floors (exact and D4)')
    c5, c5_identity = cli.load_c5(r2c, args.c5, args.device)
    r2.require(c5_identity == prep5['c5'], 'c5 is not the one the earlier prepares checked')
    meta_c5 = json.loads(Path(args.c5).with_name('best.json').read_text(encoding='utf-8'))
    ex, batch = c['widen']['exact'], c['widen']['device_check']['batch_size']
    tol = ex['tolerance']
    sets = {'subset': subset, 'cycle_val': cycle['val'], 'replay_val': replay['val']}
    sample = d1.sample_index(int(replay['val']['value'].shape[0]), ex['replay_val_rows'], ex['replay_val_seed'])
    exact_sets = {'subset': None, 'cycle_val': None, 'replay_val': sample}
    train_rows = min(ex['train_mode_rows'], desc['rows'])
    settings = d1.numeric_settings(args.device)
    exact_c5 = {k: d1.exact_values(c5, sets[k], i) for k, i in exact_sets.items()}
    exact_c5['train_mode'] = d1.exact_values(c5, subset, torch.arange(train_rows), train_mode=True)
    mark('c5 float64 CPU values')
    device_c5 = {k: d1.predict_rows(c5, v, args.device, batch) for k, v in sets.items()}
    device_c5['train_mode'] = d1.train_mode_values(c5, subset, args.device, train_rows)
    split = {k: d1.compare(device_c5[k], d1.predict_rows(c5, v, args.device, max(1, batch // 4)))
             for k, v in sets.items()}
    mark('c5 device values and the batch-split reference')
    widened = {}
    (root / c['output']['parents']).mkdir(parents=True)
    for s in SEEDS:
        init = c['widen']['init_seeds'][s]
        wide = d1.widen_value_head(c5.cpu(), c['arms']['wide']['value_channels'], init)
        checks = d1.preservation(c5, wide)  # both on the CPU
        earlier = prep5['widened'][s]['state_sha256']
        same_parent = checks['state_sha256'] == earlier == prep5['widened'][s]['parent_state_sha256_1k']
        exact = {k: d1.compare(exact_c5[k], d1.exact_values(wide, sets[k], i), i) for k, i in exact_sets.items()}
        exact['train_mode'] = d1.compare(exact_c5['train_mode'],
                                         d1.exact_values(wide, subset, torch.arange(train_rows), train_mode=True))
        mark(f'seed {s}: float64 CPU values')
        c5.to(args.device)
        wide.to(args.device)
        device = {k: d1.compare(device_c5[k], d1.predict_rows(wide, v, args.device, batch)) for k, v in sets.items()}
        device['train_mode'] = d1.compare(device_c5['train_mode'], d1.train_mode_values(wide, subset, args.device,
                                                                                         train_rows))
        layers = {k: d1.layer_differences(c5, wide, v, args.device, batch) for k, v in sets.items()}
        ckpt, meta_path = k5.parent_paths(c, root, s)
        save_checkpoint(ckpt, wide)
        meta = {**meta_c5, 'model_config': asdict(wide.config),
                'd1': {'experiment_sha256': eid, 'widened_from': c5_identity['value_checkpoint_sha256'],
                       'init_seed': init, 'value_channels': wide.config.value_channels}}
        meta_path.write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        loaded, _ = r2.load_value_model(ckpt, meta_path, args.device)
        reload = {'state_equal': state_sha256(loaded.state_dict()) == checks['state_sha256'],
                  'value_channels': loaded.config.value_channels,
                  'prediction': d1.compare(d1.predict_rows(wide, subset, args.device, batch),
                                           d1.predict_rows(loaded, subset, args.device, batch))}
        gates = {
            'bytes': checks['passed'],
            'same_parent_as_earlier': same_parent,
            'exact': all(v['finite'] and v['max'] <= tol for v in exact.values()),
            'device_finite': all(v['finite'] for v in device.values()),
            'reload': reload['state_equal'] and reload['value_channels'] == wide.config.value_channels
            and reload['prediction']['finite'] and reload['prediction']['max'] == 0.0,
        }
        ok = all(gates.values())
        widened[s] = {'init_seed': init, **checks, 'parent_state_sha256_earlier': earlier,
                      'gates': gates, 'exact': exact, 'device': device, 'layers': layers, 'reload': reload,
                      'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta_path),
                      'passed': ok}
        worst = max((v['max'] for v in exact.values() if v['max'] is not None), default=None)
        dev = max((v['max'] for v in device.values() if v['max'] is not None), default=None)
        mark(f"seed {s}: widened parent {'PASS' if ok else 'FAIL'} {gates} (float64 max |dv| {worst}, device "
             f'float32 max |dv| {dev})')
    r2.require(cli.git_state(root) == state, 'HEAD changed during prepare')
    record = {'format': PREP_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
              'data': desc, 'composition': cli.composition(subset), 'floors': fl, 'c5': c5_identity,
              'exact': {**ex, 'replay_val_sample_sha256': digest(sample.tolist()), 'train_mode_rows_used': train_rows},
              'device_check': {'batch_size': batch, 'settings': settings,
                               'c5_batch_split': {'batch_size': max(1, batch // 4), 'compare': split}},
              'widened': widened, 'timings_seconds': timings,
              'passed': all(v['passed'] for v in widened.values())}
    publish_json(out, record)
    r2.require(record['passed'], 'a widened parent failed its checks: no training')
    return record


def checked_prepare(c: dict, root: Path, eid: str) -> dict:
    from hybrid.h7c_nmask import committed
    path = root / c['output']['prepare']
    r2.require(path.is_file() and committed(path, root), f'{path}: the full prepare record is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    r2.require(rec.get('format') == PREP_FORMAT and rec.get('experiment_sha256') == eid and rec.get('passed') is True
               and rec.get('data', {}).get('rows') == c['data']['rows'],
               'the full prepare record is not a passed record of this experiment')
    return rec


# --- train -----------------------------------------------------------------------------------------------------------

def train(args, log=print) -> dict:
    """One arm and seed, as ``h7r_d1_5k.py train`` on the whole split, with the runtime environment recorded."""
    import torch
    from hybrid import h7c_train as tr
    from hybrid.h6_value_train import frozen_state, state_sha256
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1_full(args.d1_contract, root, 'train')
    r2.require(args.arm in ARMS and args.seed in SEEDS, 'arm narrow / wide and a D1 seed')
    run_dir = root / c['output']['run_dirs'][args.arm][args.seed]
    r2.require(not run_dir.exists(), f'{run_dir} exists: never overwritten')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'a D1 run needs a clean tracked tree')
    prep = checked_prepare(c, root, eid)
    r2c, ref, config = cli.reference_inputs(c, root)
    check_reference_inputs(c, ref)
    cycle, replay = cli.checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache)
    subset, desc = full_data(c, cycle, args.cycle_cache)
    r2.require(desc == prep['data'], 'the data is not the prepared one')
    c5, _ = cli.load_c5(r2c, args.c5, args.device)
    if args.arm == 'narrow':
        parent = c5
    else:
        ckpt, meta = k5.parent_paths(c, root, args.seed)
        w = prep['widened'][args.seed]
        r2.require(r2.file_sha256(ckpt) == w['checkpoint_sha256'] and r2.file_sha256(meta) == w['metadata_sha256'],
                   'the widened parent is not the prepared one')
        parent, _ = r2.load_value_model(ckpt, meta, args.device)
    n, n_replay = int(subset['value'].shape[0]), int(replay['train']['value'].shape[0])
    per_epoch = -(-2 * n // config['batch_size'])
    epochs = c['budget']['total_steps'] / per_epoch
    cfg = tr.ValueConfig.from_dict({**config, 'seed': int(args.seed), 'epochs': epochs, 'device': args.device})
    total = max(1, round(cfg.epochs * per_epoch))
    r2.require(total == c['budget']['total_steps'], f'{total} steps, the budget is {c["budget"]["total_steps"]}')
    assignment = r2.data_assignment(n, n_replay, int(args.seed), cfg.batch_size, cfg.epochs, cfg.augment_d4)
    steps = tr.evaluation_steps(total, cfg.evaluations)
    environment = d1.numeric_settings(args.device)
    log(f'[d1-full-train] {args.arm} seed {args.seed}: {n} rows, {total} steps ({per_epoch} per epoch, '
        f'{cfg.epochs:g} epochs), value_channels {parent.config.value_channels}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.monotonic()
    with d1.fit_probe(subset, steps, args.device) as probe:
        result = tr.train_value(cfg, parent, {'train': subset, 'val': cycle['val']}, replay, log=log)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    r2.require(cli.git_state(root) == state, 'HEAD changed during the D1 run')
    record = result['record']
    best = record['best']
    fits = probe['fits']
    final_fit = fits[-1]
    selected_fit = None if best is None else fits[1 + steps.index(best['step'])]
    run_dir.mkdir(parents=True)

    def save(state_dict, name):
        model = copy.deepcopy(parent).cpu()
        model.load_state_dict(state_dict)
        r2.require(state_sha256(frozen_state(model)) == r2c['frozen_state_sha256'], 'the trunk changed')
        save_checkpoint(run_dir / f'{name}.pt', model)
        meta = {'stage': tr.VALUE_LOADER_STAGE, 'h7c_stage': tr.VALUE_STAGE, 'track': 'B',
                'experiment': cli.CONTRACT_ID, 'arm': args.arm, 'seed': int(args.seed), 'which': name,
                'model_config': asdict(model.config), 'config': tr.config_dict(cfg), 'best': best, 'git': state,
                'value_trained': True, 'trunk_frozen': True, 'policy_head_frozen': True,
                'frozen_state_sha256': r2c['frozen_state_sha256'],
                'parent_value_checkpoint_sha256': r2c['models']['c5']['value_checkpoint_sha256']}
        (run_dir / f'{name}.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        return {'checkpoint_sha256': r2.file_sha256(run_dir / f'{name}.pt'),
                'metadata_sha256': r2.file_sha256(run_dir / f'{name}.json')}
    candidate = save(result['state'], 'best') if result['state'] is not None else None
    final = save(probe['final_state'], 'final')
    (run_dir / 'record.json').write_text(json.dumps({**record, 'config': tr.config_dict(cfg)}, indent=1) + '\n',
                                         encoding='utf-8')
    body = {'format': cli.TRAIN_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
            'arm': args.arm, 'seed': args.seed, 'value_channels': parent.config.value_channels,
            'subset': desc, 'assignment': assignment, 'total_steps': record['total_steps'],
            'evaluation_steps': steps, 'fits': fits,
            'last_step': {'step': total, 'fit': final_fit, 'state': final},
            'selected': {'step': None if best is None else best['step'], 'objective': None if best is None else
                         best['objective'], 'fit': selected_fit, 'state': candidate},
            'verdict': record['verdict'], 'trainer_record_sha256': r2.file_sha256(run_dir / 'record.json'),
            'environment': environment, 'seconds': round(seconds, 2), 'seconds_per_step': round(seconds / total, 4)}
    publish_json(run_dir / 'd1_train.json', body)
    return body


# --- comparability, fit ----------------------------------------------------------------------------------------------

def load_runs(c: dict, root: Path, eid: str) -> dict:
    """The six run records of this experiment (format, experiment, arm and seed checked; pairing is
    ``comparability``'s, so a mismatch is reported rather than raised)."""
    recs = {}
    for arm in ARMS:
        for s in SEEDS:
            path = root / c['output']['run_dirs'][arm][s] / 'd1_train.json'
            r2.require(path.is_file(), f'{path}: no run record')
            rec = json.loads(path.read_text(encoding='utf-8'))
            r2.require(rec.get('format') == cli.TRAIN_FORMAT and rec.get('experiment_sha256') == eid
                       and rec['arm'] == arm and rec['seed'] == s, f'{path}: not a run of this experiment')
            recs[(arm, s)] = rec
    return recs


def comparability(c: dict, prep: dict, recs: dict) -> list[str]:
    """Why the six runs are not comparable (empty when they are): the contract's environment fields equal across the
    runs and to the prepare record, one clean commit, and per seed the same data, assignment, steps and
    evaluation steps for both arms."""
    reasons = []
    fields = c['environment']['fields']
    reference = {k: prep['device_check']['settings'].get(k) for k in fields}
    for (arm, s), rec in sorted(recs.items()):
        env = rec.get('environment')
        if not isinstance(env, dict):
            reasons.append(f'{arm} {s}: no runtime environment recorded')
            continue
        diff = {k: (env.get(k), reference[k]) for k in fields if env.get(k) != reference[k]}
        if diff:
            reasons.append(f'{arm} {s}: runtime environment differs from the prepare record: {diff}')
        if rec['subset'] != prep['data']:
            reasons.append(f'{arm} {s}: the data is not the prepared one')
        if rec['total_steps'] != c['budget']['total_steps']:
            reasons.append(f"{arm} {s}: {rec['total_steps']} steps, the budget is {c['budget']['total_steps']}")
    commits = {(rec['git']['commit'], rec['git']['dirty']) for rec in recs.values()}
    if len(commits) != 1 or next(iter(commits))[1] is not False:
        reasons.append(f'the six runs did not all run on one clean commit: {sorted(commits)}')
    for s in SEEDS:
        a, b = recs[('narrow', s)], recs[('wide', s)]
        for key in ('assignment', 'total_steps', 'evaluation_steps', 'subset'):
            if a[key] != b[key]:
                reasons.append(f'seed {s}: the arms differ in {key}')
    return reasons


def r2_0b_reference(c: dict, root: Path, recs: dict, run_root: Path) -> dict:
    """Per seed, the new narrow run beside the R2-0B run of that seed (reference only, never ruled on)."""
    out = {}
    for s in SEEDS:
        old = cli.pinned(root, c['control']['r2_0b_reference'][s]['record'], f'R2-0B {s} record')
        rec = recs[('narrow', s)]
        trainer = run_root / c['output']['run_dirs']['narrow'][s] / 'record.json'
        new = json.loads(trainer.read_text(encoding='utf-8')) if trainer.is_file() else None
        curve_old = {e['step']: e['objective'] for e in old['evaluations']}
        curve_new = {} if new is None else {e['step']: e['objective'] for e in new['evaluations']}
        common = sorted(set(curve_old) & set(curve_new))
        out[s] = {'r2_0b': {'selected_step': old['best']['step'], 'objective': old['best']['objective'],
                            'verdict': old['verdict']},
                  'narrow': {'selected_step': rec['selected']['step'], 'objective': rec['selected']['objective'],
                             'verdict': rec['verdict']},
                  'same_selected_step': old['best']['step'] == rec['selected']['step'],
                  'objective_difference': None if rec['selected']['objective'] is None else
                  rec['selected']['objective'] - old['best']['objective'],
                  'curve_steps_compared': len(common),
                  'curve_max_abs_difference': max((abs(curve_new[k] - curve_old[k]) for k in common), default=None)}
    return out


def fit(args, log=print) -> dict:
    """The train-fit reading at the last step, only for comparable runs; otherwise the reasons are recorded and no
    reading is made (the stage stops)."""
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    c, csha, eid = load_d1_full(args.d1_contract, root, 'train')
    out = root / c['output']['fit']
    r2.require(not out.exists(), f'{out} exists: never overwritten')
    prep = checked_prepare(c, root, eid)
    recs = load_runs(c, root, eid)
    reasons = comparability(c, prep, recs)
    body = {'format': cli.FIT_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha,
            'comparable': not reasons, 'comparability_reasons': reasons,
            'environment_fields': list(c['environment']['fields']),
            'environment': prep['device_check']['settings'] if not reasons else
            {f'{a}{s}': r.get('environment') for (a, s), r in recs.items()},
            'reading': None if reasons else cli.fit_of(c, prep, recs),
            'parent_fit': {arm: {s: recs[(arm, s)]['fits'][0]['weighted_mse'] for s in SEEDS} for arm in ARMS},
            'selected_fit': {arm: {s: (recs[(arm, s)]['selected']['fit'] or {}).get('weighted_mse') for s in SEEDS}
                             for arm in ARMS},
            'selected_step': {arm: {s: recs[(arm, s)]['selected']['step'] for s in SEEDS} for arm in ARMS},
            'cost': {arm: {s: {'seconds': recs[(arm, s)]['seconds'], 'per_step': recs[(arm, s)]['seconds_per_step']}
                           for s in SEEDS} for arm in ARMS},
            'r2_0b_reference': r2_0b_reference(c, root, recs, root),
            'note': 'the reading uses the last-step state of both arms (same step); the selected checkpoints are '
                    'recorded apart and are what the H-dev reading uses; the R2-0B reference is not ruled on'}
    publish_json(out, body)
    r2.require(not reasons, 'the six runs are not comparable (reasons recorded in the fit record): no reading; '
                            + '; '.join(reasons))
    return body


def checked_fit(c: dict, root: Path, eid: str, prep: dict, recs: dict) -> dict:
    """The committed, comparable train-fit reading, equal to its recomputation from the run records."""
    from hybrid.h7c_nmask import committed
    path = root / c['output']['fit']
    r2.require(path.is_file() and committed(path, root), f'{path}: the train-fit record is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    r2.require(rec.get('format') == cli.FIT_FORMAT and rec.get('experiment_sha256') == eid,
               'the train-fit record is not of this experiment')
    r2.require(rec.get('comparable') is True and not comparability(c, prep, recs),
               'the six runs are not comparable: no H-dev reading')
    reading = cli.fit_of(c, prep, recs)
    r2.require(json.loads(json.dumps(reading)) == rec['reading'], 'the train-fit record differs from the run records')
    return reading


# --- evaluate --------------------------------------------------------------------------------------------------------

def checked_inputs(c: dict, root: Path, eid: str) -> tuple:
    prep = checked_prepare(c, root, eid)
    recs = load_runs(c, root, eid)
    fit_read = checked_fit(c, root, eid, prep, recs)
    r2c = cli.pinned(root, c['r2_contract'], 'R2 contract')
    eps = cli.pinned(root, c['epsilon'], 'epsilon record')
    return prep, recs, fit_read, r2c, eps


def publish_result(path: Path, body: dict) -> None:
    from hybrid.h7r_ledger import publish_json
    publish_json(path, body)


def finish(c, csha, eid, root, state, p, hdev, fit_read, r2c, eps, pred, data, info, route, started,
           resumed) -> dict:
    """The reading, published only if HEAD and the tracked tree are still those the run started with (checked after
    the reading, right before the result); otherwise nothing is written and the predictions and record stay."""
    paired, references, reading = ev.analyse(c, r2c, pred, data, info, route)
    r2.require(cli.git_state(root) == state, 'HEAD or the tracked tree changed during the evaluation: no result is '
                                             'written (the predictions and their record are kept)')
    control_reference = {s: {'narrow_primary': paired[s]['control'],
                             'r2_0b_primary': eps['seeds'][s]['primary'],
                             'difference': paired[s]['control'] - eps['seeds'][s]['primary']} for s in SEEDS}
    body = {'format': cli.EVAL_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state, 'hdev': hdev,
            'hdev_reading': reading, 'paired': paired, 'references': references,
            'r2_0b_reference': control_reference,
            'train_fit_reading': fit_read, 'reference_note': c['reference'],
            'predictions': {'path': c['output']['predictions'], 'sha256': r2.file_sha256(p['predictions'])},
            'evaluate_tool': {'tool': {'path': TOOL_PATH, 'sha256': c['sources'][TOOL_PATH]},
                              'predicted_record': {'path': c['evaluate']['predicted_record'],
                                                   'sha256': r2.file_sha256(p['record'])}},
            'resumed': resumed, 'minutes': round((time.monotonic() - started) / 60, 2),
            'scope': 'the full-scale D1 diagnosis: no D4 candidate, strength, promotion or V9 ruling by itself; a '
                     'DEVELOPMENT_SIGNAL only opens a separate D4 contract'}
    publish_result(p['result'], body)
    return body


def evaluate(args, log=print) -> dict:
    """c5 and the six selected checkpoints on the verified H-dev, read by the R2 section 7 rule; A1's write order,
    the HEAD re-check before the result, ``--resume`` after an interruption past the predictions record."""
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    started = time.monotonic()
    c, csha, eid = load_d1_full(args.d1_contract, root, 'evaluate')
    p = k5.paths(c, root)
    if args.resume:
        return resume(args, root, c, csha, eid, p, started, log)
    r2.require(not p['result'].exists(), f"{p['result']} exists: never overwritten")
    r2.require(not p['record'].exists() and not p['predictions'].exists(),
               'the predictions of an earlier attempt exist: complete them with --resume (never overwritten)')
    r2.require(not p['partial'].exists(),
               f"{p['partial']}: an unfinished predictions file without its record; it is never used and stays for "
               'inspection, the next step is a separate decision')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'the evaluation needs a clean tracked tree')
    _, recs, fit_read, r2c, eps = checked_inputs(c, root, eid)
    models, identities = ev.load_models(c, root, recs, r2c, args.c5, args.device)
    log('[d1-full-eval] H-dev predictions of c5 and the six selected checkpoints', flush=True)
    _, data, info, hdev = r2cli.check_hdev(r2c, Path(args.cache), root)
    r2.require(hdev == eps['hdev'], 'the H-dev checks differ from the R2-0B record')
    route = ho.routes(data)
    pred = {name: r2.predict(models.pop(name), data, args.device) for name in list(models)}
    r2.require(cli.git_state(root) == state, 'HEAD changed during the evaluation')
    p['predictions'].parent.mkdir(parents=True, exist_ok=True)
    tensors = {n: {k: torch.tensor(v, dtype=torch.float64) for k, v in pr.items()} for n, pr in pred.items()}
    torch.save({'experiment_sha256': eid, 'contract_sha256': csha, 'identities': identities, 'hdev': hdev,
                'game': torch.tensor(info['game']), 'ply': torch.tensor(info['ply']), 'models': tensors},
               p['partial'])
    psha = r2.file_sha256(p['partial'])
    publish_json(p['record'], {
        'format': ev.PREDICTED_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
        'identities': identities, 'hdev': hdev, 'rows': int(data['value'].shape[0]), 'models': sorted(pred),
        'predictions': {'path': c['output']['predictions'], 'sha256': psha},
        'status': 'predictions complete; the reading is computed from them next'})
    os.link(p['partial'], p['predictions'])
    p['partial'].unlink()
    log(f'[d1-full-eval] predictions recorded ({psha[:12]}); reading', flush=True)
    return finish(c, csha, eid, root, state, p, hdev, fit_read, r2c, eps, pred, data, info, route, started, None)


def resume(args, root, c, csha, eid, p, started, log) -> dict:
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli

    r2.require(not p['result'].exists(), f"{p['result']} exists: nothing to resume")
    r2.require(p['record'].is_file(), 'no predictions record: nothing to resume (a predictions file without its '
                                      'record is never used)')
    rec = json.loads(p['record'].read_text(encoding='utf-8'))
    r2.require(rec.get('format') == ev.PREDICTED_FORMAT and rec.get('experiment_sha256') == eid
               and rec.get('contract_sha256') == csha, 'the predictions record is not of this contract')
    psha = rec['predictions']['sha256']
    if not p['predictions'].exists() and p['partial'].is_file() and r2.file_sha256(p['partial']) == psha:
        os.link(p['partial'], p['predictions'])  # interrupted between the record and the link
        p['partial'].unlink()
    r2.require(p['predictions'].is_file() and r2.file_sha256(p['predictions']) == psha,
               'the predictions file is not the recorded one')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'the resumed evaluation needs a clean tracked tree')
    _, recs, fit_read, r2c, eps = checked_inputs(c, root, eid)
    _, identities = ev.load_models(c, root, recs, r2c, args.c5, args.device)
    r2.require(identities == rec['identities'], 'a checkpoint differs from the one the predictions were made with')
    _, data, info, hdev = r2cli.check_hdev(r2c, Path(args.cache), root)
    r2.require(hdev == eps['hdev'] == rec['hdev'], 'the H-dev checks differ from the recorded ones')
    stored = torch.load(p['predictions'], map_location='cpu', weights_only=True)
    n = int(data['value'].shape[0])
    expected = sorted(['c5', *(f'{arm}{s}' for arm in ARMS for s in SEEDS)])
    r2.require(stored['experiment_sha256'] == eid and stored['identities'] == identities and stored['hdev'] == hdev
               and stored['game'].tolist() == list(info['game']) and stored['ply'].tolist() == list(info['ply'])
               and sorted(stored['models']) == expected == rec['models'] and rec['rows'] == n,
               'the stored predictions are not of these models and H-dev rows')
    for name, pr in stored['models'].items():
        r2.require(set(pr) == set(r2.PREDICTIONS) and all(t.dtype == torch.float64 and t.shape == (n,)
                                                          and bool(torch.isfinite(t).all()) for t in pr.values()),
                   f'{name}: incomplete or non-finite stored predictions')
    pred = {name: {k: t.tolist() for k, t in pr.items()} for name, pr in stored['models'].items()}
    route = ho.routes(data)
    log('[d1-full-eval] resumed from the recorded predictions (no new inference)', flush=True)
    resumed = {'predicted_record_sha256': r2.file_sha256(p['record']), 'prediction_git': rec['git'],
               'resume_git': state}
    return finish(c, csha, eid, root, state, p, hdev, fit_read, r2c, eps, pred, data, info, route, started, resumed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p, t, f, e = (sub.add_parser(n) for n in ('prepare', 'train', 'fit', 'evaluate'))
    for s in (p, t, f, e):
        s.add_argument('--d1-contract', type=Path, required=True)
    for s in (p, t):
        s.add_argument('--cycle-cache', required=True)
        s.add_argument('--cycle-rows', required=True)
        s.add_argument('--replay-cache', required=True)
    for s in (p, t, e):
        s.add_argument('--c5', required=True, help='c5 value checkpoint (best.json beside it)')
        s.add_argument('--device', default='cuda')
    t.add_argument('--arm', required=True, choices=ARMS)
    t.add_argument('--seed', required=True, choices=SEEDS)
    e.add_argument('--cache', required=True, help='the H-dev cache')
    e.add_argument('--resume', action='store_true', help='complete an interrupted evaluation (no new inference)')
    args = parser.parse_args(argv)
    args.root = ROOT
    try:
        out = {'prepare': prepare, 'train': train, 'fit': fit, 'evaluate': evaluate}[args.command](args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    if args.command == 'prepare':
        print(f"[d1-full-prepare] DONE {'PASS' if out['passed'] else 'FAIL'}; {out['data']['games']} games / "
              f"{out['data']['rows']} rows; floor d4 {out['floors']['d4']['floor']:.6f}; "
              f"{out['timings_seconds']}", flush=True)
    elif args.command == 'train':
        print(f"[d1-full-train] {out['arm']} {out['seed']} {out['verdict']}: last step {out['last_step']['step']} fit "
              f"{out['last_step']['fit']['weighted_mse']:.6f}; selected step {out['selected']['step']}; "
              f"{out['seconds']} s", flush=True)
    elif args.command == 'fit':
        print(f"[d1-full-fit] {out['reading']['category']}; wide - narrow {out['reading']['wide_minus_narrow']}",
              flush=True)
    else:
        r = out['hdev_reading']
        print(f"[d1-full-eval] H-dev {r['label']}; improvement (narrow - wide) "
              f"{ {s: round(v['improvement'], 6) for s, v in r['seeds'].items()} }; control seed range "
              f"{r['control_seed_range']:.6g}; train fit {out['train_fit_reading']['category']}"
              f"{'; resumed' if out['resumed'] else ''}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
