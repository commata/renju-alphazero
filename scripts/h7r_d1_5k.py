"""R2-D1-HEAD-WIDTH-1, the 5,000-row stage: the existing value head (value_channels 1) against one widened to 4 on
the nested 111-game / 5,174-row subset the 1,000-row prepare recorded, 210 steps for both arms, seeds 31001-31003.

The 1,000-row tools stay as they are: ``scripts/h7r_d1.py`` is pinned by the 1,000-row v2 contract (its SHA-256 is
in the experiment id the committed 1,000-row records carry) and ``scripts/h7r_d1_evaluate.py`` by evaluate amendment
A1. This entry point reads its own contract (``h7r-d1-contract-5k-v1``, ``d1-5k-contract.json``) and runs the same
procedure, taking from those files the stage-independent parts unchanged: the subset selection (``subset_of``), the
floors, the widening and its checks, the fit probe and the two readings (``hybrid.h7r_d1``), the cache and model
checks, the run-record pairing and the train-fit record check, and the H-dev analysis (``h7r_d1_evaluate.analyse``).
What it adds is the link to the 1,000-row records:

    the contract pins the 1,000-row v2 contract, its experiment id, its prepare, fit and H-dev records and A1; the
    1,000-row train reading must open this stage (section 7.2);
    prepare re-selects the 5,000-row prefix and requires it to be the ``subset.nested`` the 1,000-row prepare
    recorded, and rebuilds the widened parents and requires each state SHA-256 to be the 1,000-row one;
    evaluate writes in A1's order (predictions .partial, predictions record, link, result) and ``--resume``
    completes an interrupted reading from the recorded predictions without a new inference.

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
import h7r_d1_evaluate as ev  # noqa: E402
from hybrid import h7r_d1 as d1  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d1-contract-5k-v1'
STAGE = '5k'
PREP_FORMAT = 'h7r-d1-prepare-5k-v1'
TOOL_PATH = 'scripts/h7r_d1_5k.py'
STAGES, SEEDS, ARMS = cli.STAGES, cli.SEEDS, cli.ARMS
SOURCES = (*cli.SOURCES, 'scripts/h7r_d1_evaluate.py', TOOL_PATH, 'src/hybrid/h7r_ledger.py', 'src/hybrid/h7r_d4.py')
KEYS = {'format', 'contract_id', 'stage', 'status', 'approved', 'predecessor', 'r2_contract', 'amendment', 'epsilon',
        'arms', 'widen', 'subset', 'budget', 'seeds', 'fit', 'ruling', 'reference', 'evaluate', 'sources', 'output',
        'note'}
PREDECESSOR_KEYS = {'contract', 'experiment_sha256', 'prepare', 'fit', 'result', 'evaluate_amendment', 'readings',
                    'entry_rule'}
OPENS = ('WIDTH_EFFECT', 'BOTH_FIT')  # section 7.2


def load_d1_5k(path, root: Path, stage: str, *, check_sources: bool = True) -> tuple[dict, str, str]:
    """The fixed, committed 5,000-row contract, ``stage`` and the earlier stages approved, every pin checked;
    (contract, contract file SHA-256, experiment SHA-256)."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no 5,000-row D1 contract')
    c = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(c, dict) and set(c) == KEYS,
               f'{path}: the 5,000-row contract keys differ from {sorted(KEYS)}')
    r2.require(c['format'] == FORMAT and c['contract_id'] == cli.CONTRACT_ID and c['stage'] == STAGE,
               f'{path}: not the 5,000-row D1 contract')
    r2.require(c['status'] == 'fixed', f'{path}: not fixed (a draft is never run)')
    r2.require(isinstance(c['approved'], dict) and set(c['approved']) == set(STAGES)
               and all(type(v) is bool for v in c['approved'].values()), 'D1 5k approvals: one bool per stage')
    r2.require(c['approved'][stage] is True, f'the D1 5k {stage} stage is not approved')
    earlier = STAGES[:STAGES.index(stage)]
    r2.require(all(c['approved'][e] for e in earlier), f'the D1 5k {stage} stage needs {earlier} approved first')
    r2.require(committed(path, root), f'{path}: not committed')
    for key in ('r2_contract', 'amendment', 'epsilon'):
        r2.require(cli.is_ref(c[key]), f'D1 5k {key}: a path and a full SHA-256 are required')
        cli.pinned(root, c[key], key)
    eps = cli.pinned(root, c['epsilon'], 'epsilon record')
    r2.require(c['ruling'].get('epsilon') == eps['epsilon']['value'], 'the ruling epsilon is not the R2-0B epsilon')
    pre = c['predecessor']
    r2.require(isinstance(pre, dict) and set(pre) == PREDECESSOR_KEYS
               and all(cli.is_ref(pre[k]) for k in ('contract', 'prepare', 'fit', 'result', 'evaluate_amendment')),
               'D1 5k predecessor: the 1,000-row contract and records, pinned')
    v2 = cli.pinned(root, pre['contract'], '1,000-row v2 contract')
    eid1 = cli.experiment_sha256(v2)
    r2.require(v2.get('format') == 'h7r-d1-contract-v2' and v2.get('stage') == '1k-v2'
               and pre['experiment_sha256'] == eid1, 'D1 5k predecessor: not the 1,000-row v2 experiment')
    prep1 = cli.pinned(root, pre['prepare'], '1,000-row prepare record')
    fit1 = cli.pinned(root, pre['fit'], '1,000-row train-fit record')
    res1 = cli.pinned(root, pre['result'], '1,000-row H-dev record')
    cli.pinned(root, pre['evaluate_amendment'], '1,000-row evaluate amendment')
    r2.require(prep1.get('format') == cli.PREP_FORMAT and prep1.get('passed') is True
               and prep1.get('experiment_sha256') == eid1, 'D1 5k predecessor: not the passed 1,000-row prepare')
    r2.require(fit1.get('format') == cli.FIT_FORMAT and fit1.get('experiment_sha256') == eid1
               and res1.get('format') == cli.EVAL_FORMAT and res1.get('experiment_sha256') == eid1,
               'D1 5k predecessor: the fit and H-dev records are not of the 1,000-row experiment')
    r2.require(pre['readings'] == {'train_fit': fit1['reading']['category'], 'hdev': res1['hdev_reading']['label']},
               'D1 5k predecessor: the readings differ from the 1,000-row records')
    r2.require(fit1['reading']['category'] in OPENS, 'the 1,000-row train reading does not open the 5,000-row stage')
    for key in ('arms', 'seeds', 'fit', 'reference'):
        if key == 'reference':
            r2.require({k: v for k, v in c[key].items() if k != 'scale'} == v2[key], 'D1 5k reference differs')
        else:
            r2.require(c[key] == v2[key], f'D1 5k {key}: not the 1,000-row {key}')
    r2.require(c['seeds'] == list(SEEDS) and c['arms'] == {'narrow': {'value_channels': 1},
                                                           'wide': {'value_channels': 4}}, 'D1 5k arms or seeds differ')
    r2.require({k: v for k, v in c['ruling'].items() if k != 'scope'} == {k: v for k, v in v2['ruling'].items()
                                                                          if k != 'scope'},
               'D1 5k ruling: not the 1,000-row rules')
    w = c['widen']
    r2.require(set(w) == {'init', 'init_seeds', 'exact', 'device_check', 'parents'}
               and all(w[k] == v2['widen'][k] for k in ('init', 'init_seeds', 'exact', 'device_check')),
               'D1 5k widen: the 1,000-row construction, init seeds and checks')
    s = c['subset']
    r2.require(set(s) == {'source', 'seed', 'rows', 'selection', 'expected'} and s['rows'] == 5000
               and s['seed'] == v2['subset']['seed'] and s['expected'] == prep1['subset']['nested'],
               'D1 5k subset: the nested 5,000-row prefix the 1,000-row prepare recorded')
    b = c['budget']
    r2.require(set(b) == {'config', 'total_steps', 'decided'} and b['config'] == v2['budget']['config']
               and b['total_steps'] == 210 and type(b['total_steps']) is int,
               'D1 5k budget: the c8 config and 210 steps')
    e = c['evaluate']
    r2.require(isinstance(e, dict) and set(e) == {'write_order', 'resume', 'predicted_record'} and e['resume'] is True
               and isinstance(e['predicted_record'], str) and e['predicted_record'].startswith('runs/')
               and e['predicted_record'].endswith('.json'), 'D1 5k evaluate: a predictions record under runs/')
    sources = c['sources']
    r2.require(isinstance(sources, dict) and set(sources) == set(SOURCES)
               and all(r2.is_sha(v) for v in sources.values()), 'D1 5k sources: every file pinned by SHA-256')
    if check_sources:
        changed = [p for p in SOURCES if r2.file_sha256(root / p) != sources[p]]
        r2.require(not changed, f'the checkout differs from the sources D1 5k was fixed for: {changed}')
    out = c['output']
    docs = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/5k/'
    r2.require(set(out) == {'prepare', 'parents', 'run_dirs', 'fit', 'result', 'predictions'}
               and all(out[k].startswith(docs) for k in ('prepare', 'fit', 'result'))
               and out['parents'].startswith('runs/') and out['predictions'].startswith('runs/')
               and set(out['run_dirs']) == set(ARMS) and all(set(v) == set(SEEDS) and all(
                   x.startswith('runs/') for x in v.values()) for v in out['run_dirs'].values()),
               'D1 5k outputs: records under docs/.../5k/, parents, runs and predictions under runs/')
    return c, r2.file_sha256(path), cli.experiment_sha256(c)


def predecessor(c: dict, root: Path) -> tuple[dict, dict]:
    pre = c['predecessor']
    return cli.pinned(root, pre['contract'], '1,000-row v2 contract'), cli.pinned(root, pre['prepare'],
                                                                                   '1,000-row prepare record')


def subset_5k(c: dict, cycle: dict, cycle_cache) -> tuple[dict, dict]:
    """The 5,000-row prefix by the 1,000-row selection code itself (``h7r_d1.subset_of``); its description."""
    data, desc = cli.subset_of({'subset': {'seed': c['subset']['seed'], 'rows': c['subset']['rows'],
                                           'nested_rows': c['subset']['rows']}}, cycle, cycle_cache)
    desc = {k: v for k, v in desc.items() if k != 'nested'}
    r2.require(desc == c['subset']['expected'], f"the 5,000-row subset {desc} is not the nested set the 1,000-row "
                                                f"prepare recorded {c['subset']['expected']}")
    return data, desc


def parent_paths(c: dict, root: Path, seed: str) -> tuple[Path, Path]:
    base = root / c['output']['parents']
    return base / f'wide-{seed}.pt', base / f'wide-{seed}.json'


# --- prepare ---------------------------------------------------------------------------------------------------------

def prepare(args, log=print) -> dict:
    """The 1,000-row v2 prepare on the 5,000-row subset, gated the same way (bytes, CPU float64, finite device
    outputs, reload) and in addition on each widened parent's state SHA-256 being the 1,000-row one."""
    import torch
    from hybrid.h6_value_train import state_sha256
    from hybrid.h7r_d4 import digest
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1_5k(args.d1_contract, root, 'prepare')
    out = root / c['output']['prepare']
    r2.require(not out.exists() and not (root / c['output']['parents']).exists(),
               'a D1 5k prepare output exists: never overwritten')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'prepare needs a clean tracked tree')
    _, prep1 = predecessor(c, root)
    r2c, ref, _ = cli.reference_inputs(c, root)
    cycle, replay = cli.checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache, ('val',))
    subset, desc = subset_5k(c, cycle, args.cycle_cache)
    log(f"[d1-5k-prepare] subset {desc['games']} games / {desc['rows']} rows = the 1,000-row nested set", flush=True)
    fl = d1.floors(subset)
    c5, c5_identity = cli.load_c5(r2c, args.c5, args.device)
    r2.require(c5_identity == prep1['c5'], 'c5 is not the one the 1,000-row prepare checked')
    meta_c5 = json.loads(Path(args.c5).with_name('best.json').read_text(encoding='utf-8'))
    ex, batch = c['widen']['exact'], c['widen']['device_check']['batch_size']
    tol = ex['tolerance']
    sets = {'subset': subset, 'cycle_val': cycle['val'], 'replay_val': replay['val']}
    sample = d1.sample_index(int(replay['val']['value'].shape[0]), ex['replay_val_rows'], ex['replay_val_seed'])
    exact_sets = {'subset': None, 'cycle_val': None, 'replay_val': sample}
    train_rows = min(ex['train_mode_rows'], desc['rows'])
    settings = d1.numeric_settings(args.device)
    started = time.monotonic()
    exact_c5 = {k: d1.exact_values(c5, sets[k], i) for k, i in exact_sets.items()}
    exact_c5['train_mode'] = d1.exact_values(c5, subset, torch.arange(train_rows), train_mode=True)
    log(f'[d1-5k-prepare] c5 float64 CPU values ({time.monotonic() - started:.0f} s)', flush=True)
    device_c5 = {k: d1.predict_rows(c5, v, args.device, batch) for k, v in sets.items()}
    device_c5['train_mode'] = d1.train_mode_values(c5, subset, args.device, train_rows)
    split = {k: d1.compare(device_c5[k], d1.predict_rows(c5, v, args.device, max(1, batch // 4)))
             for k, v in sets.items()}
    widened = {}
    (root / c['output']['parents']).mkdir(parents=True)
    for s in SEEDS:
        init = c['widen']['init_seeds'][s]
        wide = d1.widen_value_head(c5.cpu(), c['arms']['wide']['value_channels'], init)
        checks = d1.preservation(c5, wide)  # both on the CPU
        same_parent = checks['state_sha256'] == prep1['widened'][s]['state_sha256']
        exact = {k: d1.compare(exact_c5[k], d1.exact_values(wide, sets[k], i), i) for k, i in exact_sets.items()}
        exact['train_mode'] = d1.compare(exact_c5['train_mode'],
                                         d1.exact_values(wide, subset, torch.arange(train_rows), train_mode=True))
        c5.to(args.device)
        wide.to(args.device)
        device = {k: d1.compare(device_c5[k], d1.predict_rows(wide, v, args.device, batch)) for k, v in sets.items()}
        device['train_mode'] = d1.compare(device_c5['train_mode'], d1.train_mode_values(wide, subset, args.device,
                                                                                         train_rows))
        layers = {k: d1.layer_differences(c5, wide, v, args.device, batch) for k, v in sets.items()}
        ckpt, meta_path = parent_paths(c, root, s)
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
            'same_parent_as_1k': same_parent,
            'exact': all(v['finite'] and v['max'] <= tol for v in exact.values()),
            'device_finite': all(v['finite'] for v in device.values()),
            'reload': reload['state_equal'] and reload['value_channels'] == wide.config.value_channels
            and reload['prediction']['finite'] and reload['prediction']['max'] == 0.0,
        }
        ok = all(gates.values())
        widened[s] = {'init_seed': init, **checks, 'parent_state_sha256_1k': prep1['widened'][s]['state_sha256'],
                      'gates': gates, 'exact': exact, 'device': device, 'layers': layers, 'reload': reload,
                      'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta_path),
                      'passed': ok}
        worst = max((v['max'] for v in exact.values() if v['max'] is not None), default=None)
        dev = max((v['max'] for v in device.values() if v['max'] is not None), default=None)
        log(f"[d1-5k-prepare] seed {s}: widened parent {'PASS' if ok else 'FAIL'} {gates} (float64 max |dv| {worst}, "
            f'device float32 max |dv| {dev}; {time.monotonic() - started:.0f} s)', flush=True)
    r2.require(cli.git_state(root) == state, 'HEAD changed during prepare')
    record = {'format': PREP_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
              'subset': desc, 'subset_matches_1k_nested': desc == prep1['subset']['nested'],
              'composition': cli.composition(subset), 'floors': fl, 'c5': c5_identity,
              'exact': {**ex, 'replay_val_sample_sha256': digest(sample.tolist()), 'train_mode_rows_used': train_rows},
              'device_check': {'batch_size': batch, 'settings': settings,
                               'c5_batch_split': {'batch_size': max(1, batch // 4), 'compare': split}},
              'widened': widened, 'passed': all(v['passed'] for v in widened.values())}
    publish_json(out, record)
    r2.require(record['passed'], 'a widened parent failed its checks: no training')
    return record


def checked_prepare(c: dict, root: Path, eid: str) -> dict:
    from hybrid.h7c_nmask import committed
    path = root / c['output']['prepare']
    r2.require(path.is_file() and committed(path, root), f'{path}: the 5k prepare record is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    r2.require(rec.get('format') == PREP_FORMAT and rec.get('experiment_sha256') == eid and rec.get('passed') is True
               and rec.get('subset') == c['subset']['expected'],
               'the 5k prepare record is not a passed record of this experiment')
    return rec


# --- train / fit -----------------------------------------------------------------------------------------------------

def train(args, log=print) -> dict:
    """One arm and seed, as ``h7r_d1.py train``: the trainer on {subset, full cycle val} and the H6 replay with the
    c8 config and the step budget, observed by the fit probe; the last-step state and the T-val selected
    checkpoint recorded apart."""
    import torch
    from hybrid import h7c_train as tr
    from hybrid.h6_value_train import frozen_state, state_sha256
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1_5k(args.d1_contract, root, 'train')
    r2.require(args.arm in ARMS and args.seed in SEEDS, 'arm narrow / wide and a D1 seed')
    run_dir = root / c['output']['run_dirs'][args.arm][args.seed]
    r2.require(not run_dir.exists(), f'{run_dir} exists: never overwritten')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'a D1 run needs a clean tracked tree')
    prep = checked_prepare(c, root, eid)
    r2c, ref, config = cli.reference_inputs(c, root)
    cycle, replay = cli.checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache)
    subset, desc = subset_5k(c, cycle, args.cycle_cache)
    r2.require(desc == prep['subset'], 'the subset is not the prepared one')
    c5, _ = cli.load_c5(r2c, args.c5, args.device)
    if args.arm == 'narrow':
        parent = c5
    else:
        ckpt, meta = parent_paths(c, root, args.seed)
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
    log(f'[d1-5k-train] {args.arm} seed {args.seed}: {n} subset rows, {total} steps ({per_epoch} per epoch), '
        f'value_channels {parent.config.value_channels}', flush=True)
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
            'seconds': round(seconds, 2), 'seconds_per_step': round(seconds / total, 4)}
    publish_json(run_dir / 'd1_train.json', body)
    return body


def fit(args, log=print) -> dict:
    """The train-fit reading of the six runs at their last step, as ``h7r_d1.py fit``."""
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    c, csha, eid = load_d1_5k(args.d1_contract, root, 'train')
    out = root / c['output']['fit']
    r2.require(not out.exists(), f'{out} exists: never overwritten')
    prep = checked_prepare(c, root, eid)
    recs = cli.run_records(c, root, eid)
    reading = cli.fit_of(c, prep, recs)
    body = {'format': cli.FIT_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'reading': reading,
            'parent_fit': {arm: {s: recs[(arm, s)]['fits'][0]['weighted_mse'] for s in SEEDS} for arm in ARMS},
            'selected_fit': {arm: {s: (recs[(arm, s)]['selected']['fit'] or {}).get('weighted_mse') for s in SEEDS}
                             for arm in ARMS},
            'selected_step': {arm: {s: recs[(arm, s)]['selected']['step'] for s in SEEDS} for arm in ARMS},
            'cost': {arm: {s: {'seconds': recs[(arm, s)]['seconds'], 'per_step': recs[(arm, s)]['seconds_per_step']}
                           for s in SEEDS} for arm in ARMS},
            'note': 'the reading uses the last-step state of both arms (same step); the selected checkpoints are '
                    'recorded apart and are what the H-dev reading uses'}
    publish_json(out, body)
    return body


# --- evaluate --------------------------------------------------------------------------------------------------------

def paths(c: dict, root: Path) -> dict:
    preds = root / c['output']['predictions']
    return {'result': root / c['output']['result'], 'predictions': preds,
            'partial': preds.with_name(preds.name + '.partial'), 'record': root / c['evaluate']['predicted_record']}


def checked_inputs(c: dict, root: Path, eid: str) -> tuple:
    prep = checked_prepare(c, root, eid)
    recs = cli.run_records(c, root, eid)
    fit_read = cli.checked_fit(c, root, eid, prep, recs)
    r2c = cli.pinned(root, c['r2_contract'], 'R2 contract')
    eps = cli.pinned(root, c['epsilon'], 'epsilon record')
    return prep, recs, fit_read, r2c, eps


def publish_result(path: Path, body: dict) -> None:
    from hybrid.h7r_ledger import publish_json
    publish_json(path, body)


def finish(c, csha, eid, state, p, hdev, fit_read, r2c, pred, data, info, route, started, resumed) -> dict:
    paired, references, reading = ev.analyse(c, r2c, pred, data, info, route)
    body = {'format': cli.EVAL_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state, 'hdev': hdev,
            'hdev_reading': reading, 'paired': paired, 'references': references,
            'train_fit_reading': fit_read, 'reference_note': c['reference'],
            'predictions': {'path': c['output']['predictions'], 'sha256': r2.file_sha256(p['predictions'])},
            'evaluate_tool': {'tool': {'path': TOOL_PATH, 'sha256': c['sources'][TOOL_PATH]},
                              'predicted_record': {'path': c['evaluate']['predicted_record'],
                                                   'sha256': r2.file_sha256(p['record'])}},
            'resumed': resumed, 'minutes': round((time.monotonic() - started) / 60, 2),
            'scope': 'the 5,000-row D1 diagnosis: no D4 candidate, strength, promotion or V9 ruling'}
    publish_result(p['result'], body)
    return body


def evaluate(args, log=print) -> dict:
    """c5 and the six selected checkpoints on the verified H-dev, read by the R2 section 7 rule; written in A1's
    order so that ``--resume`` can complete an interruption after the predictions record."""
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    started = time.monotonic()
    c, csha, eid = load_d1_5k(args.d1_contract, root, 'evaluate')
    p = paths(c, root)
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
    log('[d1-5k-eval] H-dev predictions of c5 and the six selected checkpoints', flush=True)
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
    log(f'[d1-5k-eval] predictions recorded ({psha[:12]}); reading', flush=True)
    return finish(c, csha, eid, state, p, hdev, fit_read, r2c, pred, data, info, route, started, None)


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
    log('[d1-5k-eval] resumed from the recorded predictions (no new inference)', flush=True)
    resumed = {'predicted_record_sha256': r2.file_sha256(p['record']), 'prediction_git': rec['git'],
               'resume_git': state}
    return finish(c, csha, eid, state, p, hdev, fit_read, r2c, pred, data, info, route, started, resumed)


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
        print(f"[d1-5k-prepare] DONE {'PASS' if out['passed'] else 'FAIL'}; subset {out['subset']['rows']} rows; "
              f"floor d4 {out['floors']['d4']['floor']:.6f}", flush=True)
    elif args.command == 'train':
        print(f"[d1-5k-train] {out['arm']} {out['seed']} {out['verdict']}: last step {out['last_step']['step']} fit "
              f"{out['last_step']['fit']['weighted_mse']:.6f}; selected step {out['selected']['step']}; "
              f"{out['seconds']} s", flush=True)
    elif args.command == 'fit':
        print(f"[d1-5k-fit] {out['reading']['category']}; wide - narrow {out['reading']['wide_minus_narrow']}",
              flush=True)
    else:
        r = out['hdev_reading']
        print(f"[d1-5k-eval] H-dev {r['label']}; improvement (narrow - wide) "
              f"{ {s: round(v['improvement'], 6) for s, v in r['seeds'].items()} }; control seed range "
              f"{r['control_seed_range']:.6g}; train fit {out['train_fit_reading']['category']}"
              f"{'; resumed' if out['resumed'] else ''}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
