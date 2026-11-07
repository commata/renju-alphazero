"""R2-D1-HEAD-WIDTH-1, the 1,000-row first comparison: the existing value head (value_channels 1) against one widened
to 4, on the same c8 T-train subset and training budget, seeds 31001-31003. Each stage needs its own approval in the
fixed, committed D1 v2 contract (``h7r-d1-contract-v2``; it supersedes the v1 contract, whose prepare failed its
float32 tolerance, and keeps its subset, budget, arms, seeds and rules); the stages link through the
approval-independent experiment SHA-256 (as D3). Nothing is written over.

    prepare   the subset (whole c8 T-train games in a seeded order, until 1,000 rows; the nested 5,000-row prefix
              recorded too), its composition and target-conflict floors (exact and D4), the widened parents (one per
              seed: c5's head widened with its prediction preserved, ``hybrid.h7r_d1``). The gate: byte checks, CPU
              float64 values within the exact tolerance of c5's (subset, cycle val, a seeded H6 val sample, one
              train-mode batch), finite device outputs, and a reload to the same state with identical predictions.
              Recorded, not gated: the device float32 differences with their worst rows, the stage-by-stage
              diagnosis, a c5 batch-split reference and the arithmetic settings.
    train     one arm and seed: the trainer's ``train_value`` on {subset, full cycle val} and the H6 replay, the c8
              config with the step budget fixed, the narrow arm from c5 and the wide arm from its prepared parent.
              Both arms of a seed have the same data assignment. The trainer's own evaluations are observed
              (``fit_probe``): the train-subset fit of the same model at the parent and every evaluation step, and the
              state at the last step. Recorded apart: the last-step state and fit, and the checkpoint selected by the
              T-val objective with its fit. Steps and GPU seconds are recorded per run.
    fit       the train-fit reading at the last step (contract section 4), from the six run records.
    evaluate  c5 and the six selected checkpoints predicted together on the verified H-dev; the R2 rule unchanged on
              wide against the same-seed narrow run (control); components and the additive decomposition as
              references. The train-fit reading is carried beside it, never mixed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d1 as d1  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d1-contract-v2'
CONTRACT_ID = 'R2-D1-HEAD-WIDTH-1'
PREP_FORMAT = 'h7r-d1-prepare-v2'
V1_PREP_FORMAT = 'h7r-d1-prepare-v1'
TRAIN_FORMAT = 'h7r-d1-train-v1'
FIT_FORMAT = 'h7r-d1-fit-v1'
EVAL_FORMAT = 'h7r-d1-eval-v1'
STAGES = ('prepare', 'train', 'evaluate')
SEEDS = ('31001', '31002', '31003')
ARMS = ('narrow', 'wide')
SOURCES = ('src/hybrid/h7c_train.py', 'src/hybrid/h6_value_train.py', 'scripts/h7c_train_value.py',
           'src/hybrid/h7r_d1.py', 'scripts/h7r_d1.py', 'src/hybrid/h7r_d3.py', 'src/hybrid/h7r_r2.py',
           'scripts/h7r_r2.py', 'scripts/h7r_r2_seed.py', 'scripts/h7r_holdout.py', 'src/model/network.py',
           'src/model/checkpoint.py', 'src/model/config.py')
KEYS = {'format', 'contract_id', 'stage', 'status', 'approved', 'supersedes', 'r2_contract', 'amendment', 'epsilon',
        'arms', 'widen', 'subset', 'budget', 'seeds', 'fit', 'ruling', 'reference', 'sources', 'output'}
WIDEN_KEYS = {'init', 'init_seeds', 'exact', 'device_check'}


def is_ref(value) -> bool:
    return isinstance(value, dict) and set(value) == {'path', 'sha256'} and r2.is_sha(value.get('sha256'))


def pinned(root: Path, ref: dict, what: str) -> dict:
    path = root / ref['path']
    r2.require(path.is_file() and r2.file_sha256(path) == ref['sha256'], f"{ref['path']}: not the {what} D1 pins")
    return json.loads(path.read_text(encoding='utf-8'))


def git_state(root: Path) -> dict:
    def git(*a):
        return subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def experiment_sha256(contract: dict) -> str:
    from hybrid.h7r_d4 import digest
    return digest({k: v for k, v in contract.items() if k != 'approved'})


def load_d1(path, root: Path, stage: str, *, check_sources: bool = True) -> tuple[dict, str, str]:
    """The D1 contract, fixed, committed, ``stage`` and the earlier stages approved, every pin checked; returns
    (contract, contract file SHA-256, experiment SHA-256)."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no D1 contract')
    c = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the D1 contract keys differ from {sorted(KEYS)}')
    r2.require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed'
               and c['stage'] == '1k-v2', f'{path}: not the fixed 1,000-row D1 v2 contract')
    r2.require(isinstance(c['approved'], dict) and set(c['approved']) == set(STAGES)
               and all(type(v) is bool for v in c['approved'].values()), 'D1 approvals: one bool per stage')
    r2.require(c['approved'][stage] is True, f'the D1 {stage} stage is not approved')
    earlier = STAGES[:STAGES.index(stage)]
    r2.require(all(c['approved'][e] for e in earlier), f'the D1 {stage} stage needs {earlier} approved first')
    r2.require(committed(path, root), f'{path}: not committed')
    for key in ('r2_contract', 'amendment', 'epsilon'):
        r2.require(is_ref(c[key]), f'D1 {key}: a path and a full SHA-256 are required')
        pinned(root, c[key], key)
    eps = pinned(root, c['epsilon'], 'epsilon record')
    r2.require(c['ruling'].get('epsilon') == eps['epsilon']['value'], 'the ruling epsilon is not the R2-0B epsilon')
    r2.require(c['arms'] == {'narrow': {'value_channels': 1}, 'wide': {'value_channels': 4}}, 'D1 arms differ')
    r2.require(c['seeds'] == list(SEEDS), f'D1 seeds: exactly {SEEDS}')
    sup = c['supersedes']
    r2.require(isinstance(sup, dict) and set(sup) == {'contract', 'prepare', 'reason'} and is_ref(sup['contract'])
               and is_ref(sup['prepare']), 'D1 supersedes: the v1 contract and its prepare record, pinned')
    v1 = pinned(root, sup['contract'], 'superseded v1 contract')
    v1_prep = pinned(root, sup['prepare'], 'superseded v1 prepare record')
    r2.require(v1.get('format') == 'h7r-d1-contract-v1' and v1_prep.get('format') == V1_PREP_FORMAT
               and v1_prep.get('passed') is False and v1_prep.get('experiment_sha256') == experiment_sha256(v1),
               'D1 supersedes: not the failed v1 prepare of the v1 contract')
    w = c['widen']
    r2.require(set(w) == WIDEN_KEYS and w['init_seeds'] == v1['widen']['init_seeds']
               and w['init'] == v1['widen']['init'],
               'D1 widen: the v1 construction and init seeds, with the exact and device checks')
    e, dv = w['exact'], w['device_check']
    r2.require(set(e) == {'device', 'dtype', 'tolerance', 'train_mode_rows', 'replay_val_rows', 'replay_val_seed'}
               and e['device'] == 'cpu' and e['dtype'] == 'float64' and isinstance(e['tolerance'], float)
               and 0 < e['tolerance'] <= 1e-12 and type(e['train_mode_rows']) is int
               and type(e['replay_val_rows']) is int and type(e['replay_val_seed']) is int,
               'D1 exact check: CPU float64 within at most 1e-12, train-mode rows and a seeded replay val sample')
    r2.require(set(dv) == {'batch_size', 'gate'} and type(dv['batch_size']) is int and dv['gate'] == 'finite',
               'D1 device check: its batch size; only non-finite outputs fail it, differences are recorded')
    s = c['subset']
    r2.require(set(s) == {'source', 'seed', 'rows', 'nested_rows'} and s['rows'] == 1000 and s['nested_rows'] == 5000,
               'D1 subset: the 1,000-row first comparison with the nested 5,000 rows')
    b = c['budget']
    r2.require(set(b) == {'config', 'total_steps'} and is_ref(b['config']) and type(b['total_steps']) is int,
               'D1 budget: the pinned config and a total step count')
    a1 = pinned(root, c['amendment'], 'amendment')['r2_0b']
    r2.require(b['config'] == a1['config'], 'the D1 config is not the A1 (c8) config')
    f = c['fit']
    r2.require(set(f) == {'margin', 'floor', 'measured_at'} and f['floor'] == 'd4' and f['measured_at'] == 'last_step',
               'D1 fit: margin over the D4 floor at the last step')
    r2.require(c['subset'] == v1['subset'] and c['budget'] == v1['budget'] and c['arms'] == v1['arms']
               and c['seeds'] == v1['seeds'] and c['fit'] == v1['fit'],
               'D1 v2 keeps the v1 subset, budget, arms, seeds and fit rule')
    sources = c['sources']
    r2.require(isinstance(sources, dict) and set(sources) == set(SOURCES), 'D1 sources are malformed')
    if check_sources:
        changed = [p for p in SOURCES if r2.file_sha256(root / p) != sources[p]]
        r2.require(not changed, f'the checkout differs from the sources D1 was fixed for: {changed}')
    out = c['output']
    r2.require(set(out) == {'prepare', 'parents', 'run_dirs', 'fit', 'result', 'predictions'}
               and out['parents'].startswith('runs/') and out['predictions'].startswith('runs/')
               and set(out['run_dirs']) == set(ARMS) and all(set(v) == set(SEEDS) and all(
                   x.startswith('runs/') for x in v.values()) for v in out['run_dirs'].values()),
               'D1 outputs: parents, run directories and predictions under runs/')
    return c, r2.file_sha256(path), experiment_sha256(c)


# --- shared inputs ---------------------------------------------------------------------------------------------------

def reference_inputs(c: dict, root: Path):
    import h7r_r2_seed as seed
    r2c = pinned(root, c['r2_contract'], 'R2 contract')
    ref, config = seed.reference(pinned(root, c['amendment'], 'amendment'), r2c, root)
    return r2c, ref, config


def checked_caches(ref: dict, cycle_cache, cycle_rows, replay_cache=None, replay_splits=('train', 'val')):
    import h7c_train_value as tv
    tv.check_cycle_cache(Path(cycle_cache), ref['cycle'], ref['inputs']['selfplay_run_seed'], rows=Path(cycle_rows))
    cycle, cm = tv.load_cache_splits(Path(cycle_cache))
    r2.require(cm['labels_sha256'] == ref['inputs']['cycle_labels_sha256']
               and {k: v['sha256'] for k, v in cm['splits'].items()} == ref['inputs']['cycle_cache_splits'],
               'the cycle cache is not the c8 run\'s')
    replay = None
    if replay_cache is not None:
        replay, rm = tv.load_cache_splits(Path(replay_cache), splits=replay_splits)
        r2.require(rm['labels_sha256'] == ref['inputs']['replay_labels_sha256']
                   and {k: v['sha256'] for k, v in rm['splits'].items()} == ref['inputs']['replay_cache_splits'],
                   'the replay cache is not the c8 run\'s')
    return cycle, replay


def subset_of(c: dict, cycle: dict, cycle_cache) -> tuple[dict, dict]:
    """The 1,000-row subset rows of the cycle train split, and its description."""
    from hybrid.h7c_train import select_rows
    from hybrid.h7r_d4 import digest

    train = cycle['train']
    ids = json.loads((Path(cycle_cache) / 'train_games.json').read_text(encoding='utf-8'))
    index = train['game'].tolist()
    rows = {gid: 0 for gid in ids}
    for g in index:
        rows[ids[g]] += 1
    s = c['subset']
    picked = d1.select_subset(ids, rows, s['seed'], [s['rows'], s['nested_rows']])
    chosen = set(picked[s['rows']]['games'])
    keep = [ids[g] in chosen for g in index]
    import torch
    data = select_rows(train, torch.tensor(keep, dtype=torch.bool))
    desc = {'games': len(picked[s['rows']]['games']), 'rows': picked[s['rows']]['rows'],
            'games_sha256': digest(picked[s['rows']]['games']),
            'nested': {'games': len(picked[s['nested_rows']]['games']), 'rows': picked[s['nested_rows']]['rows'],
                       'games_sha256': digest(picked[s['nested_rows']]['games'])}}
    r2.require(int(data['value'].shape[0]) == desc['rows'], 'subset rows differ from the game selection')
    return data, desc


def composition(data: dict) -> dict:
    kind, weight, ply = data['kind'], data['weight'].double(), data['ply'].long()
    out = {name: {'rows': int((kind == code).sum()), 'weight': float(weight[kind == code].sum())}
           for name, code in (('outcome', 0), ('proof', 1))}
    out['ply'] = {f'{lo}-{hi}': int(((ply >= lo) & (ply < hi)).sum()) for lo, hi in ((0, 20), (20, 40), (40, 80),
                                                                                        (80, 10_000))}
    return out


def load_c5(r2c: dict, checkpoint, device: str):
    checkpoint = Path(checkpoint)
    model, identity = r2.check_model('c5', r2c['models']['c5'], checkpoint, checkpoint.with_name('best.json'),
                                     r2c['frozen_state_sha256'], device)
    return model, identity


def parent_paths(c: dict, root: Path, seed: str) -> tuple[Path, Path]:
    base = root / c['output']['parents']
    return base / f'wide-{seed}.pt', base / f'wide-{seed}.json'


# --- prepare ---------------------------------------------------------------------------------------------------------

def prepare(args, log=print) -> dict:
    """Per seed the widened parent is PASS only if: the byte checks pass; the CPU float64 values (eval mode on the
    subset, the cycle val and the seeded replay val sample; one train-mode batch) are within the exact tolerance of
    c5's; every device (float32) output is finite; and the saved checkpoint loads back to the same state with
    identical device predictions. The device differences, their worst rows, the stage-by-stage diagnosis, the c5
    batch-split reference and the arithmetic settings are recorded, never gated."""
    import torch
    from hybrid.h6_value_train import state_sha256
    from hybrid.h7r_d4 import digest
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1(args.d1_contract, root, 'prepare')
    out = root / c['output']['prepare']
    r2.require(not out.exists() and not (root / c['output']['parents']).exists(),
               'a D1 prepare output exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'prepare needs a clean tracked tree')
    r2c, ref, _ = reference_inputs(c, root)
    cycle, replay = checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache, ('val',))
    subset, desc = subset_of(c, cycle, args.cycle_cache)
    log(f"[d1-prepare] subset {desc['games']} games / {desc['rows']} rows (nested {desc['nested']['rows']} rows)",
        flush=True)
    fl = d1.floors(subset)
    c5, c5_identity = load_c5(r2c, args.c5, args.device)
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
    log(f'[d1-prepare] c5 float64 CPU values ({time.monotonic() - started:.0f} s)', flush=True)
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
            'exact': all(v['finite'] and v['max'] <= tol for v in exact.values()),
            'device_finite': all(v['finite'] for v in device.values()),
            'reload': reload['state_equal'] and reload['value_channels'] == wide.config.value_channels
            and reload['prediction']['finite'] and reload['prediction']['max'] == 0.0,
        }
        ok = all(gates.values())
        widened[s] = {'init_seed': init, **checks, 'gates': gates, 'exact': exact, 'device': device,
                      'layers': layers, 'reload': reload,
                      'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta_path),
                      'passed': ok}
        worst = max((v['max'] for v in exact.values() if v['max'] is not None), default=None)
        dev = max((v['max'] for v in device.values() if v['max'] is not None), default=None)
        log(f"[d1-prepare] seed {s}: widened parent {'PASS' if ok else 'FAIL'} {gates} (float64 max |dv| {worst}, "
            f'device float32 max |dv| {dev}; {time.monotonic() - started:.0f} s)', flush=True)
    r2.require(git_state(root) == state, 'HEAD changed during prepare')
    record = {'format': PREP_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
              'subset': desc, 'composition': composition(subset), 'floors': fl, 'c5': c5_identity,
              'exact': {**ex, 'replay_val_sample_sha256': digest(sample.tolist()), 'train_mode_rows_used': train_rows},
              'device_check': {'batch_size': batch, 'settings': settings,
                               'c5_batch_split': {'batch_size': max(1, batch // 4), 'compare': split}},
              'widened': widened, 'passed': all(v['passed'] for v in widened.values())}
    publish_json(out, record)
    r2.require(record['passed'], 'a widened parent failed its preservation checks: no training')
    return record


def checked_prepare(c: dict, root: Path, eid: str) -> dict:
    from hybrid.h7c_nmask import committed
    path = root / c['output']['prepare']
    r2.require(path.is_file() and committed(path, root), f'{path}: the prepare record is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    r2.require(rec.get('format') == PREP_FORMAT and rec.get('experiment_sha256') == eid and rec.get('passed') is True,
               'the prepare record is not a passed record of this experiment')
    return rec


# --- train -----------------------------------------------------------------------------------------------------------

def train(args, log=print) -> dict:
    import torch
    from hybrid import h7c_train as tr
    from hybrid.h6_value_train import frozen_state, state_sha256
    from hybrid.h7r_ledger import publish_json
    from model.checkpoint import save_checkpoint

    root = Path(args.root)
    c, csha, eid = load_d1(args.d1_contract, root, 'train')
    r2.require(args.arm in ARMS and args.seed in SEEDS, 'arm narrow / wide and a D1 seed')
    run_dir = root / c['output']['run_dirs'][args.arm][args.seed]
    r2.require(not run_dir.exists(), f'{run_dir} exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'a D1 run needs a clean tracked tree')
    prep = checked_prepare(c, root, eid)
    r2c, ref, config = reference_inputs(c, root)
    cycle, replay = checked_caches(ref, args.cycle_cache, args.cycle_rows, args.replay_cache)
    subset, desc = subset_of(c, cycle, args.cycle_cache)
    r2.require(desc == prep['subset'], 'the subset is not the prepared one')
    c5, _ = load_c5(r2c, args.c5, args.device)
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
    log(f'[d1-train] {args.arm} seed {args.seed}: {n} subset rows, {total} steps, value_channels '
        f'{parent.config.value_channels}', flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.monotonic()
    with d1.fit_probe(subset, steps, args.device) as probe:
        result = tr.train_value(cfg, parent, {'train': subset, 'val': cycle['val']}, replay, log=log)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    r2.require(git_state(root) == state, 'HEAD changed during the D1 run')
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
        meta = {'stage': tr.VALUE_LOADER_STAGE, 'h7c_stage': tr.VALUE_STAGE, 'track': 'B', 'experiment': CONTRACT_ID,
                'arm': args.arm, 'seed': int(args.seed), 'which': name, 'model_config': asdict(model.config),
                'config': tr.config_dict(cfg), 'best': best, 'git': state,
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
    body = {'format': TRAIN_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state,
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


def run_records(c: dict, root: Path, eid: str) -> dict:
    recs = {}
    for arm in ARMS:
        for s in SEEDS:
            path = root / c['output']['run_dirs'][arm][s] / 'd1_train.json'
            rec = json.loads(path.read_text(encoding='utf-8'))
            r2.require(rec.get('format') == TRAIN_FORMAT and rec.get('experiment_sha256') == eid
                       and rec['arm'] == arm and rec['seed'] == s, f'{path}: not a run of this experiment')
            recs[(arm, s)] = rec
    for s in SEEDS:
        a, b = recs[('narrow', s)], recs[('wide', s)]
        r2.require(a['assignment'] == b['assignment'] and a['total_steps'] == b['total_steps']
                   and a['subset'] == b['subset'], f'seed {s}: the arms are not paired (assignment, steps, subset)')
    return recs


def fit_of(c: dict, prep: dict, recs: dict) -> dict:
    final = {arm: {s: recs[(arm, s)]['last_step']['fit']['weighted_mse'] for s in SEEDS} for arm in ARMS}
    return d1.fit_reading(final, prep['floors']['d4']['floor'], c['fit']['margin'], SEEDS)


def checked_fit(c: dict, root: Path, eid: str, prep: dict, recs: dict) -> dict:
    """The committed train-fit reading (fixed before any H-dev prediction), equal to its recomputation."""
    from hybrid.h7c_nmask import committed
    path = root / c['output']['fit']
    r2.require(path.is_file() and committed(path, root), f'{path}: the train-fit record is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    r2.require(rec.get('format') == FIT_FORMAT and rec.get('experiment_sha256') == eid,
               'the train-fit record is not of this experiment')
    reading = fit_of(c, prep, recs)
    r2.require(json.loads(json.dumps(reading)) == rec['reading'], 'the train-fit record differs from the run records')
    return reading


def fit(args, log=print) -> dict:
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    c, csha, eid = load_d1(args.d1_contract, root, 'train')
    out = root / c['output']['fit']
    r2.require(not out.exists(), f'{out} exists: never overwritten')
    prep = checked_prepare(c, root, eid)
    recs = run_records(c, root, eid)
    reading = fit_of(c, prep, recs)
    body = {'format': FIT_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'reading': reading,
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

def evaluate(args, log=print) -> dict:
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    from hybrid import h7r_d3 as d3
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    started = time.monotonic()
    c, csha, eid = load_d1(args.d1_contract, root, 'evaluate')
    out, preds_path = root / c['output']['result'], root / c['output']['predictions']
    r2.require(not out.exists() and not preds_path.exists(), 'a D1 evaluation output exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'the evaluation needs a clean tracked tree')
    prep = checked_prepare(c, root, eid)
    recs = run_records(c, root, eid)
    fit_read = checked_fit(c, root, eid, prep, recs)
    r2c = pinned(root, c['r2_contract'], 'R2 contract')
    eps = pinned(root, c['epsilon'], 'epsilon record')
    models = {'c5': load_c5(r2c, args.c5, args.device)[0]}
    for (arm, s), rec in recs.items():
        r2.require(rec['verdict'] == 'CANDIDATE' and rec['selected']['state'] is not None,
                   f'{arm} {s}: no selected checkpoint (NO_CANDIDATE is kept, not evaluated)')
        run_dir = root / c['output']['run_dirs'][arm][s]
        spec = {**r2c['models']['c5'], 'value_checkpoint_sha256': rec['selected']['state']['checkpoint_sha256'],
                'value_metadata_sha256': rec['selected']['state']['metadata_sha256']}
        models[f'{arm}{s}'], _ = r2.check_model(f'{arm}{s}', spec, run_dir / 'best.pt', run_dir / 'best.json',
                                                r2c['frozen_state_sha256'], args.device)
    log('[d1-eval] H-dev predictions of c5 and the six selected checkpoints', flush=True)
    _, data, info, hdev = r2cli.check_hdev(r2c, Path(args.cache), root)
    r2.require(hdev == eps['hdev'], 'the H-dev checks differ from the R2-0B record')
    route = ho.routes(data)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    pred = {name: r2.predict(models.pop(name), data, args.device) for name in list(models)}
    bins = r2c['metrics']['ply_bins']
    groups = r2.row_groups(route, data['kind'].tolist(), info['ply'], target, info['result'], info['label_kind'], bins)
    n_games = len(info['game_ids'])
    metrics = [r2c['metrics']['primary'], *r2c['metrics']['components']]
    sums = {(n, m['prediction'], m['group']): r2.game_sums(pred[n][m['prediction']], target, weight, info['game'],
                                                          groups[m['group']], n_games) for n in pred for m in metrics}
    value = lambda n, m: r2.point_metrics(sums[(n, m['prediction'], m['group'])])[m['aggregate']]  # noqa: E731
    boot = r2c['bootstrap']
    counts = r2.multiplicities(n_games, boot['samples'], boot['seed'])
    primary = metrics[0]
    key = lambda n: (n, primary['prediction'], primary['group'])  # noqa: E731
    kinds = {k: groups[f'network/{k}'] for k in ('outcome', 'proof')}
    paired, references = {}, {}
    for s in SEEDS:
        k, w = f'narrow{s}', f'wide{s}'
        diff = r2.difference(sums[key(w)], sums[key(k)], 'macro')
        paired[s] = {'control': value(k, primary), 'wide': value(w, primary), 'difference': diff, 'improvement': -diff,
                     **r2.bootstrap(counts, sums[key(w)], sums[key(k)], 'macro', boot['level'])}
        references[s] = {
            'wide_vs_c5': {'difference': r2.difference(sums[key(w)], sums[key('c5')], 'macro'),
                           **r2.bootstrap(counts, sums[key(w)], sums[key('c5')], 'macro', boot['level'])},
            'narrow_vs_c5': {'difference': r2.difference(sums[key(k)], sums[key('c5')], 'macro')},
            'components': [{'metric': m, 'difference': value(w, m) - value(k, m)} for m in metrics[1:]],
            'decomposition_kind': d3.decompose(pred[w][primary['prediction']], pred[k][primary['prediction']], target,
                                               weight, info['game'], groups[primary['group']], kinds, n_games, counts,
                                               boot['level'])['parts']}
    reading = d1.hdev_reading(paired, c['ruling']['epsilon'], {s: paired[s]['control'] for s in SEEDS}, SEEDS)
    r2.require(git_state(root) == state, 'HEAD changed during the evaluation')
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {n: {p: torch.tensor(v, dtype=torch.float64) for p, v in pr.items()} for n, pr in pred.items()}
    torch.save({'experiment_sha256': eid, 'models': tensors}, preds_path)
    body = {'format': EVAL_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state, 'hdev': hdev,
            'hdev_reading': reading, 'paired': paired, 'references': references,
            'train_fit_reading': fit_read, 'reference_note': c['reference'],
            'predictions': {'path': c['output']['predictions'], 'sha256': r2.file_sha256(preds_path)},
            'minutes': round((time.monotonic() - started) / 60, 2),
            'scope': 'the 1,000-row D1 diagnosis: no D4 candidate, strength, promotion or V9 ruling'}
    publish_json(out, body)
    return body


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
    args = parser.parse_args(argv)
    args.root = ROOT
    try:
        out = {'prepare': prepare, 'train': train, 'fit': fit, 'evaluate': evaluate}[args.command](args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    if args.command == 'prepare':
        print(f"[d1-prepare] DONE {'PASS' if out['passed'] else 'FAIL'}; subset {out['subset']['rows']} rows; floor "
              f"d4 {out['floors']['d4']['floor']:.6f}", flush=True)
    elif args.command == 'train':
        print(f"[d1-train] {out['arm']} {out['seed']} {out['verdict']}: last step {out['last_step']['step']} fit "
              f"{out['last_step']['fit']['weighted_mse']:.6f}; selected step {out['selected']['step']}; "
              f"{out['seconds']} s", flush=True)
    elif args.command == 'fit':
        print(f"[d1-fit] {out['reading']['category']}; wide - narrow {out['reading']['wide_minus_narrow']}", flush=True)
    else:
        print(f"[d1-eval] H-dev {out['hdev_reading']['label']}; train fit {out['train_fit_reading']['category']}",
              flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
