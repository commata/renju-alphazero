"""R2-D3-NO-OUTCOME-1: the cycle outcome ablation (mask, three training runs, H-dev reading). Each stage needs its own
approval in the fixed, committed D3 contract (``h7r-d3-contract-v1``); the R0 reproduction check must have passed.

    mask      (CPU) the outcome mask of the c8 cycle cache (``hybrid.h7r_d3_mask``): every cache row linked to its
              label row; outcome rows removed, proof rows kept; per split the rows and weights removed and kept. The
              cache must be the c8 run's (labels and split SHA-256 of the A1 reference record). Writes the mask
              tensor (runs/) and its manifest (the records folder); never over a file.
    train     (GPU) one seed: the control's own path, ``h7r_r2_seed.train`` (the c8 recipe, the A1 input checks,
              the data assignment, the run record), with the trainer's ``train_value`` given the mask and
              ``cycle_selection='original'`` for that one call. The four trainer sources R0 reproduced are not
              changed. The run is then compared with the committed record of the same seed's R2-0B control run:
              the data assignment (fingerprint and every field), the step count and the training rows, directly.
              The mask's kept weight is checked against every epoch the trainer recorded.
    evaluate  (GPU) c5, the three controls and the three D3 runs predicted together on the verified H-dev. The
              controls are first reproduced: their primary metric and the contract's component metrics must equal
              the committed R2-0B values within the contract's tolerance (an aggregate-metric tolerance, not a
              per-row one), else the reading is HELD and the differences recorded. The committed
              ``seed_predictions.pt`` is used for row alignment and as a per-row diagnostic only. Then the R2 rule
              unchanged: DEVELOPMENT_SIGNAL needs, for all three seeds, I_s = L(control) - L(D3) > epsilon and the
              paired 95% upper bound of L(D3) - L(control) below 0; a control seed range above epsilon holds the
              reading. c5, the components and the additive decomposition are reported as references only.

No stage overwrites a file, rules a strength, promotes, or starts the next stage.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d3_mask as dm  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d3-contract-v1'
CONTRACT_ID = 'R2-D3-NO-OUTCOME-1'
TRAIN_FORMAT = 'h7r-d3-train-v1'
EVAL_FORMAT = 'h7r-d3-eval-v1'
STAGES = ('mask', 'train', 'evaluate')
SEEDS = ('31001', '31002', '31003')
SOURCES = ('src/hybrid/h7c_train.py', 'scripts/h7c_train_value.py', 'scripts/h7r_r2_seed.py', 'scripts/h7r_d3_r0.py',
           'scripts/h7r_d3.py', 'src/hybrid/h7r_d3_mask.py', 'src/hybrid/h7r_d3.py', 'src/hybrid/h7r_r2.py',
           'scripts/h7r_r2.py', 'scripts/h7r_holdout.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'r2_contract', 'amendment', 'epsilon', 'r0', 'sources',
        'condition', 'controls', 'control_reproduction', 'ruling', 'reference', 'output'}
CONDITION = {'mask': dm.DEFINITION, 'cycle_selection': 'original',
             'loss': 'sum_i k_i w_i (v_i - t_i)^2 / sum_i w_i over the batch (original weights)'}


def is_ref(value) -> bool:
    return isinstance(value, dict) and set(value) == {'path', 'sha256'} and isinstance(value['path'], str) \
        and r2.is_sha(value['sha256'])


def pinned(root: Path, ref: dict, what: str) -> dict:
    path = root / ref['path']
    r2.require(path.is_file() and r2.file_sha256(path) == ref['sha256'], f"{ref['path']}: not the {what} D3 pins")
    return json.loads(path.read_text(encoding='utf-8'))


def git_state(root: Path) -> dict:
    def git(*a):
        return subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}


def load_d3(path, root: Path, stage: str, *, check_sources: bool = True) -> tuple[dict, str]:
    """The D3 contract, fixed, committed, ``stage`` approved, and every pinned value checked against its source."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no D3 contract')
    c = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the D3 contract keys differ from {sorted(KEYS)}')
    r2.require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
               f'{path}: not the fixed D3 contract')
    r2.require(isinstance(c['approved'], dict) and set(c['approved']) == set(STAGES)
               and all(type(v) is bool for v in c['approved'].values()), 'D3 approvals: one bool per stage')
    r2.require(c['approved'][stage] is True, f'the D3 {stage} stage is not approved')
    r2.require(committed(path, root), f'{path}: not committed')
    for key in ('r2_contract', 'amendment', 'epsilon'):
        r2.require(is_ref(c[key]), f'D3 {key}: a path and a full SHA-256 are required')
        pinned(root, c[key], key)
    r0 = c['r0']
    r2.require(isinstance(r0, dict) and set(r0) == {'result', 'reproduction'} and all(is_ref(r0[k]) for k in r0),
               'D3 r0: the R0 result and reproduction records are required')
    result = pinned(root, r0['result'], 'R0 result')
    pinned(root, r0['reproduction'], 'R0 reproduction record')
    r2.require(result.get('verdict', {}).get('verdict') == 'PASS'
               and result.get('reproduction', {}).get('sha256') == r0['reproduction']['sha256'],
               'R0 did not pass, or names another reproduction record: no D3 stage')
    r2.require(c['condition'] == CONDITION, 'the D3 condition differs from the tool')
    eps = pinned(root, c['epsilon'], 'epsilon record')
    r2.require(c['ruling'].get('epsilon') == eps['epsilon']['value'] and eps['epsilon']['status'] == 'COMPUTED',
               'the ruling epsilon is not the R2-0B epsilon')
    r2.require(set(c['controls']) == set(SEEDS), f'D3 controls: exactly the seeds {SEEDS}')
    for seed, spec in c['controls'].items():
        r2.require(set(spec) == {'run_record', 'run_dir', 'assignment_sha256', 'total_steps', 'best_step'}
                   and is_ref(spec['run_record']) and isinstance(spec['run_dir'], str)
                   and spec['run_dir'].startswith('runs/'), f'D3 control {seed}: malformed')
        rec = pinned(root, spec['run_record'], f'control {seed} run record')
        r2.require(rec['seed'] == int(seed) and rec['role'] == 'seed_spread' and rec['verdict'] == 'CANDIDATE'
                   and rec['assignment']['sha256'] == spec['assignment_sha256']
                   and rec['assignment']['total_steps'] == spec['total_steps'] and rec['best_step'] == spec['best_step'],
                   f'D3 control {seed}: the pinned values are not the ones its run record holds')
    cr = c['control_reproduction']
    r2.require(set(cr) == {'tolerance', 'metrics', 'note'} and isinstance(cr['tolerance'], float)
               and cr['tolerance'] > 0, 'D3 control_reproduction: tolerance, metrics and note')
    r2_contract = pinned(root, c['r2_contract'], 'R2 contract')
    metrics = r2_contract['metrics']
    r2.require(cr['metrics'] == [metrics['primary'], *metrics['components']],
               'the control reproduction metrics are not the R2 primary and component metrics')
    sources = c['sources']
    r2.require(isinstance(sources, dict) and set(sources) == set(SOURCES) and all(r2.is_sha(v) for v in sources.values()),
               'D3 sources are malformed')
    r0_contract = result.get('trainer_sources') or {}
    r2.require(all(sources[s] == r0_contract.get(s) for s in r0_contract),
               'the D3 trainer sources are not the ones R0 reproduced')
    if check_sources:
        changed = [s for s in SOURCES if r2.file_sha256(root / s) != sources[s]]
        r2.require(not changed, f'the checkout differs from the sources D3 was fixed for: {changed}')
    out = c['output']
    r2.require(isinstance(out, dict) and set(out) == {'mask', 'mask_manifest', 'run_dirs', 'result', 'predictions'}
               and out['mask'].startswith('runs/') and out['predictions'].startswith('runs/')
               and set(out['run_dirs']) == set(SEEDS) and all(v.startswith('runs/') for v in out['run_dirs'].values()),
               'D3 outputs: the mask tensor, the predictions and the run directories under runs/')
    return c, r2.file_sha256(path)


# --- mask ------------------------------------------------------------------------------------------------------------

def mask(args, log=print) -> dict:
    from hybrid.h7r_ledger import publish_json
    import h7r_r2_seed as seed

    root = Path(args.root)
    c, csha = load_d3(args.d3_contract, root, 'mask')
    out_mask, out_manifest = root / c['output']['mask'], root / c['output']['mask_manifest']
    r2.require(not out_mask.exists() and not out_manifest.exists(), 'a D3 mask output exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'the mask needs a clean tracked tree')
    amendment = pinned(root, c['amendment'], 'amendment')
    ref, _ = seed.reference(amendment, pinned(root, c['r2_contract'], 'R2 contract'), root)
    manifest = json.loads((Path(args.cycle_cache) / 'manifest.json').read_text(encoding='utf-8'))
    r2.require(manifest['labels_sha256'] == ref['inputs']['cycle_labels_sha256']
               and {k: v['sha256'] for k, v in manifest['splits'].items()} == ref['inputs']['cycle_cache_splits'],
               'the cycle cache is not the c8 run\'s (A1 reference record)')
    log('[d3-mask] linking every c8 cycle cache row to its label row', flush=True)
    keep, body = dm.build_outcome_mask(Path(args.cycle_rows), Path(args.cycle_cache))
    dm.save_mask(out_mask, keep, body)
    r2.require(git_state(root) == state, 'HEAD changed during the mask build')
    record = {**body, 'contract_sha256': csha, 'git': state, 'mask_file': {'path': c['output']['mask'],
                                                                        'sha256': r2.file_sha256(out_mask)},
              'value_rows_sha256': r2.file_sha256(args.cycle_rows)}
    publish_json(out_manifest, record)
    return record


def checked_mask(c: dict, root: Path) -> tuple[dict, dict]:
    """The committed mask manifest of this contract and the mask tensor it names."""
    from hybrid.h7c_nmask import committed

    path = root / c['output']['mask_manifest']
    r2.require(path.is_file() and committed(path, root), f'{path}: the mask manifest is not committed')
    body = json.loads(path.read_text(encoding='utf-8'))
    r2.require(body.get('format') == dm.MASK_FORMAT and body.get('definition') == dm.DEFINITION,
               'the mask manifest is not a D3 outcome mask')
    tensor = root / c['output']['mask']
    r2.require(r2.file_sha256(tensor) == body['mask_file']['sha256'], f'{tensor}: not the mask the manifest names')
    return dm.load_mask(tensor, body), body


# --- train -----------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def d3_trainer(keep: dict, body: dict):
    """``hybrid.h7c_train.train_value`` given the mask and the original selection, for one call only."""
    from hybrid import h7c_train as tr

    original, calls = tr.train_value, []

    def wrapped(cfg, parent, cycle, replay, log=print, **options):
        r2.require(not options, f'unexpected trainer options {sorted(options)}: D3 sets them')
        r2.require(not calls, 'train_value called twice in one D3 run')
        dm.check_against_cache(keep, cycle, body)
        calls.append(cfg.seed)
        return original(cfg, parent, cycle, replay, log=log, cycle_keep={s: keep[s] for s in ('train', 'val')},
                        cycle_selection='original')
    tr.train_value = wrapped
    try:
        yield calls
    finally:
        tr.train_value = original


def compare_with_control(run: dict, trainer: dict, control: dict, spec: dict, body: dict) -> dict:
    """Direct comparison with the committed control record of the same seed, and the mask in every epoch."""
    checks = {
        'assignment_sha256': run['assignment']['sha256'] == spec['assignment_sha256'] == control['assignment']['sha256'],
        'assignment': run['assignment'] == control['assignment'],
        'total_steps': run['assignment']['total_steps'] == spec['total_steps'] == trainer['total_steps'],
        'train_rows_per_epoch': trainer['train_rows_per_epoch'] == 2 * control['assignment']['n_cycle'],
        'config': run['config']['sha256'] == control['config']['sha256'],  # same seed: the same config.json bytes
        'selection_original': (trainer.get('selection') or {}).get('mode') == 'original',
        'mask_every_epoch': bool(trainer.get('mask', {}).get('epochs')) and all(
            abs(e['cycle_kept_weight'] - body['splits']['train']['kept_weight'])
            <= 1e-6 * max(1.0, body['splits']['train']['kept_weight']) for e in trainer['mask']['epochs']),
    }
    return {'checks': checks, 'passed': all(checks.values())}


def train(args, log=print) -> dict:
    from hybrid.h7r_ledger import publish_json
    import h7r_r2_seed as seed

    root = Path(args.root)
    started = time.monotonic()
    c, csha = load_d3(args.d3_contract, root, 'train')
    r2.require(args.seed in SEEDS, f'seed {args.seed}: not a D3 seed {SEEDS}')
    run_dir = root / c['output']['run_dirs'][args.seed]
    r2.require(not run_dir.exists(), f'{run_dir} exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'a D3 run needs a clean tracked tree')
    keep, body = checked_mask(c, root)
    spec = c['controls'][args.seed]
    control = pinned(root, spec['run_record'], f'control {args.seed} run record')
    log(f"[d3-train] seed {args.seed}: the c8 recipe, outcome rows masked "
        f"({body['splits']['train']['removed_rows']:,} of {body['splits']['train']['rows']:,} cycle train rows), "
        f'original selection', flush=True)
    ns = argparse.Namespace(contract=root / c['r2_contract']['path'], amendment=root / c['amendment']['path'],
                            seed=int(args.seed), cycle_cache=Path(args.cycle_cache), cycle_rows=Path(args.cycle_rows),
                            replay_cache=Path(args.replay_cache), parent=Path(args.parent), device=args.device,
                            out_dir=run_dir, reproduction=root / c['r0']['reproduction']['path'], reason=None,
                            allow_dirty=False, root=root)
    with d3_trainer(keep, body) as calls:
        run = seed.train(ns, log=log)
    r2.require(calls == [int(args.seed)], 'the trainer was not called once with the D3 options')
    r2.require(git_state(root) == state, 'HEAD changed during the D3 run')
    trainer = json.loads((run_dir / 'record.json').read_text(encoding='utf-8'))
    compared = compare_with_control(run, trainer, control, spec, body)
    record = {'format': TRAIN_FORMAT, 'contract_sha256': csha, 'git': state, 'seed': args.seed,
              'mask': {'keep_sha256': body['keep_sha256'], 'manifest_sha256': r2.file_sha256(
                  root / c['output']['mask_manifest'])},
              'run_record_sha256': r2.file_sha256(run_dir / 'r2_train.json'),
              'trainer_record_sha256': r2.file_sha256(run_dir / 'record.json'),
              'verdict': run['verdict'], 'best_step': run['best_step'], 'candidate': run.get('candidate'),
              'control': {'run_record_sha256': spec['run_record']['sha256'], 'best_step': spec['best_step']},
              'comparison': compared, 'selection': trainer.get('selection'),
              'minutes': round((time.monotonic() - started) / 60, 2)}
    publish_json(run_dir / 'd3_train.json', record)
    r2.require(compared['passed'], f"the D3 run is not comparable with its control: {compared['checks']}")
    return record


# --- evaluate --------------------------------------------------------------------------------------------------------

def control_reproduction(recomputed: dict, recorded: dict, tolerance: float) -> dict:
    """Per seed and metric |recomputed - committed R2-0B| <= tolerance; anything else holds the reading."""
    cells, failures = {}, []
    for seed in SEEDS:
        rows = []
        for i, (got, want) in enumerate(zip(recomputed[seed], recorded[seed])):
            diff = None if got is None or want is None else abs(got - want)
            ok = diff is not None and diff <= tolerance
            rows.append({'metric': i, 'recomputed': got, 'recorded': want, 'abs_difference': diff, 'within': ok})
            if not ok:
                failures.append(f'control {seed} metric {i}: |{got} - {want}| > {tolerance}')
        r2.require(len(recomputed[seed]) == len(recorded[seed]), f'control {seed}: metric count differs')
        cells[seed] = rows
    return {'tolerance': tolerance, 'seeds': cells, 'reproduced': not failures, 'failures': failures}


def reading(paired: dict, epsilon: float, control_primary: dict, reproduced: dict) -> dict:
    """The R2 rule: HOLD if the controls are not reproduced or their seed range exceeds epsilon; DEVELOPMENT_SIGNAL
    if for all three seeds I_s > epsilon and the 95% upper bound of D3 - control is below 0; else NOT_ESTABLISHED."""
    hold = []
    if not reproduced['reproduced']:
        hold.append('the controls are not reproduced within the contract tolerance')
    values = [control_primary[s] for s in SEEDS]
    spread = max(values) - min(values)
    if spread > epsilon:
        hold.append(f'the control seed range {spread:.6g} exceeds epsilon {epsilon:.6g}')
    per_seed = {}
    for s in SEEDS:
        p = paired[s]
        ci = p['ci95']
        per_seed[s] = {'improvement': p['improvement'], 'beyond_epsilon': p['improvement'] > epsilon,
                       'upper95_below_0': ci is not None and ci[1] < 0}
    signal = all(v['beyond_epsilon'] and v['upper95_below_0'] for v in per_seed.values())
    label = 'HOLD' if hold else ('DEVELOPMENT_SIGNAL' if signal else 'NOT_ESTABLISHED')
    return {'label': label, 'hold': hold, 'control_seed_range': spread, 'epsilon': epsilon, 'seeds': per_seed}


def analyse(pred: dict, route: list, kind: list, info: dict, target: list, weight: list, bins: list,
            control_spec: dict, recorded_seeds: dict, epsilon: float, counts, level: float) -> tuple:
    """The control reproduction, the paired comparison of each D3 run with its control, the references and the reading,
    from the predictions of c5, ``control<seed>`` and ``d3<seed>`` on the same H-dev rows."""
    from hybrid import h7r_d3 as d3

    groups = r2.row_groups(route, kind, info['ply'], target, info['result'], info['label_kind'], bins)
    n_games = len(info['game_ids'])
    metrics = control_spec['metrics']
    sums = {(n, m['prediction'], m['group']): r2.game_sums(pred[n][m['prediction']], target, weight, info['game'],
                                                          groups[m['group']], n_games)
            for n in pred for m in metrics}
    value = lambda n, m: r2.point_metrics(sums[(n, m['prediction'], m['group'])])[m['aggregate']]  # noqa: E731
    recomputed = {s: [value(f'control{s}', m) for m in metrics] for s in SEEDS}
    recorded = {s: [recorded_seeds[s]['primary'], *recorded_seeds[s]['components']] for s in SEEDS}
    reproduced = control_reproduction(recomputed, recorded, control_spec['tolerance'])
    primary = metrics[0]
    key = lambda n: (n, primary['prediction'], primary['group'])  # noqa: E731
    rows = groups[primary['group']]
    kinds = {k: groups[f'network/{k}'] for k in ('outcome', 'proof')}
    plies = {b: set(groups[f'network/ply:{b}']) for b in r2.ply_bin_names(bins)}
    kind_ply = {f'{k}|{b}': [i for i in part if i in plies[b]] for k, part in kinds.items() for b in plies}
    paired, references = {}, {}
    for s in SEEDS:
        k, d = f'control{s}', f'd3{s}'
        diff = r2.difference(sums[key(d)], sums[key(k)], 'macro')
        paired[s] = {'control': value(k, primary), 'd3': value(d, primary), 'difference': diff,
                     'improvement': -diff, **r2.bootstrap(counts, sums[key(d)], sums[key(k)], 'macro', level)}
        p = primary['prediction']
        references[s] = {
            'd3_vs_c5': {'difference': r2.difference(sums[key(d)], sums[key('c5')], 'macro'),
                         **r2.bootstrap(counts, sums[key(d)], sums[key('c5')], 'macro', level)},
            'components': [{'metric': m, 'difference': value(d, m) - value(k, m)} for m in metrics[1:]],
            'decomposition_kind': d3.decompose(pred[d][p], pred[k][p], target, weight, info['game'], rows, kinds,
                                               n_games, counts, level)['parts'],
            'decomposition_kind_ply': d3.decompose(pred[d][p], pred[k][p], target, weight, info['game'], rows,
                                                   kind_ply, n_games, counts, level)['parts']}
    result = reading(paired, epsilon, {s: recomputed[s][0] for s in SEEDS}, reproduced)
    return reproduced, paired, references, result


def evaluate(args, log=print) -> dict:
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    import h7r_r2_seed as seed
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    started = time.monotonic()
    c, csha = load_d3(args.d3_contract, root, 'evaluate')
    out, preds_path = root / c['output']['result'], root / c['output']['predictions']
    r2.require(not out.exists() and not preds_path.exists(), 'a D3 evaluation output exists: never overwritten')
    state = git_state(root)
    r2.require(not state['dirty'], 'the evaluation needs a clean tracked tree')
    contract, contract_sha, amendment, amendment_sha = seed.approved(argparse.Namespace(
        contract=root / c['r2_contract']['path'], amendment=root / c['amendment']['path']), root)
    eps = pinned(root, c['epsilon'], 'epsilon record')
    epsilon = c['ruling']['epsilon']
    r0_sha = c['r0']['reproduction']['sha256']
    r2_0b_sha = r2.file_sha256(root / 'docs/h7r/r2/R2-DIAG-1/r2_0b/reproduction.json')

    models = {}
    c5_ckpt, c5_meta = args.c5.split(',')
    models['c5'], _ = r2.check_model('c5', contract['models']['c5'], Path(c5_ckpt), Path(c5_meta),
                                     contract['frozen_state_sha256'], args.device)
    for s in SEEDS:
        spec = c['controls'][s]
        crun = root / spec['run_dir']
        r2.require(r2.file_sha256(crun / 'r2_train.json') == spec['run_record']['sha256'],
                   f'control {s}: the local run record is not the committed one')
        rec, _ = seed.checked_training_run(crun, contract, contract_sha, amendment, amendment_sha, root, 'seed_spread',
                                           r2_0b_sha)
        models[f'control{s}'], _ = r2.check_model(f'control{s}', {
            **contract['models']['c5'], 'value_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
            'value_metadata_sha256': rec['candidate']['metadata_sha256']}, crun / 'best.pt', crun / 'best.json',
            contract['frozen_state_sha256'], args.device)
        drun = root / c['output']['run_dirs'][s]
        d3rec = json.loads((drun / 'd3_train.json').read_text(encoding='utf-8'))
        r2.require(d3rec.get('format') == TRAIN_FORMAT and d3rec.get('contract_sha256') == csha
                   and d3rec['comparison']['passed'] and d3rec['verdict'] == 'CANDIDATE'
                   and d3rec['run_record_sha256'] == r2.file_sha256(drun / 'r2_train.json'),
                   f'D3 run {s}: not a comparable candidate of this contract')
        rec, _ = seed.checked_training_run(drun, contract, contract_sha, amendment, amendment_sha, root, 'seed_spread',
                                           r0_sha)
        models[f'd3{s}'], _ = r2.check_model(f'd3{s}', {
            **contract['models']['c5'], 'value_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
            'value_metadata_sha256': rec['candidate']['metadata_sha256']}, drun / 'best.pt', drun / 'best.json',
            contract['frozen_state_sha256'], args.device)

    log('[d3-eval] H-dev predictions of c5, the three controls and the three D3 runs', flush=True)
    _, data, info, hdev = r2cli.check_hdev(contract, Path(args.cache), root)
    r2.require(hdev == eps['hdev'], 'the H-dev checks differ from the R2-0B record')
    route = ho.routes(data)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    pred = {name: r2.predict(models.pop(name), data, args.device) for name in list(models)}
    stored = torch.load(root / eps['predictions']['path'], map_location='cpu', weights_only=True)
    r2.require(r2.file_sha256(root / eps['predictions']['path']) == eps['predictions']['sha256']
               and stored['game'].tolist() == list(info['game']) and stored['ply'].tolist() == list(info['ply'])
               and stored['route'] == [r == 'network' for r in route], 'seed_predictions.pt rows are not the H-dev rows')
    diagnostic = {s: {p: float((torch.tensor(pred[f'control{s}'][p], dtype=torch.float64)
                                - stored['models'][f'seed{s}'][p]).abs().max()) for p in r2.PREDICTIONS} for s in SEEDS}
    boot = contract['bootstrap']
    counts = r2.multiplicities(len(info['game_ids']), boot['samples'], boot['seed'])
    reproduced, paired, references, result = analyse(
        pred, route, data['kind'].tolist(), info, target, weight, contract['metrics']['ply_bins'],
        c['control_reproduction'], eps['seeds'], epsilon, counts, boot['level'])
    r2.require(git_state(root) == state, 'HEAD changed during the evaluation')
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'contract_sha256': csha, 'game': stored['game'], 'ply': stored['ply'],
                'models': {n: {p: torch.tensor(v, dtype=torch.float64) for p, v in pr.items()} for n, pr in pred.items()}},
               preds_path)
    body = {'format': EVAL_FORMAT, 'contract_sha256': csha, 'git': state, 'hdev': hdev,
            'control_reproduction': reproduced, 'row_diagnostic_max_abs_vs_r2_0b': diagnostic,
            'paired': paired, 'reading': result, 'references': references,
            'reference_note': c['reference'],
            'predictions': {'path': c['output']['predictions'], 'sha256': r2.file_sha256(preds_path)},
            'minutes': round((time.monotonic() - started) / 60, 2),
            'scope': 'development diagnosis of R2-D3-NO-OUTCOME-1; no strength, promotion, V9 or D4 ruling'}
    publish_json(out, body)
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mask')
    t = sub.add_parser('train')
    e = sub.add_parser('evaluate')
    for s in (m, t, e):
        s.add_argument('--d3-contract', type=Path, required=True)
    for s in (m, t):
        s.add_argument('--cycle-cache', required=True)
        s.add_argument('--cycle-rows', required=True)
    t.add_argument('--seed', required=True, choices=SEEDS)
    t.add_argument('--replay-cache', required=True)
    t.add_argument('--parent', required=True, help='c5 value checkpoint (its best.json beside it)')
    e.add_argument('--cache', required=True, help='the H-dev cache')
    e.add_argument('--c5', required=True, help='c5 value checkpoint,metadata')
    for s in (t, e):
        s.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    args.root = ROOT
    try:
        out = {'mask': mask, 'train': train, 'evaluate': evaluate}[args.command](args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    if args.command == 'mask':
        t = out['splits']['train']
        print(f"[d3-mask] DONE train {t['removed_rows']:,} removed / {t['kept_rows']:,} kept; keep "
              f"{out['keep_sha256'][:12]}", flush=True)
        return 0
    if args.command == 'train':
        print(f"[d3-train] seed {out['seed']} {out['verdict']} best step {out['best_step']}; control comparison "
              f"{out['comparison']['checks']}", flush=True)
        return 0 if out['verdict'] == 'CANDIDATE' else 1
    r = out['reading']
    print(f"[d3-eval] {r['label']} (hold {r['hold']}); control reproduced {out['control_reproduction']['reproduced']}",
          flush=True)
    for s, p in out['paired'].items():
        print(f"  seed {s}: I = {p['improvement']:+.6f}, D3 - control 95% {p['ci95']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
