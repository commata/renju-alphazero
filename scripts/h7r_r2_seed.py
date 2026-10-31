"""H7-R R2-0B tools (amendment ``docs/h7r/r2/R2-DIAG-1-A1.json``): one training run of the c8 recipe with only the
training seed replaced, and the reproduction check of the c8 run. Both refuse to run until the amendment approves R2-0B.

    train      checks that every input is the c8 run's (committed ``docs/h7c/c8/value_record.json``: parent c5, cycle
               cache and rows, replay anchor cache, config), writes the config with the seed replaced, records the
               data assignment the seed fixes (``h7r_r2.data_assignment``: replay draws, row order, D4 symmetries),
               trains through ``h7c_train_value.run`` itself (the c8 code path, without the recipe decision whose
               config hash pins the c8 seed) and records the curve, the selected step and the state content hashes.
               H-dev is not read. A NO_CANDIDATE run keeps its record (exit 1): a failed seed is never dropped.
    reproduce  the run with the reproduction seed against the reference c8 run: checkpoint file hashes, state content
               hashes, data assignment, selected step and curve, H-dev predictions (max / mean |dv|, raw and D4) and
               the primary metric; ``h7r_r2.reproduction_verdict`` with the amendment's tolerances. NOT_REPRODUCED
               exits 1: epsilon and every later reading stop.
    epsilon    the three seed runs (exactly the amendment's seeds, each a candidate) on H-dev: their primary metric,
               their paired differences against c5 and c8, epsilon from the committed R2-0A c8:c5 interval and the
               seed range, and the R2-0A comparisons read against it (a separate record; R2-0A is not rewritten).

The order is enforced by the tools: the reproduction run first (``train --seed 20261025``), then ``reproduce``; a
training-seed run needs the passed reproduction record (``--reproduction``), ``epsilon`` needs it and the three runs.
A failed seed is never dropped or replaced (the seeds are fixed); a fixed and repeated run keeps the earlier output
and records its reason (``--reason``).

    python -u scripts/h7r_r2_seed.py train --contract docs/h7r/r2/R2-DIAG-1.contract.json \\
        --amendment docs/h7r/r2/R2-DIAG-1-A1.json --seed 20261025 --cycle-cache runs/h7c/c8/value_cache \\
        --cycle-rows runs/h7c/c8/value_rows.jsonl.gz --replay-cache data/tactical/h6/value_cache_v3 \\
        --parent runs/h7c/c5/value/best.pt --device cuda --out-dir runs/h7r/R2-DIAG-1/seed-20261025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7r_r2 as r2  # noqa: E402

TOOL_SOURCES = ('scripts/h7r_r2_seed.py', 'scripts/h7r_r2.py', 'src/hybrid/h7r_r2.py', 'scripts/h7c_train_value.py',
                'src/hybrid/h7c_train.py')
EPSILON_FORMAT = 'h7r-r2-epsilon-v1'
SAME_INPUTS = ('parent_checkpoint_sha256', 'parent_metadata_sha256', 'cycle_labels_sha256', 'replay_labels_sha256',
               'cycle_cache_splits', 'replay_cache_splits')


def approved(args, root: Path) -> tuple[dict, str, dict, str]:
    """Contract and amendment fixed and committed, R2-0B approved in the amendment (``allow_unapproved`` and
    ``allow_draft``: tests only, no command-line options)."""
    import h7r_r2 as cli
    from hybrid.h7c_nmask import committed

    contract, contract_sha = cli.approved_contract(args, root)
    amendment, amendment_sha = r2.load_amendment(args.amendment, contract, contract_sha)
    if not getattr(args, 'allow_draft', False):
        r2.require(committed(args.amendment, root), f'{args.amendment}: not committed')
    if not getattr(args, 'allow_unapproved', False):
        r2.require(amendment['r2_0b']['approved'], 'R2-0B is not approved in the amendment: no training run')
    return contract, contract_sha, amendment, amendment_sha


def environment(device: str) -> dict:
    import platform

    import torch
    out = {'torch': torch.__version__, 'python': platform.python_version(), 'device': device,
           'cuda': torch.version.cuda}
    if device.startswith('cuda') and torch.cuda.is_available():
        out['gpu'] = torch.cuda.get_device_name(0)
    return out


def passed_reproduction(path, contract_sha: str, amendment: dict, amendment_sha: str) -> tuple[dict, str]:
    """The reproduction record, consistent and passed: this contract and amendment, its verdict recomputed from its
    stored checks with the amendment's tolerances, its re-run checkpoint still the file it compared."""
    r2.require(path is not None, 'the passed reproduction record is required first (reproduce)')
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8'))
    r2.require(record.get('format') == r2.REPRODUCE_FORMAT and record.get('contract_sha256') == contract_sha
               and record.get('amendment_sha256') == amendment_sha, f'{path}: not this contract / amendment\'s record')
    again = r2.reproduction_verdict(record.get('checks') or {}, amendment['r2_0b']['reproduction']['tolerance'])
    r2.require(again == record.get('verdict'), f'{path}: the stored verdict does not follow from its checks')
    r2.require(again['verdict'] in r2.REPRODUCED, f"{path}: {again['verdict']} {again['reasons']}: no seed run, "
                                                   'no epsilon')
    run_dir = Path(record['run_dir'])
    r2.require(r2.file_sha256(run_dir / 'best.pt') == record['checkpoint_file_sha256']['rerun'],
               f'{run_dir}: the reproduction checkpoint changed')
    return record, r2.file_sha256(path)


def reference(amendment: dict, contract: dict, root: Path) -> tuple[dict, dict]:
    """The committed c8 record and config the amendment pins; the record's candidate is the contract's c8."""
    b = amendment['r2_0b']
    paths = {k: root / b[k]['path'] for k in ('reference_record', 'config')}
    for k, path in paths.items():
        r2.require(path.is_file() and r2.file_sha256(path) == b[k]['sha256'], f'{path}: not the {k} the amendment pins')
    record = json.loads(paths['reference_record'].read_text(encoding='utf-8'))
    r2.require(record.get('format') == 'h7c-value-train-v1' and record.get('verdict') == 'CANDIDATE',
               'the reference record is not a value candidate run')
    r2.require(record['candidate']['checkpoint_sha256'] == contract['models']['c8']['value_checkpoint_sha256']
               and record['candidate']['metadata_sha256'] == contract['models']['c8']['value_metadata_sha256'],
               'the reference record is not the contract\'s c8 run')
    r2.require(record['inputs']['config_sha256'] == b['config']['sha256'], 'the reference run used another config')
    config = json.loads(paths['config'].read_text(encoding='utf-8'))
    return record, config


def train(args, log=print) -> dict:
    import h7c_labels as lab
    import h7c_train_value as tv
    from hybrid.h6_value_train import frozen_state, state_sha256
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    contract, contract_sha, amendment, amendment_sha = approved(args, root)
    b = amendment['r2_0b']
    r2.require(args.seed in b['training_seeds'] + [b['reproduction_seed']],
               f"seed {args.seed}: not a training seed {b['training_seeds']} nor the reproduction seed")
    reproduction_sha = None
    if args.seed in b['training_seeds']:  # only after the reproduction check passed
        _, reproduction_sha = passed_reproduction(getattr(args, 'reproduction', None), contract_sha, amendment,
                                                  amendment_sha)
    else:
        r2.require(getattr(args, 'reproduction', None) is None, 'the reproduction run takes no reproduction record')
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    r2.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: a seed run must name a clean commit')
    out = Path(args.out_dir)
    r2.require(lab._under_data_or_runs(out) and not out.exists(), f'{out}: must be new and under data/ or runs/')
    record_ref, config = reference(amendment, contract, root)
    inputs_ref = record_ref['inputs']
    parent = Path(args.parent)
    parent_meta = parent.with_name('best.json')
    for key, path in (('parent_checkpoint_sha256', parent), ('parent_metadata_sha256', parent_meta)):
        r2.require(r2.file_sha256(path) == inputs_ref[key], f'{path}: not the reference run\'s parent')
    r2.require(inputs_ref['parent_checkpoint_sha256'] == contract['models']['c5']['value_checkpoint_sha256'],
               'the reference parent is not the contract\'s c5')
    tv.check_cycle_cache(Path(args.cycle_cache), record_ref['cycle'], inputs_ref['selfplay_run_seed'],
                         rows=Path(args.cycle_rows))
    manifests = {}
    for name, cache, labels_key, splits_key in (('cycle', args.cycle_cache, 'cycle_labels_sha256', 'cycle_cache_splits'),
                                                ('replay', args.replay_cache, 'replay_labels_sha256',
                                                 'replay_cache_splits')):
        m = json.loads((Path(cache) / 'manifest.json').read_text(encoding='utf-8'))
        r2.require(m['labels_sha256'] == inputs_ref[labels_key], f'{cache}: not the reference run\'s {name} labels')
        r2.require({k: v['sha256'] for k, v in m['splits'].items()} == inputs_ref[splits_key],
                   f'{cache}: not the reference run\'s {name} cache')
        manifests[name] = m
    cfg_ref = {k: v for k, v in record_ref['config'].items() if k != 'device'}
    from hybrid import h7c_train as tr
    r2.require({k: v for k, v in tr.config_dict(tr.ValueConfig.from_dict(config)).items() if k != 'device'} == cfg_ref,
               'the pinned config is not the reference run\'s config')
    n_cycle = manifests['cycle']['splits']['train']['states']
    n_replay = manifests['replay']['splits']['train']['states']
    assignment = r2.data_assignment(n_cycle, n_replay, args.seed, config['batch_size'], config['epochs'],
                                    config.get('augment_d4', True))

    out.mkdir(parents=True)
    run_config = out / 'config.json'
    run_config.write_text(json.dumps({**config, 'seed': args.seed}, indent=1) + '\n', encoding='utf-8')
    log(f"[h7r-r2-seed] seed {args.seed}: c8 recipe, {n_cycle:,} cycle / {n_replay:,} replay train rows, "
        f"{assignment['total_steps']} steps", flush=True)
    ns = argparse.Namespace(cycle=record_ref['cycle'], config=run_config, cycle_cache=Path(args.cycle_cache),
                            cycle_rows=Path(args.cycle_rows), replay_cache=Path(args.replay_cache), parent=parent,
                            device=args.device, out_dir=out, recipe_decision=None, allow_dirty=args.allow_dirty,
                            allow_stand_in=getattr(args, 'allow_stand_in', False))
    trained = tv.run(ns, log=log)
    for key in SAME_INPUTS:
        r2.require(trained['inputs'][key] == inputs_ref[key], f'the run\'s {key} differs from the reference run')
    r2.require({k: v for k, v in trained['config'].items() if k not in ('device', 'seed')}
               == {k: v for k, v in cfg_ref.items() if k != 'seed'} and trained['config']['seed'] == args.seed,
               'the run\'s config differs from the reference run beyond the seed')
    r2.require(trained['train_rows_per_epoch'] == 2 * n_cycle and trained['total_steps'] == assignment['total_steps'],
               'the trainer\'s sizes differ from the data assignment')
    states = None
    if trained['verdict'] == 'CANDIDATE':
        model, _ = r2.load_value_model(out / 'best.pt', out / 'best.json', 'cpu')
        state = model.state_dict()
        states = {'state_sha256': state_sha256(state),
                  'value_head_sha256': state_sha256({k: v for k, v in state.items() if k.startswith('value_head.')}),
                  'frozen_state_sha256': state_sha256(frozen_state(model))}
        r2.require(states['frozen_state_sha256'] == contract['frozen_state_sha256'], 'the trunk changed')
    body = {'format': r2.TRAIN_FORMAT, 'contract_sha256': contract_sha, 'amendment_sha256': amendment_sha,
            'role': 'reproduction' if args.seed == b['reproduction_seed'] else 'seed_spread', 'seed': args.seed,
            'reproduction_sha256': reproduction_sha, 'reason': getattr(args, 'reason', None),
            'tolerance_source': 'amendment (the parent contract leaves it null)',
            'git': git, 'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES},
            'environment': environment(args.device),
            'reference_record_sha256': b['reference_record']['sha256'],
            'config': {'path': str(run_config), 'sha256': r2.file_sha256(run_config)},
            'assignment': assignment, 'trainer_record_sha256': r2.file_sha256(out / 'record.json'),
            'verdict': trained['verdict'], 'best_step': (trained['best'] or {}).get('step'),
            'curve': [{'step': e['step'], 'objective': e['objective'], 'eligible': e['eligible']}
                      for e in trained['evaluations']],
            'candidate': trained.get('candidate'), 'states': states, 'hdev_used': False,
            'minutes': round((time.monotonic() - started) / 60, 2)}
    from hybrid.h7r_ledger import publish_json
    publish_json(out / 'r2_train.json', body)
    return body


def reproduce(args, log=print) -> dict:
    import torch

    import h7r_holdout as ho
    import h7r_r2 as cli
    from hybrid.h6_value_train import state_sha256
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    contract, contract_sha, amendment, amendment_sha = approved(args, root)
    b = amendment['r2_0b']
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    r2.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: R2 records must name a clean commit')
    r2.require(not Path(args.output).exists(), f'{args.output} exists')
    record_ref, config = reference(amendment, contract, root)
    run_dir = Path(args.run_dir)
    rerun = json.loads((run_dir / 'r2_train.json').read_text(encoding='utf-8'))
    r2.require(rerun.get('format') == r2.TRAIN_FORMAT and rerun['role'] == 'reproduction'
               and rerun['seed'] == b['reproduction_seed'] == record_ref['config']['seed'],
               'the run is not the reproduction run of the reference seed')
    r2.require(rerun['contract_sha256'] == contract_sha and rerun['amendment_sha256'] == amendment_sha
               and rerun['reference_record_sha256'] == b['reference_record']['sha256'],
               'the run names another contract, amendment or reference')
    trainer = json.loads((run_dir / 'record.json').read_text(encoding='utf-8'))
    r2.require(r2.file_sha256(run_dir / 'record.json') == rerun['trainer_record_sha256'], 'the trainer record changed')
    for key in SAME_INPUTS:
        r2.require(trainer['inputs'][key] == record_ref['inputs'][key], f'the run\'s {key} differs from the reference')

    # the reference run's assignment: its config and the (identical) cache sizes the re-run recorded
    sizes = rerun['assignment']
    r2.require(record_ref['train_rows_per_epoch'] == 2 * sizes['n_cycle'], 'the reference cycle size differs')
    ref_assignment = r2.data_assignment(sizes['n_cycle'], sizes['n_replay'], record_ref['config']['seed'],
                                        record_ref['config']['batch_size'], record_ref['config']['epochs'],
                                        record_ref['config'].get('augment_d4', True))
    r2.require(ref_assignment['total_steps'] == record_ref['total_steps'], 'the reference step count differs')

    log('[h7r-r2-seed] models: reference c8 and the reproduction run', flush=True)
    models = cli.parse_models([f'c8={args.reference_model}'], {'models': {'c8': None}})
    ref_model, ref_identity = r2.check_model('c8', contract['models']['c8'], *models['c8'],
                                             contract['frozen_state_sha256'], args.device)
    cand = rerun.get('candidate') or {}
    r2.require(rerun['verdict'] == 'CANDIDATE', 'the reproduction run has no candidate: NOT_REPRODUCED')
    spec = {'value_checkpoint_sha256': cand['checkpoint_sha256'], 'value_metadata_sha256': cand['metadata_sha256'],
            'policy_checkpoint_sha256': contract['models']['c8']['policy_checkpoint_sha256'],
            'policy_metadata_sha256': contract['models']['c8']['policy_metadata_sha256']}
    run_model, run_identity = r2.check_model('rerun', spec, run_dir / 'best.pt', run_dir / 'best.json',
                                             contract['frozen_state_sha256'], args.device)
    state = {n: state_sha256(m.state_dict()) for n, m in (('reference', ref_model), ('rerun', run_model))}
    r2.require(state['rerun'] == rerun['states']['state_sha256'], 'the run checkpoint is not the one it recorded')

    log('[h7r-r2-seed] H-dev predictions of both models', flush=True)
    _, data, info, hdev_checks = cli.check_hdev(contract, Path(args.cache), root)
    route = ho.routes(data)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    groups = r2.row_groups(route, data['kind'].tolist(), info['ply'], target, info['result'], info['label_kind'],
                           contract['metrics']['ply_bins'])
    pred = {'reference': r2.predict(ref_model, data, args.device), 'rerun': r2.predict(run_model, data, args.device)}
    delta = {p: torch.tensor(pred['rerun'][p], dtype=torch.float64) - torch.tensor(pred['reference'][p],
                                                                                    dtype=torch.float64)
             for p in r2.PREDICTIONS}
    primary = contract['metrics']['primary']
    loss = {n: r2.point_metrics(r2.game_sums(pred[n][primary['prediction']], target, weight, info['game'],
                                             groups[primary['group']], len(info['game_ids'])))['macro']
            for n in pred}
    over = b['reproduction']['max_abs_prediction_over']
    checks = {'state_sha256': (state['reference'], state['rerun']),
              'assignment': (ref_assignment['sha256'], sizes['sha256']),
              'best_step': ((record_ref.get('best') or {}).get('step'), rerun['best_step']),
              'max_abs_prediction': max(float(delta[p].abs().max()) for p in over),
              'abs_primary': abs(loss['rerun'] - loss['reference'])}
    verdict = r2.reproduction_verdict(checks, b['reproduction']['tolerance'])
    ref_curve = {e['step']: e['objective'] for e in record_ref['evaluations']}
    run_curve = {e['step']: e['objective'] for e in rerun['curve']}
    out = {'format': r2.REPRODUCE_FORMAT, 'contract_sha256': contract_sha, 'amendment_sha256': amendment_sha,
           'git': git, 'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES}, 'hdev': hdev_checks,
           'checkpoint_file_sha256': {'reference': ref_identity['value_checkpoint_sha256'],
                                      'rerun': run_identity['value_checkpoint_sha256']},
           'state_sha256': state,
           'value_head_sha256': {'reference': ref_identity['value_head_sha256'],
                                 'rerun': run_identity['value_head_sha256']},
           'assignment': {'reference': ref_assignment, 'rerun_sha256': sizes['sha256'],
                          'compare': r2.compare_assignments(ref_assignment, sizes),
                          'reference_source': {
                              'kind': 'reconstructed: the c8 run stored no fingerprint',
                              'record_sha256': b['reference_record']['sha256'], 'config_sha256': b['config']['sha256'],
                              'cache_splits': {'cycle': record_ref['inputs']['cycle_cache_splits'],
                                               'replay': record_ref['inputs']['replay_cache_splits']},
                              'trainer_sha256': {s: r2.file_sha256(ROOT / s) for s in
                                                 ('src/hybrid/h7c_train.py', 'scripts/h7c_train_value.py')},
                              'reference_trainer_commit': record_ref['git']['commit']}},
           'best_step': {'reference': checks['best_step'][0], 'rerun': checks['best_step'][1]},
           'curve': {'steps_equal': sorted(ref_curve) == sorted(run_curve),
                     'max_abs_objective_difference': max((abs(ref_curve[s] - run_curve[s]) for s in ref_curve
                                                          if s in run_curve), default=None)},
           'predictions': {p: {'max_abs': float(delta[p].abs().max()), 'mean_abs': float(delta[p].abs().mean())}
                           for p in r2.PREDICTIONS},
           'primary': {'metric': primary, **loss, 'abs_difference': checks['abs_primary']},
           'checks': {k: list(v) if isinstance(v, tuple) else v for k, v in checks.items()},
           'tolerance': b['reproduction']['tolerance'],
           'tolerance_source': 'amendment R2-DIAG-1-A1 (the parent contract leaves it null)',
           'run_dir': str(run_dir), 'environment': environment(args.device), 'verdict': verdict,
           'minutes': round((time.monotonic() - started) / 60, 2),
           'scope': 'a check that finds reproduction problems; one re-run does not estimate GPU nondeterminism'}
    from hybrid.h7r_ledger import publish_json
    publish_json(args.output, out)
    return out


REPLAY_MATCH = 1e-5  # c5 / c8 recomputed on H-dev must agree with the committed R2-0A metric to this


def beyond(difference, ci, eps) -> bool:
    """A difference whose interval excludes 0 on its side and whose size exceeds epsilon (descriptive)."""
    if difference is None or ci is None or eps is None:
        return False
    return (difference < -eps and ci[1] < 0) or (difference > eps and ci[0] > 0)


def epsilon_command(args, log=print) -> dict:
    import torch

    import h7r_holdout as ho
    import h7r_r2 as cli
    from hybrid.h6_value_train import state_sha256
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    contract, contract_sha, amendment, amendment_sha = approved(args, root)
    b = amendment['r2_0b']
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    r2.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: R2 records must name a clean commit')
    for path in [args.output] + ([args.predictions] if args.predictions else []):
        r2.require(not Path(path).exists(), f'{path} exists')
    if args.predictions:
        import h7c_labels as lab
        r2.require(lab._under_data_or_runs(Path(args.predictions)), 'predictions go under data/ or runs/')
    reproduction, reproduction_sha = passed_reproduction(args.reproduction, contract_sha, amendment, amendment_sha)
    pin = amendment['x1']['inputs']['result']
    r2.require(r2.file_sha256(args.result) == pin['sha256'], f'{args.result}: not the committed R2-0A record')
    result = json.loads(Path(args.result).read_text(encoding='utf-8'))
    r2.require(result['contract_sha256'] == contract_sha and result['stage'] == 'R2-0A', 'not this contract\'s R2-0A')

    runs = {}
    for run_dir in map(Path, args.run_dir or []):
        rec = json.loads((run_dir / 'r2_train.json').read_text(encoding='utf-8'))
        r2.require(rec.get('format') == r2.TRAIN_FORMAT and rec.get('role') == 'seed_spread',
                   f'{run_dir}: not a seed-spread run')
        r2.require(rec['seed'] not in runs, f"seed {rec['seed']} given twice")
        r2.require(rec['contract_sha256'] == contract_sha and rec['amendment_sha256'] == amendment_sha
                   and rec['reproduction_sha256'] == reproduction_sha,
                   f'{run_dir}: another contract, amendment or reproduction record')
        r2.require(rec['verdict'] == 'CANDIDATE', f"seed {rec['seed']}: {rec['verdict']}: stop (never dropped)")
        runs[rec['seed']] = (run_dir, rec)
    r2.require(sorted(runs) == sorted(b['training_seeds']),
               f"exactly the seeds {b['training_seeds']} are needed, got {sorted(runs)}")

    log('[h7r-r2-seed] models: c5, c8 and the three seed runs', flush=True)
    given = cli.parse_models(args.model, {'models': {'c5': None, 'c8': None}})
    models, identity = {}, {}
    for name in ('c5', 'c8'):
        models[name], identity[name] = r2.check_model(name, contract['models'][name], *given[name],
                                                      contract['frozen_state_sha256'], args.device)
    for seed in sorted(runs):
        run_dir, rec = runs[seed]
        spec = {'value_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                'value_metadata_sha256': rec['candidate']['metadata_sha256'],
                'policy_checkpoint_sha256': contract['models']['c5']['policy_checkpoint_sha256'],
                'policy_metadata_sha256': contract['models']['c5']['policy_metadata_sha256']}
        name = f'seed{seed}'
        models[name], identity[name] = r2.check_model(name, spec, run_dir / 'best.pt', run_dir / 'best.json',
                                                      contract['frozen_state_sha256'], args.device)
        r2.require(state_sha256(models[name].state_dict()) == rec['states']['state_sha256'],
                   f'{run_dir}: the checkpoint is not the state the run recorded')

    log('[h7r-r2-seed] H-dev predictions', flush=True)
    _, data, info, hdev_checks = cli.check_hdev(contract, Path(args.cache), root)
    for key, value in hdev_checks.items():
        r2.require(result['hdev'].get(key) == value, f'H-dev {key} differs from the R2-0A record')
    route = ho.routes(data)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    metrics = contract['metrics']
    groups = r2.row_groups(route, data['kind'].tolist(), info['ply'], target, info['result'], info['label_kind'],
                           metrics['ply_bins'])
    n_games = len(info['game_ids'])
    predictions = {}
    for name in list(models):
        predictions[name] = r2.predict(models.pop(name), data, args.device)
    wanted = [metrics['primary']] + list(metrics.get('components', []))
    cells = sorted({(m['prediction'], m['group']) for m in wanted})
    sums = {(n, p, g): r2.game_sums(predictions[n][p], target, weight, info['game'], groups[g], n_games)
            for n in predictions for p, g in cells}
    for name in ('c5', 'c8'):  # the environment reproduces the committed R2-0A values
        for m in wanted:
            value = r2.point_metrics(sums[(name, m['prediction'], m['group'])])[m['aggregate']]
            committed_value = result['models'][name]['metrics'][m['prediction']][m['group']][m['aggregate']]
            r2.require(abs(value - committed_value) <= REPLAY_MATCH,
                       f"{name} {m}: {value} differs from the committed R2-0A value {committed_value}")
    boot = contract['bootstrap']
    counts = r2.multiplicities(n_games, boot['samples'], boot['seed'])
    primary = metrics['primary']

    def pick(sa, m):
        return r2.point_metrics(sa[(m['prediction'], m['group'])])[m['aggregate']]

    seed_metrics = {}
    for seed in sorted(runs):
        name = f'seed{seed}'
        own = {(p, g): sums[(name, p, g)] for p, g in cells}
        seed_metrics[seed] = {'primary': pick(own, primary), 'components': [pick(own, m) for m in wanted[1:]],
                              'identity': identity[name], 'best_step': runs[seed][1]['best_step'],
                              'assignment_sha256': runs[seed][1]['assignment']['sha256']}
    seed_values = [seed_metrics[s]['primary'] for s in sorted(runs)]
    ref = result['summary'][contract['epsilon']['reference']]['primary']
    eps = r2.epsilon(ref['difference'], ref['ci95'], seed_values, reproduction['verdict']['verdict'])
    eps.update({'reference': contract['epsilon']['reference'], 'reference_difference': ref['difference'],
                'reference_ci95': ref['ci95'], 'seed_values': dict(zip(sorted(runs), seed_values)),
                'seed_mean': sum(seed_values) / 3,
                'seed_sd': (sum((v - sum(seed_values) / 3) ** 2 for v in seed_values) / 2) ** 0.5,
                'unit': 'squared value error (game macro, D4 / network)',
                'meaning': 'empirical development threshold: not a statistically guaranteed minimal detectable '
                           'effect, not a strength or promotion criterion'})
    value = eps['value']
    paired = {}
    for seed in sorted(runs):
        for other in ('c5', 'c8'):
            spec = f'seed{seed}:{other}'
            paired[spec] = []
            for m in wanted:
                key = (m['prediction'], m['group'])
                d = r2.difference(sums[(f'seed{seed}',) + key], sums[(other,) + key], m['aggregate'])
                ci = r2.bootstrap(counts, sums[(f'seed{seed}',) + key], sums[(other,) + key], m['aggregate'],
                                  boot['level'])
                paired[spec].append({'metric': m, 'difference': d, **ci})
            paired[spec][0]['beyond_epsilon'] = beyond(paired[spec][0]['difference'], paired[spec][0]['ci95'], value)
    reread = {}
    for spec, summary in result['summary'].items():
        p = summary['primary']
        reread[spec] = {'difference': p['difference'], 'ci95': p['ci95'], 'epsilon': value,
                        'reading': None if value is None else ('BEYOND_EPSILON' if beyond(p['difference'], p['ci95'],
                                                                                          value) else 'WITHIN_EPSILON')}
    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'contract_sha256': contract_sha, 'game': torch.tensor(info['game'], dtype=torch.int32),
                    'ply': torch.tensor(info['ply'], dtype=torch.int16), 'route': [r == 'network' for r in route],
                    'models': {n: {p: torch.tensor(v, dtype=torch.float64) for p, v in pr.items()}
                               for n, pr in predictions.items()}}, path)
    out = {'format': EPSILON_FORMAT, 'stage': 'R2-0B', 'contract_sha256': contract_sha,
           'amendment_sha256': amendment_sha, 'reproduction_sha256': reproduction_sha,
           'reproduction_verdict': reproduction['verdict'], 'r2_0a_sha256': pin['sha256'], 'git': git,
           'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES}, 'environment': environment(args.device),
           'hdev': hdev_checks, 'tolerance_source': 'amendment R2-DIAG-1-A1 (the parent contract leaves it null)',
           'seeds': seed_metrics, 'epsilon': eps, 'paired': paired, 'r2_0a_reread': reread,
           'predictions': ({'path': str(args.predictions), 'sha256': r2.file_sha256(args.predictions)}
                           if args.predictions else None),
           'minutes': round((time.monotonic() - started) / 60, 2),
           'scope': 'R2-0A comparisons are single runs: BEYOND_EPSILON is descriptive, not the DEVELOPMENT_SIGNAL '
                    'rule (three seeds against a direct control); no promotion, confirmation or strength claim'}
    from hybrid.h7r_ledger import publish_json
    publish_json(args.output, out)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--contract', type=Path, required=True)
    t.add_argument('--amendment', type=Path, required=True)
    t.add_argument('--seed', type=int, required=True)
    t.add_argument('--cycle-cache', type=Path, required=True)
    t.add_argument('--cycle-rows', type=Path, required=True)
    t.add_argument('--replay-cache', type=Path, required=True)
    t.add_argument('--parent', type=Path, required=True, help='c5 value checkpoint (its best.json beside it)')
    t.add_argument('--device', default='cuda')
    t.add_argument('--out-dir', type=Path, required=True)
    t.add_argument('--reproduction', type=Path, help='the passed reproduction record (training seeds only)')
    t.add_argument('--reason', help='why this run repeats an earlier one (the earlier output is kept)')
    t.add_argument('--allow-dirty', action='store_true', help='tests only')
    r = sub.add_parser('reproduce')
    r.add_argument('--contract', type=Path, required=True)
    r.add_argument('--amendment', type=Path, required=True)
    r.add_argument('--cache', type=Path, required=True, help='the H-dev cache')
    r.add_argument('--reference-model', required=True, help='c8 value checkpoint,metadata')
    r.add_argument('--run-dir', type=Path, required=True, help='the reproduction run (train --seed 20261025)')
    r.add_argument('--device', default='cuda')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--allow-dirty', action='store_true', help='tests only')
    e = sub.add_parser('epsilon')
    e.add_argument('--contract', type=Path, required=True)
    e.add_argument('--amendment', type=Path, required=True)
    e.add_argument('--cache', type=Path, required=True, help='the H-dev cache')
    e.add_argument('--result', type=Path, required=True, help='the committed R2-0A record')
    e.add_argument('--reproduction', type=Path, required=True, help='the passed reproduction record')
    e.add_argument('--run-dir', type=Path, action='append', required=True, help='each seed run (three)')
    e.add_argument('--model', action='append', required=True, help='c5=checkpoint,metadata and c8=...')
    e.add_argument('--device', default='cuda')
    e.add_argument('--predictions', type=Path, help='row-level predictions (under data/ or runs/)')
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = parser.parse_args(argv)
    try:
        out = {'train': train, 'reproduce': reproduce, 'epsilon': epsilon_command}[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.command == 'epsilon':
        eps = out['epsilon']
        print(f"[h7r-r2-seed] epsilon {eps['status']}: {eps['value']} (sample {eps['sample']}, seed {eps['seed']}; "
              f"seed values {eps['seed_values']})", flush=True)
        for spec, r in out['r2_0a_reread'].items():
            print(f"  {spec:8s} {r['difference']:+.6f} {r['ci95']} -> {r['reading']}", flush=True)
        return 0 if eps['status'] == 'COMPUTED' else 1
    if args.command == 'train':
        print(f"[h7r-r2-seed] seed {out['seed']}: {out['verdict']}, best step {out['best_step']}", flush=True)
        return 0 if out['verdict'] == 'CANDIDATE' else 1
    print(f"[h7r-r2-seed] reproduction: {out['verdict']['verdict']} {out['verdict']['reasons']}", flush=True)
    return 0 if out['verdict']['verdict'] in r2.REPRODUCED else 1


if __name__ == '__main__':
    raise SystemExit(main())
