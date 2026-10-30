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
            'git': git, 'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES},
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
    r2.require(rerun['contract_sha256'] == contract_sha and rerun['reference_record_sha256'] == b['reference_record']['sha256'],
               'the run names another contract or reference')
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
                          'compare': r2.compare_assignments(ref_assignment, sizes)},
           'best_step': {'reference': checks['best_step'][0], 'rerun': checks['best_step'][1]},
           'curve': {'steps_equal': sorted(ref_curve) == sorted(run_curve),
                     'max_abs_objective_difference': max((abs(ref_curve[s] - run_curve[s]) for s in ref_curve
                                                          if s in run_curve), default=None)},
           'predictions': {p: {'max_abs': float(delta[p].abs().max()), 'mean_abs': float(delta[p].abs().mean())}
                           for p in r2.PREDICTIONS},
           'primary': {'metric': primary, **loss, 'abs_difference': checks['abs_primary']},
           'tolerance': b['reproduction']['tolerance'], 'verdict': verdict,
           'minutes': round((time.monotonic() - started) / 60, 2),
           'scope': 'a check that finds reproduction problems; one re-run does not estimate GPU nondeterminism'}
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
    args = parser.parse_args(argv)
    try:
        out = train(args) if args.command == 'train' else reproduce(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.command == 'train':
        print(f"[h7r-r2-seed] seed {out['seed']}: {out['verdict']}, best step {out['best_step']}", flush=True)
        return 0 if out['verdict'] == 'CANDIDATE' else 1
    print(f"[h7r-r2-seed] reproduction: {out['verdict']['verdict']} {out['verdict']['reasons']}", flush=True)
    return 0 if out['verdict']['verdict'] in r2.REPRODUCED else 1


if __name__ == '__main__':
    raise SystemExit(main())
