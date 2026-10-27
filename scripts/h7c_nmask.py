"""H7C-NMASK-1 (``VALUE_ONLY_NETWORK_MASK``, ``hybrid.h7c_nmask``): the mask, the confirmatory sample, the decision
and the one masked value training (docs/h7-c1.md, "VALUE_ONLY_NETWORK_MASK 계약").

    mask      the c8 value rows and cycle cache -> ``runs/h7c/nmask/mask.pt`` (one keep bool per cache row) and
              ``docs/h7c/nmask/mask_manifest.json`` (row-by-row link, counts and weights by split and label kind,
              SHA-256); its exact_leaf rows by kind must be the committed c8 D4 classification
    sample    the c8 self-play games the E2 sensitivity did not use -> ``docs/h7c/nmask/confirm_sample.json``
    decide    ``docs/h7c/nmask/decision.json`` from the committed records (c8 verdict / evidence / value record, B2
              value config, the read-only check = PASS, the mask manifest, the sample): commit it before training
    train     the masked value training from c5 with c8's data and settings -> ``runs/h7c/nmask/value/`` (``best.pt``,
              ``best.json``, ``record.json``); the parent's and the control's (c8) val numbers are recomputed with
              the same definitions and the unmasked ones must reproduce the c8 value record (or the run fails)

    python -u scripts/h7c_nmask.py mask
    python -u scripts/h7c_nmask.py sample
    python -u scripts/h7c_nmask.py decide
    python -u scripts/h7c_nmask.py train --device cuda [--smoke]

A dirty tree is refused; ``train`` refuses an uncommitted decision. ``--smoke`` trains one step into
``runs/h7c/nmask/smoke/`` (marked ``smoke``: never a candidate of the evaluation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7c_nmask as nm  # noqa: E402

TRAIN_FORMAT = 'h7c-nmask-train-v1'
WMSE_TOLERANCE = 1e-4  # h7c_diagnose's: the trainer's numbers recomputed on the same rows and model
SMOKE = {'epochs': 0.01, 'evaluations': 1}


def _git(allow_dirty: bool) -> dict:
    from h7c_arena import _git as git_state

    git = git_state()
    if git['dirty'] is not False and not allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): every H7C-NMASK-1 file names its code commit')
    return git


def _refuse_existing(*paths) -> None:
    for path in paths:
        if Path(path).exists():
            raise ValueError(f'{path} exists: an H7C-NMASK-1 file is never overwritten')


def _root(args) -> Path:
    return Path(getattr(args, 'root', None) or ROOT)


def mask(args, log=print) -> dict:
    from hybrid.progress import Progress

    root = _root(args)
    git = _git(args.allow_dirty)
    out_pt, out_manifest = Path(args.mask), root / nm.RECORDS['mask_manifest']
    _refuse_existing(out_pt, out_manifest)
    evidence = nm.read(nm.RECORDS['c8_evidence'], root)['files_sha256']
    rows, cache = Path(args.value_rows), Path(args.cycle_cache)
    nm.require(nm.file_sha256(rows) == evidence[nm.PATHS['value_rows']], f'{rows} is not the c8 value rows')
    nm.require(nm.file_sha256(cache / 'manifest.json') == evidence[nm.PATHS['cycle_cache'] + '/manifest.json'],
               f'{cache} is not the c8 value cache')
    games = sum(len(json.loads((cache / f'{s}_games.json').read_text())) for s in ('train', 'val', 'test'))
    keep, body = nm.build_mask(rows, cache, Progress('h7c-nmask-mask', games, 'games', log=log))
    nm.check_mask_against_d4(body, nm.read(nm.RECORDS['d4_c8'], root))
    body.update(git=git, value_rows_sha256=nm.file_sha256(rows), mask_file=nm.PATHS['mask'])
    nm.save_mask(out_pt, keep, body)
    body['mask_file_sha256'] = nm.file_sha256(out_pt)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(body, indent=1) + '\n', encoding='utf-8')
    return body


def sample(args, log=print) -> dict:
    import h7c_labels as lab

    root = _root(args)
    git = _git(args.allow_dirty)
    out = root / nm.RECORDS['sample']
    _refuse_existing(out)
    evidence = nm.read(nm.RECORDS['c8_evidence'], root)['files_sha256']
    summary = nm.read(nm.RECORDS['e2_summary'], root)
    selfplay = Path(args.selfplay)
    sha = nm.file_sha256(selfplay)
    nm.require(sha == evidence[nm.PATHS['selfplay']] == summary['identity']['selfplay_jsonl_sha256'],
               f'{selfplay} is not the c8 self-play the c8 evidence and the E2 summary name')
    games = lab.read_games([selfplay])
    lab.check_games(games)
    body = {**nm.confirm_sample(games, summary), 'git': git}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1) + '\n', encoding='utf-8')
    log(f"[h7c-nmask] confirmatory sample: {body['games']} games, {body['plies']:,} plies "
        f"(E2 sample {body['e2_sample']['games']} excluded)", flush=True)
    return body


def decide(args, log=print) -> dict:
    root = _root(args)
    _git(args.allow_dirty)
    out = root / nm.RECORDS['decision']
    _refuse_existing(out)
    for name in ('check', 'mask_manifest', 'sample'):
        nm.require(args.allow_uncommitted or nm.committed(root / nm.RECORDS[name], root),
                   f'{nm.RECORDS[name]} must be committed before the decision')
    decision = nm.expected_decision(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=1) + '\n', encoding='utf-8')
    log(f'[h7c-nmask] decision written: {out} (commit it before training or evaluation)', flush=True)
    return decision


# --- the training ------------------------------------------------------------------------------------------------

def _close(a, b) -> bool:
    if a is None or b is None:  # an empty effective cycle val set gives no WMSE: both must be empty
        return a is None and b is None
    return abs(a - b) <= WMSE_TOLERANCE


def _val_report(model, val_sets: dict, device) -> dict:
    from hybrid.h6_value_train import evaluate
    from hybrid.h7c_train import masked_metric

    cycle, replay = evaluate(model, val_sets['cycle'], device), evaluate(model, val_sets['replay'], device)
    masked = masked_metric(model, val_sets['cycle'], device)
    mean = lambda *xs: sum(x for x in xs if x is not None) / sum(x is not None for x in xs)  # noqa: E731 (trainer's)
    return {'cycle_wmse_masked': masked['masked_wmse'], 'cycle_wmse_original': cycle['weighted_mse'],
            'replay_wmse': replay['weighted_mse'],
            'objective_masked': mean(masked['masked_wmse'], replay['weighted_mse']),
            'objective_original': mean(cycle['weighted_mse'], replay['weighted_mse']),
            'cycle_val_rows': masked['rows'], 'cycle_val_kept_rows': masked['kept_rows']}


def train(args, log=print) -> dict:
    import h7c_train_value as tv
    from hybrid import h7c_train as tr
    from model.checkpoint import load_checkpoint, save_checkpoint
    from model.config import ACTION_INDEX_VERSION, ENCODER_VERSION, ModelConfig

    root = _root(args)
    git = _git(args.allow_dirty)
    decision, decision_sha = nm.load_decision(args.decision, root, require_committed=not args.allow_uncommitted)
    out = Path(args.out_dir or (root / ('runs/h7c/nmask/smoke' if args.smoke else 'runs/h7c/nmask/value')))
    _refuse_existing(out / 'best.pt', out / 'record.json')
    src, value = decision['source'], decision['value']
    parent_path, control_path = Path(args.parent), Path(args.control)
    pair = lambda p: {'value_checkpoint_sha256': nm.file_sha256(p),  # noqa: E731
                      'value_metadata_sha256': nm.file_sha256(p.with_name('best.json'))}
    checks = {
        'config': nm.file_sha256(args.config) == value['config_sha256'],
        'settings': tr.config_dict(tr.ValueConfig.from_dict({**json.loads(Path(args.config).read_text(
            encoding='utf-8')), 'device': args.device})) == value['settings'],
        'parent value = c5': pair(parent_path) == {k: decision['parent'][k] for k in pair(parent_path)},
        'control value = c8': pair(control_path) == {k: decision['control'][k] for k in pair(control_path)},
        'value rows': nm.file_sha256(args.cycle_rows) == src['value_rows_sha256'],
        'mask manifest': nm.file_sha256(root / nm.RECORDS['mask_manifest']) == decision['mask']['manifest_sha256'],
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    nm.require(not failed, f'not the inputs of the decision ({", ".join(failed)})')
    parent_meta = json.loads(parent_path.with_name('best.json').read_text(encoding='utf-8'))
    nm.require(parent_meta.get('stage') == tr.VALUE_LOADER_STAGE and parent_meta.get('trunk_frozen') is True
               and parent_meta.get('policy_head_frozen') is True, f'{parent_path}: not a value head on a frozen trunk')
    anchor = tv.replay_anchor(parent_path, parent_meta)
    nm.require(anchor == value['replay_anchor_labels_sha256'], 'the parent\'s replay anchor is not the decision\'s')
    tv.check_cycle_cache(Path(args.cycle_cache), src['cycle'], src['selfplay_run_seed'], rows=Path(args.cycle_rows))
    cycle, cycle_manifest = tv.load_cache_splits(Path(args.cycle_cache))
    nm.require(cycle_manifest['labels_sha256'] == src['cycle_labels_sha256'] and {
        k: v['sha256'] for k, v in cycle_manifest['splits'].items()} == src['cycle_cache_splits'],
        'the cycle cache is not the c8 cache')
    source = tr.value_rows_source(Path(args.cycle_rows), src['cycle'], cycle_manifest['labels_sha256'])
    nm.require(source['models'] == decision['parent'], 'the c8 self-play was not played by c5')
    replay, replay_manifest = tv.load_cache_splits(Path(args.replay_cache))
    nm.require(replay_manifest['labels_sha256'] == anchor and {
        k: v['sha256'] for k, v in replay_manifest['splits'].items()} == src['replay_cache_splits'],
        'the replay cache is not the c8 run\'s')
    body = nm.read(nm.RECORDS['mask_manifest'], root)
    keep = nm.load_mask(Path(args.mask), body)
    nm.require(body['keep_sha256'] == decision['mask']['keep_sha256'], 'the mask is not the decision\'s')
    settings = {**json.loads(Path(args.config).read_text(encoding='utf-8')), 'device': args.device,
                **(SMOKE if args.smoke else {})}
    cfg = tr.ValueConfig.from_dict(settings)
    parent = load_checkpoint(parent_path, ModelConfig(**parent_meta['model_config']), device=args.device)
    log(f"[h7c-nmask] training from c5 with the mask ({sum(v['masked_rows'] for v in body['splits'].values()):,} "
        f"cycle rows masked){' SMOKE' if args.smoke else ''}", flush=True)
    result = tr.train_value(cfg, parent, cycle, replay, log=log, cycle_keep={s: keep[s] for s in ('train', 'val')})
    record = {'format': TRAIN_FORMAT, 'experiment': nm.EXPERIMENT, 'smoke': bool(args.smoke), 'git': git,
              'decision_sha256': decision_sha, 'config': tr.config_dict(cfg),
              'inputs': {'parent': decision['parent'], 'control': decision['control'],
                         'config_sha256': value['config_sha256'], 'cycle_labels_sha256': src['cycle_labels_sha256'],
                         'replay_labels_sha256': replay_manifest['labels_sha256'], 'source': source,
                         'mask_keep_sha256': body['keep_sha256'], 'mask_manifest_sha256': decision['mask'][
                             'manifest_sha256']},
              **result['record']}
    control_meta = json.loads(control_path.with_name('best.json').read_text(encoding='utf-8'))
    control = load_checkpoint(control_path, ModelConfig(**control_meta['model_config']), device=args.device)
    from hybrid.h6_value_train import freeze
    freeze(control)
    record['control_val'] = _val_report(control, result['val_sets'], args.device)
    record['definitions'] = {'masked': nm.CYCLE_VAL_METRIC, 'original': 'the c8 record\'s unmasked metrics',
                             'note': 'masked and original numbers are separate definitions: compare like with like'}
    reproduced = {
        'parent original cycle / replay WMSE = the c8 record\'s parent': _close(
            record['parent_val']['cycle_wmse_original'], value['control_parent_val']['cycle_wmse'])
        and _close(record['parent_val']['replay_wmse'], value['control_parent_val']['replay_wmse']),
        'control original cycle / replay WMSE = the c8 record\'s best': _close(
            record['control_val']['cycle_wmse_original'], value['control_best']['cycle_wmse'])
        and _close(record['control_val']['replay_wmse'], value['control_best']['replay_wmse']),
        'frozen trunk and policy head = c8\'s': record['frozen_state_sha256'] == value['frozen_state_sha256'],
    }
    if not args.smoke:
        reproduced['steps and rows = c8\'s'] = {k: record[k] for k in value['steps']} == value['steps']
    record['reproduction'] = reproduced
    nm.require(all(reproduced.values()), f'the run is not comparable with c8: {reproduced}')
    out.mkdir(parents=True, exist_ok=True)
    if result['state'] is not None:
        parent.load_state_dict(result['state'])
        save_checkpoint(out / 'best.pt.tmp', parent)
        meta = {'stage': tr.VALUE_LOADER_STAGE, 'h7c_stage': tr.VALUE_STAGE, 'track': 'B', 'cycle': src['cycle'],
                'experiment': nm.EXPERIMENT, 'recipe': nm.RECIPE, 'smoke': bool(args.smoke),
                'nmask_decision_sha256': decision_sha, 'mask_keep_sha256': body['keep_sha256'], 'loss': nm.LOSS,
                'policy_trained': True, 'value_trained': True, 'trunk_frozen': True, 'policy_head_frozen': True,
                'parent_h3_checkpoint_sha256': parent_meta['parent_h3_checkpoint_sha256'],
                'parent_h3_metadata_sha256': parent_meta.get('parent_h3_metadata_sha256'),
                'parent_value_checkpoint_sha256': decision['parent']['value_checkpoint_sha256'],
                'frozen_state_sha256': record['frozen_state_sha256'], 'model_config': parent_meta['model_config'],
                'encoder_version': ENCODER_VERSION, 'action_index_version': ACTION_INDEX_VERSION,
                'training_sources': {'cycle_labels_sha256': src['cycle_labels_sha256'],
                                     'replay_anchor_labels_sha256': anchor},
                'source_models': source['models'], 'source_selfplay_git_commit': source['selfplay_git_commit'],
                'source_runtime_hash': source['runtime_hash'], 'mix': 'cycle:replay = 50:50 per epoch',
                'config': tr.config_dict(cfg), 'recipe_decision_sha256': None, 'best': record['best'], 'git': git}
        (out / 'best.json.tmp').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        (out / 'best.pt.tmp').replace(out / 'best.pt')
        (out / 'best.json.tmp').replace(out / 'best.json')
        record['candidate'] = {'checkpoint_sha256': nm.file_sha256(out / 'best.pt'),
                               'metadata_sha256': nm.file_sha256(out / 'best.json')}
    (out / 'record.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    return record


def check_candidate(value_checkpoint, decision: dict, decision_sha: str) -> dict:
    """An H7C-NMASK-1 candidate of this decision (never a smoke run); returns its record's candidate SHA-256."""
    meta = json.loads(Path(value_checkpoint).with_name('best.json').read_text(encoding='utf-8'))
    record = json.loads(Path(value_checkpoint).with_name('record.json').read_text(encoding='utf-8'))
    ok = {
        'experiment': (meta.get('experiment'), record.get('experiment')) == (nm.EXPERIMENT, nm.EXPERIMENT),
        'not a smoke run': meta.get('smoke') is False and record.get('smoke') is False,
        'this decision': meta.get('nmask_decision_sha256') == record.get('decision_sha256') == decision_sha,
        'trained from c5 on c5\'s self-play': meta.get('source_models') == decision['parent'],
        'the candidate files': record.get('candidate') == {
            'checkpoint_sha256': nm.file_sha256(value_checkpoint),
            'metadata_sha256': nm.file_sha256(Path(value_checkpoint).with_name('best.json'))},
        'a clean training tree': (record.get('git') or {}).get('dirty') is False,
        'reproduction passed': bool(record.get('reproduction')) and all(record['reproduction'].values()),
    }
    failed = sorted(k for k, v in ok.items() if not v)
    nm.require(not failed, f'{value_checkpoint}: not the H7C-NMASK-1 candidate ({", ".join(failed)})')
    return record['candidate']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = {name: sub.add_parser(name) for name in ('mask', 'sample', 'decide', 'train')}
    for q in p.values():
        q.add_argument('--allow-dirty', action='store_true', help='tests only')
    p['mask'].add_argument('--value-rows', type=Path, default=ROOT / nm.PATHS['value_rows'])
    p['mask'].add_argument('--cycle-cache', type=Path, default=ROOT / nm.PATHS['cycle_cache'])
    p['mask'].add_argument('--mask', type=Path, default=ROOT / nm.PATHS['mask'])
    p['sample'].add_argument('--selfplay', type=Path, default=ROOT / nm.PATHS['selfplay'])
    p['decide'].add_argument('--allow-uncommitted', action='store_true', help='tests only')
    t = p['train']
    t.add_argument('--decision', type=Path, default=ROOT / nm.RECORDS['decision'])
    t.add_argument('--config', type=Path, default=ROOT / 'configs/h7c/c1_value.json')
    t.add_argument('--cycle-cache', type=Path, default=ROOT / nm.PATHS['cycle_cache'])
    t.add_argument('--cycle-rows', type=Path, default=ROOT / nm.PATHS['value_rows'])
    t.add_argument('--replay-cache', type=Path, default=ROOT / nm.PATHS['replay_cache'])
    t.add_argument('--parent', type=Path, default=ROOT / nm.PATHS['parent_value'])
    t.add_argument('--control', type=Path, default=ROOT / nm.PATHS['control_value'])
    t.add_argument('--mask', type=Path, default=ROOT / nm.PATHS['mask'])
    t.add_argument('--device', default='cuda')
    t.add_argument('--out-dir', type=Path)
    t.add_argument('--smoke', action='store_true', help='one step into runs/h7c/nmask/smoke/ (never a candidate)')
    t.add_argument('--allow-uncommitted', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out = {'mask': mask, 'sample': sample, 'decide': decide, 'train': train}[args.command](args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == 'train':
        print(json.dumps({k: out.get(k) for k in ('verdict', 'smoke', 'best', 'mask', 'control_val',
                                                  'reproduction')}, indent=1, default=str), flush=True)
        return 0 if out['verdict'] == 'CANDIDATE' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
