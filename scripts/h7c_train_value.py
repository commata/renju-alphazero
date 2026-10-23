"""H7-c value-head continuation of one cycle (hybrid.h7c_train.train_value; settings fixed in configs/h7c/).

    python scripts/build_h6_value_cache.py --labels runs/h7c/c1/value_rows.jsonl.gz --out-dir runs/h7c/c1/value_cache
    python -u scripts/h7c_train_value.py --cycle 1 --config configs/h7c/c1_value.json \\
        --cycle-cache runs/h7c/c1/value_cache --replay-cache data/tactical/h6/value_cache_v3 \\
        --parent runs/h6_value_head_v3/best.pt --device cuda --out-dir runs/h7c/c1/value

The parent is the cycle's starting value checkpoint (c1: the frozen H6 v3); the replay cache must be built from the
parent's own label rows (its ``labels_sha256`` = the parent's). The cycle cache must hold only this cycle's self-play
games (``h7c:c<cycle>:``). Output: ``best.pt`` + ``best.json`` in the H6-2a loader contract (``hybrid.h6_value_leaf``
loads it; stage ``H6-2a`` names that contract, ``h7c_stage`` the step), and ``record.json``. ``NO_CANDIDATE``
(every evaluation's H6 val weighted MSE above 1.05 x the parent's): no ``best.pt``, exit 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FORMAT = 'h7c-value-train-v1'


def load_cache_splits(cache: Path, splits=('train', 'val')) -> tuple[dict, dict]:
    from hybrid.h6_value_cache import load_value_cache

    data, manifest = {}, None
    for split in splits:
        data[split], manifest = load_value_cache(cache, split, verify_hash=True)
    return data, manifest


def check_cycle_cache(cache: Path, cycle: int) -> None:
    prefix = f'h7c:c{cycle}:'
    for split in ('train', 'val', 'test'):
        path = cache / f'{split}_games.json'
        games = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        bad = [g for g in games if not str(g).startswith(prefix)]
        if bad:
            raise ValueError(f'{cache}: {split} holds games of another source or cycle, e.g. {bad[0]}')


def run(args, log=print) -> dict:
    from hybrid import h7c_train as tr
    from model.checkpoint import load_checkpoint, save_checkpoint
    from model.config import ACTION_INDEX_VERSION, ENCODER_VERSION, ModelConfig
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty: a candidate must name its code commit')
    out = Path(args.out_dir)
    if (out / 'best.pt').exists() or (out / 'record.json').exists():
        raise ValueError(f'{out} already holds a run')
    parent_path = Path(args.parent)
    parent_meta = json.loads(parent_path.with_name('best.json').read_text(encoding='utf-8'))
    if parent_meta.get('stage') != tr.VALUE_LOADER_STAGE or parent_meta.get('value_trained') is not True \
            or parent_meta.get('trunk_frozen') is not True or parent_meta.get('policy_head_frozen') is not True:
        raise ValueError(f'{parent_path}: not an H6-2a value head on a frozen trunk')
    check_cycle_cache(Path(args.cycle_cache), args.cycle)
    cycle, cycle_manifest = load_cache_splits(Path(args.cycle_cache))
    replay, replay_manifest = load_cache_splits(Path(args.replay_cache))
    if replay_manifest['labels_sha256'] != parent_meta.get('labels_sha256'):
        raise ValueError(f"the replay cache's labels ({replay_manifest['labels_sha256'][:12]}) are not the parent's "
                         f"({str(parent_meta.get('labels_sha256'))[:12]})")
    cfg = tr.ValueConfig.from_dict({**json.loads(Path(args.config).read_text(encoding='utf-8')), 'device': args.device})
    parent = load_checkpoint(parent_path, ModelConfig(**parent_meta['model_config']), device=args.device)
    result = tr.train_value(cfg, parent, cycle, replay, log=log)
    record = {'format': FORMAT, 'cycle': args.cycle, 'config': tr.config_dict(cfg), 'git': git,
              'inputs': {'parent_checkpoint_sha256': tr.file_sha256(parent_path),
                         'parent_metadata_sha256': tr.file_sha256(parent_path.with_name('best.json')),
                         'cycle_labels_sha256': cycle_manifest['labels_sha256'],
                         'replay_labels_sha256': replay_manifest['labels_sha256'],
                         'cycle_cache_splits': {k: v['sha256'] for k, v in cycle_manifest['splits'].items()},
                         'replay_cache_splits': {k: v['sha256'] for k, v in replay_manifest['splits'].items()}},
              **result['record']}
    out.mkdir(parents=True, exist_ok=True)
    if result['state'] is not None:
        parent.load_state_dict(result['state'])
        save_checkpoint(out / 'best.pt.tmp', parent)
        meta = {'stage': tr.VALUE_LOADER_STAGE, 'h7c_stage': tr.VALUE_STAGE, 'track': 'B', 'cycle': args.cycle,
                'policy_trained': True, 'value_trained': True, 'trunk_frozen': True, 'policy_head_frozen': True,
                'parent_h3_checkpoint_sha256': parent_meta['parent_h3_checkpoint_sha256'],
                'parent_h3_metadata_sha256': parent_meta.get('parent_h3_metadata_sha256'),
                'parent_value_checkpoint_sha256': record['inputs']['parent_checkpoint_sha256'],
                'frozen_state_sha256': result['record']['frozen_state_sha256'],
                'model_config': parent_meta['model_config'], 'encoder_version': ENCODER_VERSION,
                'action_index_version': ACTION_INDEX_VERSION,
                'labels_sha256': {'cycle': cycle_manifest['labels_sha256'], 'replay': replay_manifest['labels_sha256']},
                'mix': 'cycle:replay = 50:50 per epoch', 'config': tr.config_dict(cfg),
                'best': result['record']['best'], 'git': git}
        (out / 'best.json.tmp').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        (out / 'best.pt.tmp').replace(out / 'best.pt')
        (out / 'best.json.tmp').replace(out / 'best.json')
        record['candidate'] = {'checkpoint_sha256': tr.file_sha256(out / 'best.pt'),
                               'metadata_sha256': tr.file_sha256(out / 'best.json')}
    (out / 'record.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cycle-cache', type=Path, required=True)
    parser.add_argument('--replay-cache', type=Path, required=True, help="the parent's own H6 value cache")
    parser.add_argument('--parent', type=Path, required=True, help="the cycle's starting value (c1: H6 v3)")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = parser.parse_args(argv)
    try:
        record = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({k: record[k] for k in ('verdict', 'total_steps', 'warmup_steps', 'union_masking', 'best')},
                     indent=1), flush=True)
    return 0 if record['verdict'] == 'CANDIDATE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
