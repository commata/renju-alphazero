"""H7-c value-head continuation of one cycle (hybrid.h7c_train.train_value; settings fixed in configs/h7c/).

    python scripts/build_h6_value_cache.py --labels runs/h7c/c1/value_rows.jsonl.gz --out-dir runs/h7c/c1/value_cache
    python -u scripts/h7c_train_value.py --cycle 1 --config configs/h7c/c1_value.json \\
        --cycle-cache runs/h7c/c1/value_cache --cycle-rows runs/h7c/c1/value_rows.jsonl.gz \\
        --replay-cache data/tactical/h6/value_cache_v3 \\
        --parent runs/h6_value_head_v3/best.pt --device cuda --out-dir runs/h7c/c1/value

The parent is the cycle's starting value checkpoint (c1: the frozen H6 v3; later the champion's value), and it must
be the value that played the cycle's self-play: ``--cycle-rows`` (the label file the cycle cache was built from, its
SHA-256 checked) names the self-play models in every row. The cycle cache must hold only this cycle's games
(``h7c:c<cycle>:``). The replay cache is the anchor of every cycle: the frozen H6 v3 value cache. Its
``labels_sha256`` must be the frozen H6 v3's: read from the parent's metadata when the parent is the frozen H6 v3
(checkpoint and metadata SHA-256 = ``analysis.tb_v3.MODELS``), from ``training_sources.replay_anchor_labels_sha256``
when it is an H7-c value candidate (which carries it on), and refused otherwise. Output: ``best.pt`` + ``best.json`` in the H6-2a loader contract (``hybrid.h6_value_leaf``
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


def replay_anchor(parent_path: Path, parent_meta: dict, allow_stand_in: bool = False) -> str:
    """The labels SHA-256 of the replay anchor (the frozen H6 v3 cache) as the parent's lineage records it."""
    from analysis import tb_v3
    from hybrid import h7c_train as tr

    if parent_meta.get('h7c_stage') == tr.VALUE_STAGE:
        anchor = (parent_meta.get('training_sources') or {}).get('replay_anchor_labels_sha256')
        if not isinstance(anchor, str) or len(anchor) != 64:
            raise ValueError(f'{parent_path}: an H7-c value candidate without its replay anchor')
        return anchor
    frozen = (tr.file_sha256(parent_path), tr.file_sha256(parent_path.with_name('best.json'))) == (
        tb_v3.MODELS['value_checkpoint_sha256'], tb_v3.MODELS['value_metadata_sha256'])
    if not frozen and not allow_stand_in:
        raise ValueError(f'{parent_path}: neither the frozen H6 v3 nor an H7-c value candidate')
    anchor = parent_meta.get('labels_sha256')
    if not isinstance(anchor, str):
        raise ValueError(f'{parent_path}: no H6 labels SHA-256 in its metadata')
    return anchor


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
    anchor = replay_anchor(parent_path, parent_meta, getattr(args, 'allow_stand_in', False))
    check_cycle_cache(Path(args.cycle_cache), args.cycle)
    cycle, cycle_manifest = load_cache_splits(Path(args.cycle_cache))
    source = tr.value_rows_source(Path(args.cycle_rows), args.cycle, cycle_manifest['labels_sha256'])
    parent_shas = {'value_checkpoint_sha256': tr.file_sha256(parent_path),
                   'value_metadata_sha256': tr.file_sha256(parent_path.with_name('best.json'))}
    if {k: source['models'][k] for k in parent_shas} != parent_shas:
        raise ValueError(f"{parent_path} is not the value that played the cycle's self-play "
                         f"({source['models']['value_checkpoint_sha256'][:12]})")
    replay, replay_manifest = load_cache_splits(Path(args.replay_cache))
    if replay_manifest['labels_sha256'] != anchor:
        raise ValueError(f"the replay cache's labels ({replay_manifest['labels_sha256'][:12]}) are not the replay "
                         f"anchor's ({anchor[:12]}, the frozen H6 v3 cache)")
    cfg = tr.ValueConfig.from_dict({**json.loads(Path(args.config).read_text(encoding='utf-8')), 'device': args.device})
    parent = load_checkpoint(parent_path, ModelConfig(**parent_meta['model_config']), device=args.device)
    result = tr.train_value(cfg, parent, cycle, replay, log=log)
    record = {'format': FORMAT, 'cycle': args.cycle, 'config': tr.config_dict(cfg), 'git': git,
              'inputs': {'parent_checkpoint_sha256': tr.file_sha256(parent_path),
                         'parent_metadata_sha256': tr.file_sha256(parent_path.with_name('best.json')),
                         'cycle_labels_sha256': cycle_manifest['labels_sha256'],
                         'replay_labels_sha256': replay_manifest['labels_sha256'], 'source': source,
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
                'training_sources': {'cycle_labels_sha256': cycle_manifest['labels_sha256'],
                                     'replay_anchor_labels_sha256': anchor},
                'source_models': source['models'], 'source_selfplay_git_commit': source['selfplay_git_commit'],
                'source_runtime_hash': source['runtime_hash'],
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
    parser.add_argument('--cycle-rows', type=Path, required=True, help='the value rows the cycle cache was built from')
    parser.add_argument('--replay-cache', type=Path, required=True, help='the replay anchor: the frozen H6 v3 cache')
    parser.add_argument('--parent', type=Path, required=True, help="the cycle's starting value (c1: H6 v3)")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    parser.add_argument('--allow-stand-in', action='store_true', help='tests only: a parent other than H6 v3')
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
