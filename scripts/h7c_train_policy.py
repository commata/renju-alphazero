"""H7-c policy fine-tune of one cycle (hybrid.h7c_train.train_policy; settings fixed in configs/h7c/, docs/h7-c1.md).

    python -u scripts/h7c_train_policy.py --cycle 1 --config configs/h7c/c1_policy.json \\
        --policy-rows runs/h7c/c1/policy_rows.jsonl.gz --parent runs/h3_policy_64x4/best.pt \\
        --h3-cache data/external/renjunet/h3_cache --device cuda --out-dir runs/h7c/c1/policy

The teacher (KL) and the guard baseline are always the frozen H3 (``analysis.tb_v3.MODELS``); the parent is the
cycle's starting policy (c1: the same H3), and it must be the policy that played the cycle's self-play: the rows'
``h7c_provenance`` (one model pair, every game id of this cycle) names it by SHA-256. The output is a policy-only checkpoint ``best.pt`` with ``best.json``
(``hybrid.h4_policy.RootPolicy`` loads it; ``analysis.tb_v3_net`` takes it as a candidate) and ``record.json``. No
eligible checkpoint (every one more than 2 points of RenjuNet top-1 below H3): ``NO_CANDIDATE``, no ``best.pt``,
exit 1. A dirty tree is refused (``--allow-dirty`` for tests only).
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

FORMAT = 'h7c-policy-train-v1'


def load_policy_net(checkpoint: Path, device: str):
    """A policy-only checkpoint with its metadata (``RootPolicy``'s contract)."""
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig

    meta = json.loads(Path(checkpoint).with_suffix('.json').read_text(encoding='utf-8'))
    if meta.get('policy_trained') is not True or meta.get('value_trained') is not False:
        raise ValueError(f'{checkpoint}: not a policy-only checkpoint')
    cfg = meta['config']
    return load_checkpoint(checkpoint, ModelConfig(channels=cfg['channels'], blocks=cfg['blocks']), device=device), meta


def run(args, log=print) -> dict:
    import torch

    from analysis import tb_v3
    from hybrid import h7c_train as tr
    from hybrid.h3_cache import load_cache
    from hybrid.h3_train import H3Config, evaluate
    from model.checkpoint import save_checkpoint
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty: a candidate must name its code commit')
    out = Path(args.out_dir)
    if (out / 'best.pt').exists() or (out / 'record.json').exists():
        raise ValueError(f'{out} already holds a run')
    teacher_path = Path(args.teacher)
    shas = {'checkpoint': tr.file_sha256(teacher_path), 'metadata': tr.file_sha256(teacher_path.with_suffix('.json'))}
    if (shas['checkpoint'], shas['metadata']) != (tb_v3.MODELS['policy_checkpoint_sha256'],
                                                  tb_v3.MODELS['policy_metadata_sha256']) and not args.allow_stand_in:
        raise ValueError(f'{teacher_path} is not the frozen H3 (the KL teacher and guard baseline)')
    cfg = tr.PolicyConfig.from_dict({**json.loads(Path(args.config).read_text(encoding='utf-8')), 'device': args.device})
    parent, parent_meta = load_policy_net(Path(args.parent), args.device)
    teacher, _ = load_policy_net(teacher_path, args.device)
    source = tr.policy_rows_source(Path(args.policy_rows), args.cycle)
    parent_shas = {'policy_checkpoint_sha256': tr.file_sha256(args.parent),
                   'policy_metadata_sha256': tr.file_sha256(Path(args.parent).with_suffix('.json'))}
    if {k: source['models'][k] for k in parent_shas} != parent_shas:
        raise ValueError(f"{args.parent} is not the policy that played the cycle's self-play "
                         f"({source['models']['policy_checkpoint_sha256'][:12]})")
    rows = tr.load_policy_rows(Path(args.policy_rows))
    train, val = tr.policy_tensors(rows['train']), tr.policy_tensors(rows.get('val', []))
    guard_data, _ = load_cache(Path(args.h3_cache), 'val', verify_hash=True)
    h3_cfg = H3Config(seed=cfg.seed)
    result = tr.train_policy(cfg, parent, teacher, train, val, guard_data,
                             lambda model, data: evaluate(model, data, h3_cfg, torch.device(args.device)), log=log)
    record = {'format': FORMAT, 'cycle': args.cycle, 'config': tr.config_dict(cfg), 'git': git,
              'inputs': {'policy_rows': str(args.policy_rows), 'policy_rows_sha256': tr.file_sha256(args.policy_rows),
                         'parent_checkpoint_sha256': tr.file_sha256(args.parent),
                         'parent_metadata_sha256': tr.file_sha256(Path(args.parent).with_suffix('.json')),
                         'teacher_checkpoint_sha256': shas['checkpoint'], 'h3_cache': str(args.h3_cache),
                         'source': source},
              **result['record']}
    out.mkdir(parents=True, exist_ok=True)
    if result['state'] is not None:
        parent.load_state_dict(result['state'])
        save_checkpoint(out / 'best.pt.tmp', parent)
        meta = {'stage': tr.POLICY_STAGE, 'track': 'B', 'cycle': args.cycle, 'policy_trained': True,
                'value_trained': False, 'config': {'channels': parent_meta['config']['channels'],
                                                   'blocks': parent_meta['config']['blocks'], **tr.config_dict(cfg)},
                'parent_checkpoint_sha256': record['inputs']['parent_checkpoint_sha256'],
                'teacher_checkpoint_sha256': shas['checkpoint'], 'policy_rows_sha256': record['inputs']['policy_rows_sha256'],
                'source_models': source['models'], 'source_selfplay_git_commit': source['selfplay_git_commit'],
                'source_runtime_hash': source['runtime_hash'],
                'best': result['record']['best'], 'teacher_guard_top1': result['record']['teacher_guard']['top1'],
                'git': git}
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
    parser.add_argument('--policy-rows', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True, help="the cycle's starting policy (c1: H3)")
    parser.add_argument('--teacher', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt', help='the frozen H3')
    parser.add_argument('--h3-cache', type=Path, required=True, help='the H3 cache (its val split is the guard)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    parser.add_argument('--allow-stand-in', action='store_true', help='tests only: a teacher other than H3')
    args = parser.parse_args(argv)
    try:
        record = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({k: record[k] for k in ('verdict', 'total_steps', 'warmup_steps', 'best')}, indent=1), flush=True)
    return 0 if record['verdict'] == 'CANDIDATE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
