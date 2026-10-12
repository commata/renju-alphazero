"""H6-2a: value head only on the frozen H3 network (docs/h6-value.md).

    python scripts/train_h6_value.py --config configs/hybrid/h6_value_head.yaml
    python scripts/train_h6_value.py --config configs/hybrid/h6_value_head.yaml --resume   # continue

Refuses to start from a dirty worktree (``require_clean_git``); ``--allow-dirty`` is for smoke runs
only and is recorded in best.json. Overrides change the run, so a resumed run must use the same values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_value_train import Trainer, load_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='continue from <out_dir>/snapshot.pt')
    parser.add_argument('--allow-dirty', action='store_true', help='smoke runs only')
    for name, kind in (('device', str), ('batch-size', int), ('max-steps', int), ('out-dir', str),
                       ('torch-threads', int), ('eval-every', int), ('eval-max-states', int),
                       ('checkpoint-every', int), ('log-every', int), ('cache-dir', str), ('h3-checkpoint', str),
                       ('h6-report', str)):
        parser.add_argument(f'--{name}', type=kind)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ('config', 'resume', 'allow_dirty') and v is not None}
    if args.allow_dirty:
        overrides['require_clean_git'] = False
    cfg = load_config(args.config, overrides)
    trainer = Trainer(cfg)
    snapshot = Path(cfg.out_dir) / 'snapshot.pt'
    if args.resume:
        trainer.resume(snapshot)
        print(f'resumed at step {trainer.step}/{trainer.total_steps}', flush=True)
    elif snapshot.exists():
        parser.error(f'{snapshot} exists: use --resume or another --out-dir')
    split = trainer.manifest['splits']
    print(f"device {cfg.device}, git {trainer.git}, train states {split['train']['states']} "
          f"(proof weight share {split['train']['by_kind']['proof']['weight_share']}), val states "
          f"{split['val']['states']}, {trainer.steps_per_epoch} steps/epoch, {trainer.total_steps} steps", flush=True)
    result = trainer.train()
    best = result['best']
    print(json.dumps({'step': result['step'], 'samples_per_second': round(result['samples_per_second'], 1),
                      'best_step': best and best['step'], 'best_val_weighted_mse': best and best['weighted_mse'],
                      'best_val_proof': best and best['metrics']['by_kind']['proof'],
                      'best_val_outcome': best and best['metrics']['by_kind']['outcome']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
