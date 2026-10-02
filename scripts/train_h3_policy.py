"""H3: policy-only pretraining on the RenjuNet cache (docs/mcts-v8-teacher.md §12.11).

    python scripts/train_h3_policy.py --config configs/hybrid/h3_policy_64x4.yaml
    python scripts/train_h3_policy.py --config configs/hybrid/h3_policy_64x4.yaml --resume   # continue

Overrides for smoke tests and throughput runs (they change the run, so a resumed
run must use the same values): --device, --batch-size, --max-steps, --out-dir,
--torch-threads, --eval-every, --eval-max-states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h3_train import Trainer, load_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='continue from <out_dir>/snapshot.pt')
    for name, kind in (('device', str), ('batch-size', int), ('max-steps', int), ('out-dir', str),
                       ('torch-threads', int), ('eval-every', int), ('eval-max-states', int),
                       ('checkpoint-every', int), ('log-every', int), ('cache-dir', str)):
        parser.add_argument(f'--{name}', type=kind)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ('config', 'resume') and v is not None}
    cfg = load_config(args.config, overrides)
    trainer = Trainer(cfg)
    snapshot = Path(cfg.out_dir) / 'snapshot.pt'
    if args.resume:
        trainer.resume(snapshot)
        print(f'resumed at step {trainer.step}/{trainer.total_steps}', flush=True)
    elif snapshot.exists():
        parser.error(f'{snapshot} exists: use --resume or another --out-dir')
    print(f'device {cfg.device}, train states {trainer.train_data["target"].shape[0]}, '
          f'{trainer.steps_per_epoch} steps/epoch, {trainer.total_steps} steps', flush=True)
    result = trainer.train()
    print(json.dumps({'step': result['step'], 'samples_per_second': round(result['samples_per_second'], 1),
                      'best_step': result['best'] and result['best']['step'],
                      'best_val': result['best'] and {k: result['best']['metrics'][k]
                                                      for k in ('top1', 'top3', 'top5', 'ce')}}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
