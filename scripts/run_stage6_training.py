"""Stage 6 self-play training loop CLI (new run, stop, resume, checkpoint verification).

New run:   --config configs/stage6_mvp.yaml [--stop-after-generation N] [--run-dir DIR]
Resume:    --resume runs/<run>/checkpoints/latest.pt [--config SAME.yaml]
Verify:    --verify-checkpoint runs/<run>/checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from training.config import load_config, validate_config  # noqa: E402
from training.loop import run_training, verify_checkpoint  # noqa: E402
from training.training_checkpoint import load_checkpoint_payload  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='Stage 6 YAML config')
    parser.add_argument('--resume', type=Path, help='training checkpoint (e.g. latest.pt)')
    parser.add_argument('--run-dir', type=Path, help='explicit new run directory')
    parser.add_argument('--generations', type=int,
                        help='override training.generations (total target)')
    parser.add_argument('--stop-after-generation', type=int,
                        help='stop once checkpoint generation reaches this value')
    parser.add_argument('--verify-checkpoint', type=Path,
                        help='reload a checkpoint and play one validated self-play game')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    if args.verify_checkpoint:
        torch.set_num_threads(1)
        print(json.dumps(verify_checkpoint(args.verify_checkpoint), indent=2))
        return 0
    if args.resume is None and args.config is None:
        parser.error('--config is required for a new run')
    if args.resume is not None and args.run_dir is not None:
        parser.error('--run-dir cannot be combined with --resume (resume uses its own run)')
    if args.config is not None:
        config = load_config(args.config)
    elif args.resume is not None:
        config = load_checkpoint_payload(args.resume)['config']
    if args.generations is not None:
        config['training']['generations'] = args.generations
        validate_config(config)
    state = run_training(config, run_dir=args.run_dir, resume=args.resume,
                         stop_after=args.stop_after_generation,
                         log=lambda message: print(message, flush=True))
    print(json.dumps({'run_dir': str(state.run_dir), 'generation': state.generation,
                      'global_step': state.global_step, 'buffer_size': len(state.buffer),
                      'latest': str(state.run_dir / 'checkpoints' / 'latest.pt')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
