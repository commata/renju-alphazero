"""Why does ``training_seconds`` grow during a long run? (Stage 8 Gate 2 diagnostic)

In Gate 2 the 50-step training phase grew from ~6.4 s to ~22 s per generation while
self-play time stayed flat. This separates the two candidate causes on a checkpoint:

1. arithmetic: the same 50 training steps (real replay buffer, augmentation, Adam)
   timed as-is and with ``torch.set_flush_denormal(True)``, plus a count of subnormal
   (denormal) values in the weights and Adam state - subnormals make CPU float ops slow;
2. logging: 50 ``MetricsLogger.log`` appends (one per training step, as in the loop) to a
   copy of the run's ``metrics.jsonl`` vs a new small file - on Windows, reopening a large
   file for every append can be slow (e.g. real-time antivirus scanning).

Nothing in the run directory is modified (the metrics test uses a temporary copy).

    python scripts/diagnose_stage8_training_time.py --run-dir runs/stage8_d16 --generation 320
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from training.dataset import build_batch  # noqa: E402
from training.metrics import MetricsLogger  # noqa: E402
from training.trainer import train_step  # noqa: E402
from training.training_checkpoint import load_training_state  # noqa: E402


def count_subnormal(tensors) -> tuple[int, int]:
    tiny = torch.finfo(torch.float32).tiny
    sub = total = 0
    for t in tensors:
        if not torch.is_floating_point(t):
            continue
        t = t.detach().float()
        sub += int(((t != 0) & (t.abs() < tiny)).sum())
        total += t.numel()
    return sub, total


def time_training(checkpoint: Path, steps: int, flush_denormal: bool) -> float:
    state = load_training_state(checkpoint)
    t = state.config['training']
    torch.set_flush_denormal(flush_denormal)   # returns support, not the previous mode
    try:
        started = perf_counter()
        for _ in range(steps):
            batch = build_batch(state.buffer, t['batch_size'], sample_rng=state.sample_rng,
                                augment_rng=state.augment_rng,
                                augment=state.config['augmentation']['enabled'])
            train_step(state.model, state.optimizer, batch, state.config['loss']['value_weight'],
                       t['grad_clip'], state.config['loss']['l2_coeff'])
        return perf_counter() - started
    finally:
        torch.set_flush_denormal(False)


def time_appends(path: Path, count: int) -> float:
    logger = MetricsLogger(path)
    event = {'type': 'train', 'generation': 0, 'global_step': 0, 'policy_loss': 2.4,
             'value_loss': 0.1, 'total_loss': 2.5, 'grad_norm': 1.0, 'lr': 0.001,
             'buffer_size': 10000}
    started = perf_counter()
    for _ in range(count):
        logger.log(event)
    return perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--generation', type=int, help='checkpoint_genNNN.pt (default: latest.pt)')
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = (args.run_dir / 'checkpoints' /
                  (f'checkpoint_gen{args.generation:03d}.pt' if args.generation is not None
                   else 'latest.pt'))
    state = load_training_state(checkpoint)
    optimizer_state = [v for s in state.optimizer.state.values() for v in s.values()
                       if torch.is_tensor(v)]
    weights_sub, weights_total = count_subnormal(state.model.state_dict().values())
    adam_sub, adam_total = count_subnormal(optimizer_state)
    report = {'checkpoint': str(checkpoint),
              'subnormal_weights': [weights_sub, weights_total],
              'subnormal_adam_state': [adam_sub, adam_total],
              'train_seconds': time_training(checkpoint, args.steps, False),
              'train_seconds_flush_denormal': time_training(checkpoint, args.steps, True)}
    metrics = args.run_dir / 'metrics.jsonl'
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / 'metrics_copy.jsonl'
        shutil.copyfile(metrics, big)
        report['metrics_size_mb'] = round(metrics.stat().st_size / 1e6, 2)
        report['append_seconds_large_file'] = time_appends(big, args.steps)
        report['append_seconds_small_file'] = time_appends(Path(tmp) / 'small.jsonl', args.steps)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
