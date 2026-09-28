"""Stage 7 batch microbenchmark (plan §5, completion criterion 4).

Evaluates the same fixed positions with ``PolicyValueEvaluator.evaluate_batch`` at batch
sizes B = 1/2/4/8/16/32 and reports, per B:

- ``full``: evaluate_batch end to end (legal mask + encode + stack + forward + masked
  softmax + CPU transfer), the cost PUCT pays per leaf evaluation;
- ``forward``: the NN forward alone on pre-built input planes;

as median latency per batch, ms per position and positions per second, for each torch
thread count requested. It also records the numerical contract of plan §5.1: the max
|prior| and |value| difference between batched and B=1 outputs for the same positions
(float32 kernels may differ slightly with B; bit-exactness is not required).

Nothing in search or training changes; this only measures whether batching would pay
off on this machine before Stage 8 builds batched/parallel self-play.

    python scripts/benchmark_stage7_batch.py --checkpoint runs/stage7d_b32/checkpoints/checkpoint_gen080.pt \
        --threads 1 4 --output runs/stage7_batch_benchmark.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from model.config import ModelConfig  # noqa: E402
from model.evaluator import PolicyValueEvaluator  # noqa: E402
from model.network import PolicyValueNet  # noqa: E402
from profile_stage7_search import load_positions  # noqa: E402
from search.evaluator import EvaluationSnapshot  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402


def snapshots_for(count: int) -> list[EvaluationSnapshot]:
    games = load_positions(None)
    if len(games) < count:
        raise SystemExit(f'need {count} positions, have {len(games)}')
    return [EvaluationSnapshot.from_game(g, g.legal_moves()) for g in games[:count]]


def _median_seconds(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        fn()
        samples.append(perf_counter() - started)
    return statistics.median(samples)


def benchmark(model, snapshots, batch_sizes, repeats: int, warmup: int) -> list[dict]:
    evaluator = PolicyValueEvaluator(model)
    single = [evaluator.evaluate_batch([s])[0] for s in snapshots]
    rows = []
    for batch in batch_sizes:
        groups = [snapshots[i:i + batch] for i in range(0, len(snapshots) - batch + 1, batch)]
        encoded = [[evaluator.encode(s) for s in group] for group in groups]
        planes = [torch.stack([x for x, _ in enc]) for enc in encoded]

        def run_full():
            for group in groups:
                evaluator.evaluate_batch(group)

        def run_forward():
            with torch.inference_mode():
                for x in planes:
                    model(x)

        for _ in range(warmup):
            run_full()
            run_forward()
        positions = batch * len(groups)
        full = _median_seconds(run_full, repeats)
        forward = _median_seconds(run_forward, repeats)
        batched = [r for group in groups for r in evaluator.evaluate_batch(group)]
        max_prior = max(max(abs(a - b) for a, b in zip(r.priors, s.priors))
                        for r, s in zip(batched, single))
        max_value = max(abs(r.value - s.value) for r, s in zip(batched, single))
        rows.append({
            'batch': batch, 'batches': len(groups), 'positions': positions,
            'full_latency_ms': 1000 * full / len(groups),
            'full_ms_per_position': 1000 * full / positions,
            'full_positions_per_second': positions / full,
            'forward_latency_ms': 1000 * forward / len(groups),
            'forward_ms_per_position': 1000 * forward / positions,
            'forward_positions_per_second': positions / forward,
            'max_abs_prior_diff_vs_b1': max_prior,
            'max_abs_value_diff_vs_b1': max_value,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Stage 6/7 training checkpoint')
    source.add_argument('--random-init', type=int, metavar='SEED')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--positions', type=int, default=128,
                        help='fixed positions evaluated per measurement (multiple of max B)')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    if args.checkpoint is not None:
        model, info = load_model_from_training_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(args.random_init)
        model = PolicyValueNet(ModelConfig()).eval()
        info = {'random_init_seed': args.random_init}
    snapshots = snapshots_for(args.positions)

    runs = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        rows = benchmark(model, snapshots, args.batch_sizes, args.repeats, args.warmup)
        runs.append({'torch_threads': threads, 'rows': rows})
        print(f'threads {threads}')
        print('   B | full ms/batch | full ms/pos | full pos/s | fwd ms/pos | fwd pos/s | '
              'max|dprior| max|dvalue|')
        for r in rows:
            print(f"{r['batch']:4d} | {r['full_latency_ms']:13.2f} | {r['full_ms_per_position']:11.3f} | "
                  f"{r['full_positions_per_second']:10.0f} | {r['forward_ms_per_position']:10.3f} | "
                  f"{r['forward_positions_per_second']:9.0f} | {r['max_abs_prior_diff_vs_b1']:.1e} "
                  f"{r['max_abs_value_diff_vs_b1']:.1e}", flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format_version': 'stage7-batch-benchmark-v1', **info,
            'torch_version': str(torch.__version__), 'python_version': platform.python_version(),
            'platform': platform.platform(), 'processor': platform.processor(),
            'positions': len(snapshots), 'repeats': args.repeats, 'runs': runs}, indent=1),
            encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
