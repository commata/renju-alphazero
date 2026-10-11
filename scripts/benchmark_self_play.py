"""Stage 8 G0 self-play throughput: device x simulations x lock-step parallel games.

Plays the same ``--games`` self-play games (fixed seeds, the checkpoint's self-play
search with ``--simulations`` overridden) for every combination and reports wall time,
games/hour, moves/s, network evaluations/s and the mean lock-step batch. The records of
every run are compared with the first run of the same device and simulation count:
lock-step batching must not change a game (``same_records``).

Nothing is trained and no run directory is touched.

    python scripts/benchmark_self_play.py --checkpoint runs/anchors/ADA1360.pt \\
        --devices cpu cuda --simulations 50 200 400 --parallel 1 4 8 16 \\
        --output runs/gpu/self_play_benchmark.json

Stage 8 MP: ``--workers 1 2 3 4 6`` deals the games to worker processes (each plays its
share lock-step). ``same_records`` compares every row with the first row of the same
device and simulation count, across worker counts too.

    python scripts/benchmark_self_play.py --checkpoint runs/anchors/S400C.pt \\
        --devices cuda --simulations 400 --parallel 16 --workers 1 2 3 4 6 \\
        --output runs/gpu/mp_benchmark.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from model.evaluator import PolicyValueEvaluator  # noqa: E402
from model.precision import set_tf32  # noqa: E402
from search.alphazero import drive  # noqa: E402
from search.batched import LockstepStats, run_lockstep  # noqa: E402
from training.config import self_play_search_config  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.self_play import record_hash, self_play_steps  # noqa: E402
from training.workers import SelfPlayPool  # noqa: E402
from training.training_checkpoint import load_checkpoint_payload  # noqa: E402


def play(evaluator, search, seeds: list[int], parallel: int) -> tuple[list, LockstepStats, float]:
    stats = LockstepStats()
    games = []
    started = perf_counter()
    for first in range(0, len(seeds), parallel):
        group = [self_play_steps(search, seed) for seed in seeds[first:first + parallel]]
        if parallel > 1:
            games.extend(run_lockstep(group, evaluator, stats))
        else:
            games.append(drive(group[0], evaluator))
    if evaluator.device.type == 'cuda':
        torch.cuda.synchronize(evaluator.device)
    return games, stats, perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['cpu'])
    parser.add_argument('--simulations', type=int, nargs='+', default=[50])
    parser.add_argument('--parallel', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--workers', type=int, nargs='+', default=[1],
                        help='self-play processes (Stage 8 MP); each plays its share lock-step '
                             'with min(parallel, share) games')
    parser.add_argument('--games', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--tf32', action='store_true', help='allow TF32 on CUDA (default off)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_tf32(args.tf32)
    base = self_play_search_config(load_checkpoint_payload(args.checkpoint)['config'])
    rng = Random(args.seed)
    seeds = [rng.getrandbits(63) for _ in range(args.games)]
    model, info = load_model_from_training_checkpoint(args.checkpoint)
    references = {}
    rows = []
    for device in args.devices:
        evaluator = PolicyValueEvaluator(model, device=device)
        play(evaluator, replace(base, num_simulations=8), seeds[:2], 2)   # warm-up
        for workers in args.workers:
            pool = None
            if workers > 1:
                pool = SelfPlayPool(workers, device, threads=args.threads, tf32=args.tf32)
                pool.play(model, replace(base, num_simulations=8), seeds[:workers],
                          parallel=1)   # warm-up: spawn, imports, CUDA context
            try:
                for sims in args.simulations:
                    search = replace(base, num_simulations=sims)
                    for parallel in args.parallel:
                        if pool is None:
                            games, stats, seconds = play(evaluator, search, seeds, parallel)
                            mean_batch = stats.mean_batch if parallel > 1 else 1.0
                        else:
                            started = perf_counter()
                            games, summary = pool.play(model, search, seeds, parallel=parallel)
                            seconds = perf_counter() - started
                            mean_batch = summary['mean_batch'] if summary['rounds'] else 1.0
                        hashes = [record_hash(g.record) for g in games]
                        key = (device, sims)
                        references.setdefault(key, hashes)
                        moves = sum(len(g.record.moves) for g in games)
                        evaluations = sum(s.evaluator_calls for g in games for s in g.move_stats)
                        row = {'device': device, 'simulations': sims, 'workers': workers,
                               'parallel': parallel, 'games': len(games), 'moves': moves,
                               'seconds': seconds,
                               'games_per_hour': 3600 * len(games) / seconds,
                               'moves_per_s': moves / seconds,
                               'evaluations_per_s': evaluations / seconds,
                               'mean_batch': mean_batch,
                               'same_records': hashes == references[key]}
                        rows.append(row)
                        print(f"{device:5s} sims {sims:4d} workers {workers:2d} parallel "
                              f"{parallel:3d}: {seconds:7.1f} s, "
                              f"{row['games_per_hour']:7.0f} games/h, "
                              f"{row['moves_per_s']:6.1f} moves/s, "
                              f"{row['evaluations_per_s']:7.0f} evals/s, "
                              f"batch {row['mean_batch']:5.1f}, "
                              f"same records {row['same_records']}", flush=True)
            finally:
                if pool is not None:
                    pool.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'checkpoint': str(args.checkpoint), 'checkpoint_info': info,
            'torch': str(torch.__version__), 'threads': args.threads, 'tf32': args.tf32,
            'workers': args.workers,
            'cuda_device': (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                            else None),
            'base_search': base.to_dict(), 'seed': args.seed, 'rows': rows}, indent=1),
            encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
