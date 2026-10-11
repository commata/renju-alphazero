"""Stage 8 MP: self-play in worker processes (docs/stage8-plan.md §12.24 step 3).

One generation's games are split over ``self_play_workers`` processes; each process
plays its share with the G0 lock-step engine on its own copy of the model (64x4 is a
few MB, so every process can hold one on the GPU). Execution only: the seeds are drawn
by the caller in the usual order and dealt round-robin (game ``i`` -> worker
``i % W``, which also spreads long and short games), and the games come back in seed
order. A game depends only on its seed and its evaluations, so on the CPU the records
equal the single-process ones; on a GPU a different batch composition can change float
rounding (grade E1, same as lock-step).

The pool is created once per training run and reused every generation (``spawn``
start method: the only one on Windows, and safe with CUDA). Each generation sends the
current weights as CPU tensors; a worker keeps its model and only loads the new state.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from time import perf_counter

import torch

from model.config import ModelConfig
from model.evaluator import PolicyValueEvaluator
from model.network import PolicyValueNet
from model.precision import set_tf32
from search.alphazero import SearchConfig, drive
from search.batched import LockstepStats, run_lockstep

from .self_play import self_play_steps

_WORKER: dict = {}


def _init_worker(device: str, threads: int, flush_denormal: bool, tf32: bool,
                 deterministic: bool) -> None:
    torch.set_num_threads(threads)
    torch.set_flush_denormal(flush_denormal)
    torch.use_deterministic_algorithms(deterministic)
    if device.startswith('cuda'):
        set_tf32(tf32)
    _WORKER.clear()
    _WORKER['device'] = device


def _model(model_config: dict, state_dict: dict):
    key = tuple(sorted(model_config.items()))
    if _WORKER.get('model_key') != key:
        _WORKER['model'] = PolicyValueNet(ModelConfig(**model_config)).to(_WORKER['device'])
        _WORKER['model_key'] = key
    model = _WORKER['model']
    model.load_state_dict(state_dict)
    return model.eval()


def play_share(model_config: dict, state_dict: dict, search: dict, seeds: list[int],
               parallel: int, checkpoint_hash: str | None, runtime_env: dict | None):
    """Worker task: play ``seeds`` (lock-step groups of ``parallel``); return the games,
    the lock-step statistics and the seconds this worker spent."""
    started = perf_counter()
    config = SearchConfig.from_dict(search)
    model = _model(model_config, state_dict)
    stats = LockstepStats()
    games = []
    with torch.inference_mode():
        evaluator = PolicyValueEvaluator(model, device=_WORKER['device'])
        for first in range(0, len(seeds), parallel):
            group = [self_play_steps(config, seed, checkpoint_hash=checkpoint_hash,
                                     runtime_env=runtime_env)
                     for seed in seeds[first:first + parallel]]
            games.extend(run_lockstep(group, evaluator, stats) if len(group) > 1
                         else [drive(group[0], evaluator)])
    if _WORKER['device'].startswith('cuda'):
        torch.cuda.synchronize()
    return games, stats, perf_counter() - started


def deal(seeds: list[int], workers: int) -> list[list[int]]:
    """Round-robin shares: share ``w`` holds seeds ``w, w + W, w + 2W, ...``."""
    return [seeds[w::workers] for w in range(workers) if seeds[w::workers]]


def undeal(shares: list[list], count: int) -> list:
    """Inverse of ``deal`` for the per-share results."""
    ordered = [None] * count
    workers = len(shares)
    for w, share in enumerate(shares):
        for k, item in enumerate(share):
            ordered[w + k * workers] = item
    return ordered


class SelfPlayPool:
    """``workers`` spawned processes playing self-play shares on ``device``."""

    def __init__(self, workers: int, device: str, *, threads: int = 1,
                 flush_denormal: bool = False, tf32: bool = False,
                 deterministic: bool | None = None):
        if type(workers) is not int or workers < 1:
            raise ValueError('workers must be a positive integer')
        if deterministic is None:
            deterministic = torch.are_deterministic_algorithms_enabled()
        self.workers = workers
        self.device = str(device)
        self._executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_worker,
            initargs=(self.device, threads, flush_denormal, tf32, deterministic))

    def play(self, model, search: SearchConfig, seeds: list[int], *, parallel: int,
             checkpoint_hash: str | None = None, runtime_env: dict | None = None
             ) -> tuple[list, dict]:
        """Play one game per seed; games in seed order plus combined statistics."""
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model_config = asdict(model.config)
        shares = deal(list(seeds), self.workers)
        futures = [self._executor.submit(play_share, model_config, state_dict,
                                         search.to_dict(), share, min(parallel, len(share)),
                                         checkpoint_hash, runtime_env)
                   for share in shares]
        results = [future.result() for future in futures]
        stats = LockstepStats()
        for _, share_stats, _ in results:
            stats.rounds += share_stats.rounds
            stats.evaluations += share_stats.evaluations
            stats.batch_sizes.update(share_stats.batch_sizes)
        games = undeal([games for games, _, _ in results], len(seeds))
        summary = {**stats.to_dict(), 'workers': len(shares),
                   'worker_seconds': [seconds for _, _, seconds in results]}
        return games, summary

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
