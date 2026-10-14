"""Engine worker processes for the web server: absolute deadline, best-so-far, 29 s watchdog (docs/web-ser5.md §3.6).

The web process never searches. ``EngineSupervisor.select`` sends the game (move history) and the
**absolute** deadlines to a preloaded worker process and returns as soon as one of these happens:

- the worker returns its move (the normal path: the engine's own deadline stopped the search);
- the watchdog time passes (``received + watchdog_seconds``): the last best-so-far move the worker
  reported is played at once, and the worker is killed and replaced *after* the answer is chosen;
- the worker dies or reports an error: the best-so-far move, else a fallback made here.

Times are ``time.monotonic()`` values; on Linux and Windows that clock is system-wide, so the
worker reads the same deadline the server stamped when the request arrived. Queue time (waiting
for the worker) is inside the deadline: it shortens the search, it never extends the answer.

One worker searches at a time (concurrency 1, the mini PC's cores are not shared between
searches); a warm spare is kept loaded so a killed worker is replaced without a model load on the
next request. Workers run one thread per process for torch and the BLAS libraries.

``AgentSpec`` names a factory ``"module:function"`` and its keyword arguments; the factory runs in
the worker (so the policy checkpoint is loaded there, never in the web process).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


@dataclass(frozen=True)
class AgentSpec:
    factory: str                      # "module:function"; called as factory(seed=..., **kwargs)
    kwargs: dict = field(default_factory=dict)


@dataclass
class MoveResult:
    move: tuple[int, int]
    source: str                       # 'engine' / 'watchdog' / 'worker_error' / 'no_worker' / 'fallback'
    diagnostics: dict = field(default_factory=dict)
    rt: dict = field(default_factory=dict)
    best: list = field(default_factory=list)   # [(stage, move)] as reported by the worker
    queue_seconds: float | None = None         # received -> worker started the search
    engine_seconds: float | None = None        # the worker's select_move (None: no result)
    wall_seconds: float = 0.0                  # received -> move ready here
    deadline_seconds: float | None = None
    watchdog: bool = False
    worker_restarted: bool = False
    error: str | None = None


# -- worker process ------------------------------------------------------------
def _load_factory(name: str):
    module, _, function = name.partition(':')
    return getattr(__import__(module, fromlist=[function]), function)


def _json_value(value):
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _worker_main(conn, spec: AgentSpec, src_paths: list[str]) -> None:  # pragma: no cover - runs in a child
    os.environ.update(THREAD_ENV)  # before torch / numpy are imported in this process
    import sys

    for path in src_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    from renju import Game

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    factory = _load_factory(spec.factory)
    agents: dict[str, Any] = {}  # one agent per game: its random stream continues across the moves
    conn.send(('ready', os.getpid()))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        request_id, game_key, seed, history, deadline_at = message
        started = monotonic()
        conn.send(('started', request_id, started))
        try:
            agent = agents.get(game_key)
            if agent is None:
                if len(agents) >= 8:
                    agents.pop(next(iter(agents)))
                agent = agents[game_key] = factory(seed=seed, **spec.kwargs)
            game = Game()
            for move in history:
                game.play(*move)

            def progress(stage, move):
                conn.send(('progress', request_id, stage, [int(move[0]), int(move[1])]))

            if getattr(agent, 'supports_deadline', False):
                from .deadline import Deadline

                deadline = None if deadline_at is None else Deadline(deadline_at)
                move = agent.select_move(game, deadline=deadline, progress=progress)
            else:
                move = agent.select_move(game)
            seconds = monotonic() - started
            diagnostics = getattr(agent, 'diagnostics', None)
            diagnostics = (diagnostics if isinstance(diagnostics, dict) else
                           vars(diagnostics) if diagnostics is not None else {})
            info = getattr(agent, 'rt_info', None)
            rt = vars(info) if info is not None else {}
            conn.send(('result', request_id, [int(move[0]), int(move[1])], seconds,
                       _json_value(diagnostics), _json_value(rt), getattr(agent, 'name', type(agent).__name__)))
        except Exception:  # noqa: BLE001 - reported to the server, the worker stays up
            conn.send(('error', request_id, traceback.format_exc(limit=5)))


class _Worker:
    def __init__(self, ctx, spec: AgentSpec, src_paths: list[str]):
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker_main, args=(child, spec, src_paths), daemon=True,
                                   name='rt30-engine-worker')
        self.process.start()
        child.close()
        self.pid = None

    def wait_ready(self, timeout: float) -> bool:
        if self.pid is not None:
            return True
        if self.conn.poll(timeout):
            try:
                kind, pid = self.conn.recv()
            except (EOFError, OSError):
                return False
            if kind == 'ready':
                self.pid = pid
                return True
        return False

    def alive(self) -> bool:
        return self.process.is_alive()

    def kill(self) -> None:
        try:
            self.process.terminate()
            self.process.join(1.0)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(1.0)
        finally:
            self.conn.close()

    def close(self) -> None:
        try:
            self.conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self.process.join(2.0)
        if self.process.is_alive():
            self.kill()


# -- supervisor ----------------------------------------------------------------
class EngineSupervisor:
    def __init__(self, spec: AgentSpec, *, engine_deadline: float = 27.5, watchdog: float = 29.0,
                 spare: bool = True, load_timeout: float = 120.0, src_paths: list[str] | None = None,
                 fallback=None, log=print):
        if not 0 < engine_deadline < watchdog:
            raise ValueError('need 0 < engine_deadline < watchdog')
        self.spec, self.engine_deadline, self.watchdog = spec, engine_deadline, watchdog
        self.spare_enabled, self.load_timeout, self.log = spare, load_timeout, log
        self.fallback = fallback or fallback_move
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
        self.src_paths = src_paths or [here]
        self._ctx = mp.get_context('spawn')  # same behaviour on Linux and Windows; torch-safe
        self._lock = threading.Lock()        # one search at a time
        self._spare_lock = threading.Lock()
        self._request = 0
        self.restarts = 0
        self.watchdogs = 0
        self._worker = _Worker(self._ctx, spec, self.src_paths)
        self._spare: _Worker | None = None
        self._spare_starting = False
        self._closed = False
        if not self._worker.wait_ready(load_timeout):
            raise RuntimeError('engine worker did not start')
        self._ensure_spare()

    # spares are started in the background so a restart never waits for a model load
    def _ensure_spare(self) -> None:
        if not self.spare_enabled:
            return

        with self._spare_lock:  # held only for these checks, never during a model load
            if self._spare_starting or (self._spare is not None and self._spare.alive()):
                return
            self._spare_starting = True

        def start():
            spare = _Worker(self._ctx, self.spec, self.src_paths)
            ready = spare.wait_ready(self.load_timeout)
            with self._spare_lock:
                self._spare_starting = False
                if ready and not self._closed:
                    self._spare = spare
                    return
            spare.kill()

        threading.Thread(target=start, name='rt30-spare', daemon=True).start()

    def _replace_worker(self) -> None:
        old = self._worker
        threading.Thread(target=old.kill, name='rt30-kill', daemon=True).start()  # never wait for it
        with self._spare_lock:
            spare, self._spare = self._spare, None
        if spare is not None and spare.alive():
            self._worker = spare
        else:
            self._worker = _Worker(self._ctx, self.spec, self.src_paths)  # cold start: waits on next request
        self.restarts += 1
        self._ensure_spare()

    def select(self, *, game_key: str, seed: int, history, received_at: float | None = None) -> MoveResult:
        """The move for the position after ``history`` (list of (row, col)), answered by
        ``received_at + watchdog`` at the latest."""
        received = monotonic() if received_at is None else received_at
        deadline_at = received + self.engine_deadline
        watchdog_at = received + self.watchdog
        history = [tuple(m) for m in history]

        def finish(result: MoveResult) -> MoveResult:
            result.wall_seconds = monotonic() - received
            result.deadline_seconds = self.engine_deadline
            return result

        if not self._lock.acquire(timeout=max(0.0, watchdog_at - monotonic() - 0.5)):
            return finish(MoveResult(self.fallback(history), 'no_worker', error='queue: no worker before the watchdog'))
        try:
            return finish(self._run(game_key, seed, history, deadline_at, watchdog_at, received))
        finally:
            self._lock.release()

    def _run(self, game_key, seed, history, deadline_at, watchdog_at, received) -> MoveResult:
        worker = self._worker
        if not worker.wait_ready(max(0.0, watchdog_at - monotonic() - 0.5)):
            self._replace_worker()
            return MoveResult(self.fallback(history), 'no_worker', error='worker not ready', worker_restarted=True)
        self._request += 1
        request_id = self._request
        best: list = []
        started_at = None
        try:
            worker.conn.send((request_id, game_key, seed, history, deadline_at))
            while True:
                left = watchdog_at - monotonic()
                if left <= 0 or not worker.conn.poll(left):
                    break  # watchdog
                kind, rid, *payload = worker.conn.recv()
                if rid != request_id:
                    continue  # a late message of an earlier, abandoned request
                if kind == 'started':
                    started_at = payload[0]
                elif kind == 'progress':
                    best.append((payload[0], tuple(payload[1])))
                elif kind == 'result':
                    move, seconds, diagnostics, rt, name = payload
                    return MoveResult(tuple(move), 'engine', diagnostics=diagnostics, rt=rt, best=best,
                                      queue_seconds=None if started_at is None else started_at - received,
                                      engine_seconds=seconds)
                elif kind == 'error':
                    self.log(f'[rt30] engine worker error:\n{payload[0]}', flush=True)
                    return MoveResult(best[-1][1] if best else self.fallback(history), 'worker_error', best=best,
                                      error=payload[0].strip().splitlines()[-1],
                                      queue_seconds=None if started_at is None else started_at - received)
        except (EOFError, OSError, BrokenPipeError) as exc:
            self._replace_worker()
            return MoveResult(best[-1][1] if best else self.fallback(history), 'worker_error', best=best,
                              error=f'worker died: {type(exc).__name__}', worker_restarted=True)
        # Watchdog: answer first with what the worker reported, then replace it.
        self.watchdogs += 1
        move = best[-1][1] if best else self.fallback(history)
        result = MoveResult(move, 'watchdog' if best else 'fallback', best=best, watchdog=True,
                            queue_seconds=None if started_at is None else started_at - received)
        self._replace_worker()
        result.worker_restarted = True
        self.log(f'[rt30] watchdog after {self.watchdog:.1f}s: played {move} ({result.source}), worker replaced',
                 flush=True)
        return result

    def spare_ready(self) -> bool:
        with self._spare_lock:
            return self._spare is not None and self._spare.alive() and self._spare.pid is not None

    def close(self) -> None:
        with self._spare_lock:
            self._closed = True
        for worker in (self._worker, self._spare):
            if worker is not None:
                worker.close()


def fallback_move(history) -> tuple[int, int]:
    """Millisecond rule when no engine move exists: win now, else block a five, else the legal
    move nearest the last stone. Only for a dead or hung worker that reported nothing."""
    from renju import Game
    from search.mcts_v5 import _winning_moves

    game = Game()
    for move in history:
        game.play(*move)
    legal = game.legal_moves()
    legal_set = set(legal)
    for player in (game.to_play, -game.to_play):
        wins = [m for m in _winning_moves(game, player) if m in legal_set]
        if wins:
            return min(wins)
    last = game.history[-1] if game.history else (7, 7)
    return min(legal, key=lambda m: (max(abs(m[0] - last[0]), abs(m[1] - last[1])), m))


# -- agent factories (run in the worker) -------------------------------------------
def make_rt30_agent(seed: int = 42, *, checkpoint: str | None = None, uniform_prior: bool = False,
                    timing: dict | None = None, fast: bool = True, attack_order: str = 'frozen',
                    attack_scheduler: str = 'fair'):
    """TB-RT30 with the TB-Engine-v1 configuration: the H3 policy from ``checkpoint`` (hash-checked),
    or ``uniform_prior=True`` without one (not TB1: for testing the server without the checkpoint)."""
    from analysis.tb_engine_v1 import TB_ENGINE_V1

    from .engine import RT30Agent

    config = dict(TB_ENGINE_V1)
    policy = None
    if checkpoint:
        from analysis.tb_engine_v1 import check_checkpoint
        from hybrid.h4_policy import RootPolicy

        check_checkpoint(checkpoint)
        policy = _cached_policy(checkpoint)
    elif uniform_prior:
        config['puct_prior'] = 'uniform'
    else:
        raise ValueError('make_rt30_agent needs checkpoint=... or uniform_prior=True')
    if fast:
        from . import fast_windows

        fast_windows.install()
    agent = RT30Agent(seed=seed, root_policy=policy, timing=timing, attack_order=attack_order,
                      attack_scheduler=attack_scheduler, **config)
    agent.supports_deadline = True
    agent.name = 'TB-RT30' if checkpoint else 'TB-RT30 (uniform prior)'
    return agent


_POLICIES: dict[str, Any] = {}


def _cached_policy(checkpoint: str):
    if checkpoint not in _POLICIES:
        from hybrid.h4_policy import RootPolicy

        _POLICIES[checkpoint] = RootPolicy(checkpoint, threads=1)
    return _POLICIES[checkpoint]
