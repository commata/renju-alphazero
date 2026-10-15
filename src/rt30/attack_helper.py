"""X2: V8-B in a resident helper process beside the main search (docs/web-ser5.md §9.8).

One helper per engine process, started on first use and kept for every later move (starting a
process per move would cost more than most V8-B searches). A request carries the game's move
history, V8-B's budgets and the absolute deadline (``time.monotonic`` is one clock for every
process of the machine); the helper replays the moves, runs ``rt30.engine._own_vct_attack`` (the
deadline-aware copy of the frozen V8-B: same candidates, budgets and fair scheduler) with every
exact speed-up installed, and answers with the move (a proven WIN or None), whether time cut it,
and V8-B's diagnostics. Answers carry the request id, so a late answer to an earlier move is dropped.

A helper that cannot start (for example inside a daemonic process, which may not have children)
or that dies is reported as unavailable; the engine then runs V8-B inline, as TB-RT30-v1 does.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import sys
import traceback
from time import monotonic, perf_counter

ATTACK_FIELDS = ('v8_attack_candidates', 'v8_attack_checked', 'v8_attack_calls', 'v8_attack_nodes',
                 'v8_attack_budget_exhausted', 'v8_attack_seconds', 'v8_attack_status', 'v8_attack_move',
                 'v8_attack_rank')


def _helper_main(conn, paths):
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    from analysis.mcts_v8 import SearchDiagnostics
    from renju import Game
    from search.mcts_v5 import _RootContext

    from rt30 import engine, fast3
    from rt30.deadline import Deadline

    fast3.install_all()
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message[0] == 'stop':
            return
        _, request_id, history, params, at = message
        try:
            started = perf_counter()
            game = Game()
            for move in history:
                game.play(*move)
            diag = SearchDiagnostics()
            context = _RootContext(game.legal_moves(), diag)
            move, cut = engine._own_vct_attack(
                game, context, diag, node_limit=params['node_limit'], call_limit=params['call_limit'],
                node_budget=params['node_budget'], deadline=None if at is None else Deadline(at),
                order=params.get('order', 'frozen'), scheduler='fair')
            answer = {'move': None if move is None else [int(move[0]), int(move[1])], 'timed_out': cut,
                      'diag': {k: getattr(diag, k) for k in ATTACK_FIELDS},
                      'seconds': perf_counter() - started}
            conn.send(('result', request_id, answer))
        except Exception:  # noqa: BLE001 - reported to the engine, which falls back
            conn.send(('error', request_id, traceback.format_exc()))


class AttackHelper:
    def __init__(self):
        ctx = mp.get_context('spawn')  # as rt30.supervisor: same behaviour on Linux and Windows
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_helper_main, args=(child, list(sys.path)), daemon=True,
                                   name='rt30-attack-helper')
        self.process.start()
        child.close()
        self._next = 0
        self.last_error: str | None = None

    def alive(self) -> bool:
        return self.process.is_alive()

    def submit(self, history, params: dict, at: float | None) -> int:
        self._next += 1
        self.conn.send(('attack', self._next, [list(m) for m in history], params, at))
        return self._next

    def result(self, request_id: int, until: float | None = None) -> dict | None:
        """The answer to ``request_id``; None when ``until`` (monotonic) passes first or the helper died."""
        while True:
            timeout = None if until is None else max(0.0, until - monotonic())
            try:
                ready = self.conn.poll(timeout)
            except (EOFError, OSError):
                return None
            if not ready:
                return None
            try:
                kind, answer_id, payload = self.conn.recv()
            except (EOFError, OSError):
                return None
            if answer_id != request_id:
                continue  # a late answer to an earlier move
            if kind == 'error':
                self.last_error = payload
                return None
            return payload

    def close(self):
        try:
            self.conn.send(('stop',))
        except (OSError, ValueError):
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()


_HELPER: AttackHelper | None = None
_UNAVAILABLE: str | None = None


def get_helper() -> AttackHelper | None:
    """This process's helper (started on first use); None when it cannot run here."""
    global _HELPER, _UNAVAILABLE
    if _HELPER is not None and _HELPER.alive():
        return _HELPER
    if _UNAVAILABLE is not None:
        return None
    try:
        _HELPER = AttackHelper()
    except (AssertionError, OSError, RuntimeError) as exc:  # e.g. 'daemonic processes are not allowed to have children'
        _UNAVAILABLE = f'{type(exc).__name__}: {exc}'
        _HELPER = None
    return _HELPER


def unavailable_reason() -> str | None:
    return _UNAVAILABLE


@atexit.register
def _close():
    if _HELPER is not None:
        _HELPER.close()
