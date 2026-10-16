"""X2's V8-B helper as a sibling of the web engine worker (docs/web-ser5.md §9.9, §9.11).

TB-RT30-v3 sends V8-B to a resident helper process (``rt30.attack_helper``). In evaluation runs the
engine process starts that helper itself. The web engine worker cannot: it is a daemonic process
(``rt30.supervisor``), and a daemonic process may not have children. So the supervisor starts the
helper **beside** the worker, from the web process, and hands the worker the helper's end of a pipe:

    web process (EngineSupervisor)
     ├─ engine worker  (daemon)  ── pipe ──┐
     └─ V8-B helper    (daemon)  ──────────┘   attack_helper._helper_main, the evaluated code

In the worker, ``install(conn, pid)`` makes ``attack_helper.get_helper()`` return a ``SiblingHelper``
on that pipe. It is ``AttackHelper`` itself (same ``submit`` / ``result`` messages and request ids)
except for how it checks and stops the process, which the worker did not start. Nothing of the
frozen v3 files changes: the helper runs ``attack_helper._helper_main`` and the engine
``engine_v3`` as evaluated. The supervisor kills and replaces the helper with its worker.
"""
from __future__ import annotations

import os

from . import attack_helper


class SiblingHelper(attack_helper.AttackHelper):
    """``AttackHelper`` on a pipe to a helper process started by the supervisor (not by this process)."""

    def __init__(self, conn, pid: int):  # noqa: D107 - no process is started here
        self.conn, self.pid = conn, pid
        self.process = None
        self._next = 0
        self.last_error = None

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        return not self.conn.closed

    def close(self):
        try:
            self.conn.send(('stop',))
        except (OSError, ValueError):
            pass


def install(conn, pid: int) -> SiblingHelper:
    """Make ``attack_helper.get_helper()`` in this process return the sibling helper."""
    helper = SiblingHelper(conn, pid)
    attack_helper._HELPER = helper
    attack_helper._UNAVAILABLE = None
    return helper


def start_helper(ctx, src_paths: list[str]):
    """Start a V8-B helper process (``attack_helper._helper_main``); returns ``(process, worker_end)``."""
    worker_end, helper_end = ctx.Pipe()
    process = ctx.Process(target=attack_helper._helper_main, args=(helper_end, list(src_paths)), daemon=True,
                          name='rt30-attack-helper')
    process.start()
    helper_end.close()
    return process, worker_end
