"""The web's legacy opponent: TB-RT30-v4 (the previous web engine, TB-RT30-v3, under TB-V3's native builder).

Status comes from v4's same-move gate record (``scripts/check_rt30_v4_gate.py``), not from ``tb_rt30_v4.STATUS``
(that file is in v4's own lock, so recording the gate there would change v4):
    legacy-certified   ``docs/rt30-results/v4_same_move.json`` says V4_SAME_MOVE_PASS for exactly the v4 lock of
                       this checkout, and every locked file is that one
    candidate          otherwise (no record, FAIL, or another v4): the web does not serve it unless told to
                       (``--rt30-allow-candidate``, local tests only)

``LazySupervisor`` starts the engine worker (and its V8-B helper) only when a game first asks it for a move, and
closes it after ``idle_seconds`` without a search (the web passes its session TTL, so every game that used it has
expired by then). The latest-certified TB-V3 worker stays preloaded; this one costs nothing until it is chosen.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from time import monotonic
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GATE_RECORD = ROOT / 'docs' / 'rt30-results' / 'v4_same_move.json'
LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v4.sha256'
FORMAT = 'tb-rt30-v4-same-move-v1'
LABEL = '이전 웹 모델 - TB-RT30-v4 Legacy'
CANDIDATE_LABEL = 'TB-RT30-v4 (후보, same-move gate 전)'


def lock_entries() -> dict[str, str]:
    return dict(reversed(line.split(maxsplit=1)) for line in LOCK.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#'))


def status(record_path: Path = GATE_RECORD) -> tuple[str, str]:
    """(``'legacy-certified'`` or ``'candidate'``, the reason)."""
    lock = lock_entries()
    bad = [p for p, digest in lock.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    if bad:
        return 'candidate', f'locked files changed: {bad[:3]}'
    if not Path(record_path).is_file():
        return 'candidate', f'no same-move gate record ({Path(record_path).name})'
    record = json.loads(Path(record_path).read_text(encoding='utf-8'))
    if record.get('format') != FORMAT or record.get('verdict') != 'V4_SAME_MOVE_PASS':
        return 'candidate', f"gate verdict {record.get('verdict')!r}"
    if record.get('v4_lock') != lock:
        return 'candidate', 'the gate record was made on another v4'
    return 'legacy-certified', f"same-move gate PASS ({record.get('compared')} comparisons)"


class LazySupervisor:
    """An ``EngineSupervisor`` built on the first search and closed after ``idle_seconds`` without one."""

    def __init__(self, make, *, idle_seconds: float = 3600.0, log=print):
        self._make, self.idle_seconds, self.log = make, idle_seconds, log
        self._lock = threading.Lock()
        self._supervisor = None
        self._busy = 0
        self._last = monotonic()
        self.starts = 0
        self.closed_idle = 0
        self._timer = None
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self._supervisor is not None

    def _ensure(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('the legacy engine is closed')
            created = self._supervisor is None
            if created:
                started = monotonic()
                self.log('[rt30] legacy engine: starting its worker (first use)', flush=True)
                self._supervisor = self._make()
                self.starts += 1
                self.log(f'[rt30] legacy engine: worker ready in {monotonic() - started:.1f} s', flush=True)
            self._busy += 1
            return self._supervisor, created

    def select(self, *, game_key, seed, history, received_at=None):
        supervisor, created = self._ensure()
        try:
            if created:
                received_at = monotonic()  # its deadline counts from a ready worker, not from the cold start
            return supervisor.select(game_key=game_key, seed=seed, history=history, received_at=received_at)
        finally:
            with self._lock:
                self._busy -= 1
                self._last = monotonic()
            self._arm()

    def _arm(self) -> None:
        if self.idle_seconds <= 0:
            return
        timer = threading.Timer(self.idle_seconds, self._close_if_idle)
        timer.daemon = True
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = timer
        timer.start()

    def _close_if_idle(self) -> None:
        with self._lock:
            if self._supervisor is None or self._busy or monotonic() - self._last < self.idle_seconds:
                return
            supervisor, self._supervisor = self._supervisor, None
            self.closed_idle += 1
        self.log(f'[rt30] legacy engine: idle {self.idle_seconds:g} s, worker closed', flush=True)
        supervisor.close()

    def stats(self) -> dict[str, Any]:
        supervisor = self._supervisor
        out = {'loaded': supervisor is not None, 'starts': self.starts, 'closed_idle': self.closed_idle,
               'idle_seconds': self.idle_seconds}
        if supervisor is not None:
            out.update(supervisor_stats(supervisor))
        return out

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
            supervisor, self._supervisor = self._supervisor, None
        if supervisor is not None:
            supervisor.close()


def supervisor_stats(supervisor) -> dict[str, Any]:
    """Worker state of an ``EngineSupervisor`` for /api/engines."""
    worker = getattr(supervisor, '_worker', None)
    alive = bool(worker is not None and worker.alive())
    return {'loaded': True, 'ready': alive and getattr(worker, 'pid', None) is not None,
            'worker_pid': getattr(worker, 'pid', None), 'spare_ready': supervisor.spare_ready(),
            'restarts': supervisor.restarts, 'watchdogs': supervisor.watchdogs}
