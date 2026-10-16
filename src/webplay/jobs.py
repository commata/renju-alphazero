"""The AI job queue of the web server (docs/web-ser5.md §8).

A human move that needs an AI reply becomes a job. Search slots (``slots`` dispatcher threads)
take the jobs in arrival order and run ``job.run(slot)`` without any session lock held, so polling
(``/api/state``) never waits for a search. The engine deadline starts when a slot takes the job,
not when the move arrived: waiting in the queue never shortens the search.

- **Room first.** ``reserve()`` takes a queue place *before* the human move is applied and raises
  ``QueueFull`` (HTTP 503) when there is none, so a refused request leaves the game as it was.
  The place is then filled with ``submit(ticket, job)`` or given back with ``release(ticket)``.
- **One per session.** The session refuses a second request while it has a job (HTTP 409).
- **Slot affinity.** A job may name a ``slot`` (the slot that searched the game's earlier moves:
  an engine worker keeps one agent, and its random stream, per game). A slot takes the oldest job
  that names it or no slot; with one slot (the default) this is plain FIFO.
- **Cancel.** ``cancel(job)`` removes a job that no slot has taken yet (a new game while queued).
"""
from __future__ import annotations

import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable


class QueueFull(Exception):
    """Every queue place is taken."""


@dataclass
class AIJob:
    run: Callable[[int], None]      # called once in a slot thread, no lock held
    session: Any = None
    slot: int | None = None
    created_at: float = field(default_factory=monotonic)
    started_at: float | None = None


class _Ticket:
    __slots__ = ('used',)

    def __init__(self):
        self.used = False


class AIJobQueue:
    def __init__(self, *, max_queue: int = 4, slots: int = 1, log=print):
        if max_queue < 1 or slots < 1:
            raise ValueError('need max_queue >= 1 and slots >= 1')
        self.max_queue, self.slots, self.log = max_queue, slots, log
        self._cond = threading.Condition()
        self._jobs: deque[AIJob] = deque()
        self._reserved = 0
        self._running: dict[int, AIJob] = {}
        self._closed = False
        self.done = 0
        self.failed = 0
        self._threads = [threading.Thread(target=self._loop, args=(slot,), name=f'ai-slot-{slot}', daemon=True)
                         for slot in range(slots)]
        for thread in self._threads:
            thread.start()

    # -- producer side ----------------------------------------------------------
    def reserve(self) -> _Ticket:
        with self._cond:
            if self._closed:
                raise QueueFull('the server is stopping')
            if len(self._jobs) + self._reserved >= self.max_queue:
                raise QueueFull(f'{self.max_queue} AI moves are already waiting')
            self._reserved += 1
            return _Ticket()

    def release(self, ticket: _Ticket) -> None:
        with self._cond:
            if not ticket.used:
                ticket.used = True
                self._reserved -= 1

    def submit(self, ticket: _Ticket, job: AIJob) -> None:
        with self._cond:
            if ticket.used:
                raise RuntimeError('ticket already used')
            ticket.used = True
            self._reserved -= 1
            self._jobs.append(job)
            self._cond.notify_all()

    def cancel(self, job: AIJob) -> bool:
        """Remove ``job`` if no slot has taken it; True when it was removed."""
        with self._cond:
            try:
                self._jobs.remove(job)
            except ValueError:
                return False
            self._cond.notify_all()
            return True

    def position(self, job: AIJob) -> int | None:
        """1 for the next job to start; None when it is running or gone."""
        with self._cond:
            for index, queued in enumerate(self._jobs):
                if queued is job:
                    return index + 1
            return None

    def stats(self) -> dict:
        with self._cond:
            return {'queued': len(self._jobs), 'reserved': self._reserved, 'running': len(self._running),
                    'slots': self.slots, 'max_queue': self.max_queue, 'done': self.done, 'failed': self.failed}

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            self._jobs.clear()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout)

    # -- slots --------------------------------------------------------------------
    def _take(self, slot: int) -> AIJob | None:
        with self._cond:
            while True:
                if self._closed:
                    return None
                for job in self._jobs:
                    if job.slot is None or job.slot == slot:
                        self._jobs.remove(job)
                        job.slot = slot
                        job.started_at = monotonic()
                        self._running[slot] = job
                        return job
                self._cond.wait()

    def _loop(self, slot: int) -> None:
        while True:
            job = self._take(slot)
            if job is None:
                return
            try:
                job.run(slot)
                ok = True
            except Exception:  # noqa: BLE001 - the session reports its own errors; this is a last resort
                ok = False
                self.log(f'[web] AI job failed in slot {slot}:\n{traceback.format_exc(limit=5)}', flush=True)
            with self._cond:
                self._running.pop(slot, None)
                self.done += ok
                self.failed += not ok
                self._cond.notify_all()
