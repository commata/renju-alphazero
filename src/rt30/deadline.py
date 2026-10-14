"""Absolute wall-clock deadline shared by every search phase of one move (docs/web-ser5.md §3.2).

The server stamps ``Deadline.at_monotonic(received + 27.5)`` when the request arrives, so queue
time counts; the worker gets the absolute time, never a relative budget. ``CheckLimit`` expires
after a fixed number of checks instead of a time: tests use it to cut a search at every
possible point and get the same cut on every run.
"""
from __future__ import annotations

from time import monotonic

from analysis.mcts_v8 import _BudgetExhausted


class SearchTimeout(_BudgetExhausted):
    """A deadline cut. Subclasses the budget-cut exception, so ``_BudgetedSolver._bounded`` turns
    it into UNKNOWN and nothing on its way up is cached (docs/web-ser5.md §3.2)."""


class Deadline:
    def __init__(self, at: float, clock=monotonic):
        self.at, self.clock = at, clock

    @classmethod
    def after(cls, seconds: float, clock=monotonic) -> Deadline:
        return cls(clock() + seconds, clock)

    def remaining(self) -> float:
        return self.at - self.clock()

    def expired(self) -> bool:
        return self.clock() >= self.at


class CheckLimit:
    """Expires on the ``limit``-th call of ``expired`` (0: already expired)."""

    def __init__(self, limit: int):
        self.limit, self.checks = limit, 0

    def remaining(self) -> float:
        return float(self.limit - self.checks)

    def expired(self) -> bool:
        self.checks += 1
        return self.checks > self.limit


def check(deadline) -> None:
    if deadline is not None and deadline.expired():
        raise SearchTimeout
