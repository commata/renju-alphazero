"""Progress lines for long offline jobs (anything expected to take an hour or more).

Printed to stdout (``flush=True``) so ``2>&1 | Tee-Object`` on Windows shows them without
PowerShell turning them into error records. One line every ``every_seconds`` and at the end:

    [proofs] 12,000/109,264 games  11.0%  8.2 min  ETA 66 min
"""
from __future__ import annotations

from time import perf_counter


class Progress:
    def __init__(self, label: str, total: float | None, unit: str = '', *, every_seconds: float = 60.0, log=print):
        self.label, self.total, self.unit = label, total, unit
        self.every, self.log = every_seconds, log
        self.done = 0.0
        self.started = self.last = perf_counter()

    def update(self, amount: float = 1, *, extra: str = '') -> None:
        self.done += amount
        now = perf_counter()
        if now - self.last >= self.every or (self.total is not None and self.done >= self.total):
            self.last = now
            self.log(self.line(now, extra), flush=True)

    def line(self, now: float | None = None, extra: str = '') -> str:
        elapsed = (now or perf_counter()) - self.started
        unit = f' {self.unit}' if self.unit else ''
        if self.total is None:  # total unknown: count and rate only
            text = f'[{self.label}] {self.done:,.0f}{unit}  {elapsed / 60:.1f} min'
            return f'{text}  {extra}' if extra else text
        share = self.done / self.total if self.total else 1.0
        eta = elapsed * (1 - share) / share if share > 0 else float('nan')
        text = (f'[{self.label}] {self.done:,.0f}/{self.total:,.0f}{unit}  {100 * share:.1f}%  '
                f'{elapsed / 60:.1f} min  ETA {eta / 60:.0f} min')
        return f'{text}  {extra}' if extra else text


def stage(label: str, started: float, log=print) -> None:
    """One line when a short stage after the long one finishes."""
    log(f'[{label}] done, {(perf_counter() - started) / 60:.1f} min', flush=True)
