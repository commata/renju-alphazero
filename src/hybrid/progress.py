"""Progress lines for long offline jobs (anything expected to take an hour or more).

Printed to stdout (``flush=True``) so ``2>&1 | Tee-Object`` on Windows shows them without
PowerShell turning them into error records. One line every ``every_seconds`` and at the end:

    [proofs] 12,000/109,264 games  11.0%  8.2 min  ETA 66 min

With a process pool, use ``run_chunks`` (results as they complete, put back in input order) or wait on
futures with a timeout and call ``heartbeat``: a line is then printed every ``every_seconds`` even while
no result arrives (one slow chunk must not silence the log).
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
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

    def heartbeat(self, extra: str = '') -> None:
        """A line with the current numbers when ``every_seconds`` passed without an update."""
        now = perf_counter()
        if now - self.last >= self.every:
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


def run_chunks(pool, fn, chunks: list, progress: Progress, amount, *, unit_name: str = 'chunks') -> list:
    """``[fn(c) for c in chunks]`` on ``pool``, collected as each finishes (no head-of-line wait);
    ``amount(chunk)`` is the progress credit of a finished chunk. A heartbeat line every
    ``progress.every`` seconds while nothing finishes. The result keeps the input order."""
    futures = {pool.submit(fn, chunk): i for i, chunk in enumerate(chunks)}
    results = [None] * len(chunks)
    pending, finished = set(futures), 0
    try:
        while pending:
            done, pending = wait(pending, timeout=progress.every, return_when=FIRST_COMPLETED)
            for future in done:
                i = futures[future]
                results[i] = future.result()
                finished += 1
                progress.update(amount(chunks[i]), extra=f'{finished:,}/{len(chunks):,} {unit_name}')
            if not done:
                progress.heartbeat(f'{finished:,}/{len(chunks):,} {unit_name}, {len(pending):,} pending')
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    return results


def stage(label: str, started: float, log=print) -> None:
    """One line when a short stage after the long one finishes."""
    log(f'[{label}] done, {(perf_counter() - started) / 60:.1f} min', flush=True)
