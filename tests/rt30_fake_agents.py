"""Agents for tests/test_rt30_supervisor.py (imported by the engine worker processes)."""
import os
from time import monotonic, sleep


class _Agent:
    supports_deadline = True

    def __init__(self, seed, behaviour, seconds=0.0):
        self.seed, self.behaviour, self.seconds = seed, behaviour, seconds
        self.name = f'fake-{behaviour}'
        self.diagnostics = {}
        self.calls = 0

    def select_move(self, game, deadline=None, progress=None):
        self.calls += 1
        legal = game.legal_moves()
        first = min(legal)
        self.diagnostics = {'pid': os.getpid(), 'calls': self.calls, 'seed': self.seed,
                            'remaining': None if deadline is None else deadline.remaining()}
        if progress is not None:
            progress('emergency', first)
        if self.behaviour == 'quick':
            return first
        if self.behaviour == 'deadline':  # search until the engine deadline, like RT30
            while deadline.remaining() > 0.05:
                sleep(0.01)
            return max(legal)
        if self.behaviour == 'hang':      # ignores the deadline: only the watchdog stops it
            progress('puct', max(legal))
            sleep(3600)
        if self.behaviour == 'hang_silent':
            sleep(3600)
        if self.behaviour == 'crash':
            raise RuntimeError('boom')
        if self.behaviour == 'exit':
            os._exit(3)
        if self.behaviour == 'slow':
            sleep(self.seconds)
            return first
        raise ValueError(self.behaviour)


def make(seed=42, behaviour='quick', seconds=0.0):
    return _Agent(seed, behaviour, seconds)
