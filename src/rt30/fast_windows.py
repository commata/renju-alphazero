"""Exact drop-in for ``search.mcts_v5._threat_windows`` (docs/web-ser5.md §3.1).

The frozen function scans all 572 five-cell windows in Python on every call; on the slow V8-B
positions of E3-A it is 77% of the run time. ``threat_windows`` returns the same windows in the
same order (``WINDOWS`` order), vectorized with numpy when it is importable, else with an
unrolled pure-Python loop (both are tested against the frozen function).

It is evaluated eagerly where the frozen one is a lazy generator. That is exact because no
caller changes the board while it iterates (every caller only reads cells inside the loop), and
the callers that stop early (``next(...)``) only lose the early exit.

``install()`` rebinds the name in every loaded module that imported the frozen function
(``search.mcts_v5`` itself, ``search.mcts_v7``, ``analysis.mcts_v8``, ...), so it changes this
process only. The web engine worker installs it; benchmark and gate runs choose explicitly.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

from renju import SIZE
from search import mcts_v5

FROZEN = mcts_v5._threat_windows
WINDOWS = mcts_v5.WINDOWS
_FLAT = tuple(tuple(r * SIZE + c for r, c in window) for window in WINDOWS)

try:
    import numpy as _np
except ImportError:  # numpy is not a declared dependency; the pure version is the fallback
    _np = None


def threat_windows_pure(game, player: int, minimum: int = 3):
    # Board values are -1/0/1; multiplied by ``player`` an opponent stone is negative.
    flat = [value * player for row in game.board for value in row]
    out = []
    for index, (a, b, c, d, e) in enumerate(_FLAT):
        va, vb, vc, vd, ve = flat[a], flat[b], flat[c], flat[d], flat[e]
        if va >= 0 and vb >= 0 and vc >= 0 and vd >= 0 and ve >= 0 and va + vb + vc + vd + ve >= minimum:
            out.append(WINDOWS[index])
    return iter(out)


if _np is not None:
    _INDEX = _np.array(_FLAT, dtype=_np.intp)

    def threat_windows_numpy(game, player: int, minimum: int = 3):
        values = _np.array(game.board, dtype=_np.int8).ravel()[_INDEX]
        keep = ~(values == -player).any(axis=1) & ((values == player).sum(axis=1) >= minimum)
        return iter([WINDOWS[i] for i in _np.flatnonzero(keep).tolist()])

    threat_windows = threat_windows_numpy
else:
    threat_windows_numpy = None
    threat_windows = threat_windows_pure


def _bindings(target):
    # Every module-level name bound to ``target``, except this module's own (``FROZEN`` etc.).
    return [(module, name) for module in list(sys.modules.values())
            if module is not None and getattr(module, '__name__', None) != __name__
            for name, value in list(getattr(module, '__dict__', {}).items()) if value is target]


def installed() -> bool:
    return mcts_v5._threat_windows is not FROZEN


def install(implementation=None) -> int:
    """Rebind the frozen function to ``implementation`` (default ``threat_windows``) in every
    loaded module; returns how many names were rebound. Import the engine modules first."""
    implementation = implementation or threat_windows
    bound = _bindings(FROZEN)
    for module, name in bound:
        setattr(module, name, implementation)
    return len(bound)


def uninstall() -> int:
    """Put the frozen function back wherever a replacement from this module is bound."""
    replacements = {threat_windows_pure, threat_windows_numpy} - {None}
    count = 0
    for target in replacements:
        for module, name in _bindings(target):
            setattr(module, name, FROZEN)
            count += 1
    return count


@contextmanager
def fast_windows(implementation=None):
    install(implementation)
    try:
        yield
    finally:
        uninstall()
