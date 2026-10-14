"""Exact drop-in for ``search.mcts_v5._threat_windows`` (docs/web-ser5.md §3.1).

The frozen function scans all 572 five-cell windows in Python on every call; on the slow V8-B
positions of E3-A it is 77% of the run time. ``threat_windows`` returns the same windows in the
same order (``WINDOWS`` order). Three versions, all tested against the frozen function:

- ``threat_windows_codes`` (default with numpy, 2nd speed-up §3.9): each window as a base-3 code
  (empty 0, black 1, white 2) and one lookup in a 243-entry table per (player, minimum);
- ``threat_windows_numpy`` (1st speed-up): the per-window counts with numpy reductions;
- ``threat_windows_pure``: an unrolled loop, used without numpy.

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

    _POW = _np.array([81, 27, 9, 3, 1], dtype=_np.int16)

    def _table(player: int, minimum: int):
        keep = _np.zeros(243, dtype=bool)
        for code in range(243):
            cells = [(code // 3 ** k) % 3 for k in range(4, -1, -1)]
            values = [0 if d == 0 else (1 if d == 1 else -1) for d in cells]
            keep[code] = -player not in values and values.count(player) >= minimum
        return keep

    _TABLES: dict = {}

    def threat_windows_codes(game, player: int, minimum: int = 3):
        table = _TABLES.get((player, minimum))
        if table is None:
            table = _TABLES[(player, minimum)] = _table(player, minimum)
        # -1 % 3 == 2: white is digit 2, black 1, empty 0.
        codes = (_np.array(game.board, dtype=_np.int8).ravel() % 3)[_INDEX].astype(_np.int16) @ _POW
        return iter([WINDOWS[i] for i in _np.flatnonzero(table[codes]).tolist()])

    threat_windows = threat_windows_codes
else:
    threat_windows_numpy = threat_windows_codes = None
    threat_windows = threat_windows_pure

REPLACEMENTS = tuple(f for f in (threat_windows_pure, threat_windows_numpy, threat_windows_codes) if f is not None)


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
    bound = [b for target in (FROZEN, *REPLACEMENTS) for b in _bindings(target)]  # also switches versions
    for module, name in bound:
        setattr(module, name, implementation)
    return len(bound)


def uninstall() -> int:
    """Put the frozen function back wherever a replacement from this module is bound."""
    count = 0
    for target in REPLACEMENTS:
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
