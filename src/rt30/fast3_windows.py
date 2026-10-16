"""Third exact speed-up (X1-A): the threat-window scan in C (docs/web-ser5.md §9.8).

``threat_windows_native`` returns exactly what ``search.mcts_v5._threat_windows`` yields, in
WINDOWS order, like the fast2 version (``fast_windows.threat_windows_codes``) it replaces. The
fast2 version copies the 15x15 Python board into a new numpy array on every call (about 2.6 calls
per VCF node); the C scan reads the board lists in place and allocates only the result list.
It is stateless (the board is written directly by many frozen helpers, so no incremental state is
kept) and never writes the board.

The module is optional: ``available()`` is False until ``python -m rt30.native.build`` has built
it, and ``rt30.fast3.install_all`` then keeps the fast2 scan (same moves, slower).
"""
from __future__ import annotations

from . import fast_windows

try:
    from .native import _threat_windows as _native
except ImportError:  # not built on this machine
    _native = None

WINDOWS = fast_windows.WINDOWS  # a tuple, never rebound (no module-level alias of a function here)
_INDEX = bytes(i for window in fast_windows._FLAT for i in window)


def _bindings(target):
    # As fast_windows._bindings, without this module's own names (``threat_windows_native`` itself).
    return [(module, name) for module, name in fast_windows._bindings(target) if module.__name__ != __name__]


def available() -> bool:
    return _native is not None


def threat_windows_native(game, player: int, minimum: int = 3):
    return iter(_native.windows(game.board, player, minimum, _INDEX, WINDOWS))


def installed() -> bool:
    return fast_windows.mcts_v5._threat_windows is threat_windows_native


def install() -> int:
    """Rebind the frozen scan (or a fast2 one) to the native scan in every loaded module."""
    if _native is None:
        raise RuntimeError('rt30.native is not built: python -m rt30.native.build')
    bound = [b for target in (fast_windows.FROZEN, *fast_windows.REPLACEMENTS) for b in _bindings(target)]
    for module, name in bound:
        setattr(module, name, threat_windows_native)
    return len(bound)


def uninstall() -> int:
    bound = _bindings(threat_windows_native)
    for module, name in bound:
        setattr(module, name, fast_windows.FROZEN)
    return len(bound)
