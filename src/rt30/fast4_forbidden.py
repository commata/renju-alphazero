"""Fourth exact speed-up (H7 P1a-1, docs/h6-value.md): black forbidden points in C (native-core-plan X1-B).

``forbidden_native`` is ``renju.rules.forbidden_reason`` for every input: the same quiet test, the same exact
classifier (exact five, overline, double four as distinct four-stone sets, double three with the recursive
legality of the straight-four extension), the same order of checks and so the same reason, the same
ValueError on an occupied or outside point. P0 found most of V8-C's VCF time in black legality checks
(``_is_legal_for_player`` -> ``forbidden_reason``). The VCF search visits the same nodes; only the speed
differs.

``install`` rebinds the name in every loaded module except ``renju.rules`` itself (whose ``forbidden_reason``
stays the Python oracle, and whose ``legal_black_points`` keeps its own classifier). The module is optional:
``available()`` is False until ``python -m rt30.native.build`` has built it.
"""
from __future__ import annotations

import sys

from renju import rules

try:
    from .native import _forbidden as _native
except ImportError:  # not built on this machine
    _native = None

FROZEN = rules.forbidden_reason
forbidden_native = _native.forbidden if _native is not None else None


def available() -> bool:
    return _native is not None


def _bindings(target):
    return [(module, name) for module in list(sys.modules.values())
            if module is not None and getattr(module, '__name__', None) not in (__name__, 'renju.rules')
            for name, value in list(getattr(module, '__dict__', {}).items()) if value is target]


def installed() -> bool:
    from search import mcts
    return forbidden_native is not None and mcts.forbidden_reason is forbidden_native


def install() -> int:
    """Rebind ``forbidden_reason`` to the native one in every loaded module (import the engine first)."""
    if _native is None:
        raise RuntimeError('rt30.native is not built: python -m rt30.native.build')
    bound = _bindings(FROZEN)
    for module, name in bound:
        setattr(module, name, forbidden_native)
    return len(bound)


def uninstall() -> int:
    if forbidden_native is None:
        return 0
    bound = _bindings(forbidden_native)
    for module, name in bound:
        setattr(module, name, FROZEN)
    return len(bound)
