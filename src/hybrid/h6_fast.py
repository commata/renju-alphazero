"""Exact solver speed-ups for the H6 label builder (``scripts/build_h6_labels.py --fast``).

The modes reuse the TB-RT30 exact speed-ups (docs/web-ser5.md §3.9, §9.8) and nothing else:

- ``off``: the frozen Python solver;
- ``fast2``: ``rt30.fast`` (threat-window scan with numpy, or the unrolled pure scan without numpy,
  and the two-cell winning-extension rule);
- ``fast3``: fast2 with the native C threat-window scan (X1-A). Strict: if the native module is not
  built (``python -m rt30.native.build``) or is stale, installing fails; there is no silent fallback.
- ``fast4``: fast3 with black forbidden points in C (``rt30.fast4_forbidden``, H7 P1a-1). Strict as fast3.

Each replacement returns the frozen function's result for every input (same windows in the same
order, same extension sets), so the VCF search visits the same nodes: results, node counts and
budget cuts are the same, and so are the H6 labels. Only the speed differs. Nothing here edits a
frozen file; ``install`` rebinds names in this process only, so a process pool installs in every
worker (``ProcessPoolExecutor(initializer=install, initargs=(mode,))``: spawn and forkserver workers
do not inherit the parent's rebinding).
"""
from __future__ import annotations

MODES = ('off', 'fast2', 'fast3', 'fast4')

# Every engine module the builder reaches (proof_label, label_position, the budgeted VCF check,
# --vct1's _own_vct_attack). They must be imported before installing: install rebinds the names
# that are already bound, and later importers copy the (rebound) name from search.mcts_v5/mcts_v321.
_ENGINE_MODULES = ('search.mcts_v321', 'search.mcts_v4', 'search.mcts_v5', 'search.mcts_v7',
                   'analysis.threats', 'analysis.tactical_labels', 'analysis.mcts_v8')

_installed_mode = None


def _import_engine() -> None:
    from importlib import import_module

    for name in _ENGINE_MODULES:
        import_module(name)


def install(mode: str) -> dict:
    """Put this process on ``mode`` (idempotent) and return ``state()``. ``off`` restores the frozen
    functions. Raises when ``fast3`` cannot run its native scan."""
    global _installed_mode
    if mode not in MODES:
        raise ValueError(f'unknown fast mode {mode!r} (one of {", ".join(MODES)})')
    _import_engine()
    from rt30 import fast, fast3, fast3_windows, fast4_forbidden

    if mode in ('fast3', 'fast4'):
        from rt30.native import build

        if not fast3_windows.available() or (mode == 'fast4' and not fast4_forbidden.available()):
            raise RuntimeError(f'--fast {mode} needs the native modules: python -m rt30.native.build (run in src/)')
        reason = build.stale()
        if reason is not None:
            raise RuntimeError(f'--fast {mode}: the native build is stale ({reason}); '
                               'rebuild with python -m rt30.native.build --if-stale (run in src/)')
    fast4_forbidden.uninstall()
    fast3.uninstall_all()
    if mode == 'fast2':
        fast.install_all()
    elif mode in ('fast3', 'fast4'):
        fast3.install_all()
    if mode == 'fast4':
        fast4_forbidden.install()
    _installed_mode = mode
    current = state()
    if current['active'] != expected_active(mode):
        raise RuntimeError(f'--fast {mode}: active {current["active"]}, expected {expected_active(mode)}')
    from rt30 import fast_extensions, fast_windows

    if mode != 'off' and (fast_windows._bindings(fast_windows.FROZEN) or fast_extensions._bindings(fast_extensions.FROZEN)):
        raise RuntimeError(f'--fast {mode}: a loaded module still binds a frozen function')
    from rt30 import fast4_forbidden

    if mode == 'fast4' and fast4_forbidden._bindings(fast4_forbidden.FROZEN):
        raise RuntimeError('--fast fast4: a loaded module still binds the Python forbidden_reason')
    return current


def installed_mode() -> str:
    return _installed_mode or 'off'


def expected_active(mode: str) -> dict:
    from rt30 import fast_extensions, fast_windows

    windows = {'off': fast_windows.FROZEN.__name__, 'fast2': fast_windows.threat_windows.__name__,
               'fast3': 'threat_windows_native', 'fast4': 'threat_windows_native'}[mode]
    extensions = fast_extensions.FROZEN.__name__ if mode == 'off' else 'winning_extensions_in_direction'
    out = {'threat_windows': windows, 'winning_extensions': extensions}
    if mode == 'fast4':  # the key exists only when the native forbidden points are bound
        out['forbidden'] = 'forbidden'
    return out


def state() -> dict:
    """What this process runs: the functions actually bound in the frozen modules (not the request)."""
    from renju import game, rules
    from search import mcts, mcts_v5, mcts_v7, mcts_v321

    windows = {mcts_v5._threat_windows, mcts_v7._threat_windows}
    extensions = {mcts_v321._fast_winning_extensions_in_direction, mcts_v5._fast_winning_extensions_in_direction}
    forbidden = {mcts.forbidden_reason, mcts_v7.forbidden_reason, game.forbidden_reason}
    if len(windows) != 1 or len(extensions) != 1 or len(forbidden) != 1:
        raise RuntimeError('fast speed-ups are only partly installed')
    active = {'threat_windows': windows.pop().__name__, 'winning_extensions': extensions.pop().__name__}
    bound = forbidden.pop()
    if bound is not rules.forbidden_reason:
        active['forbidden'] = bound.__name__
    return {'mode': _installed_mode, 'active': active}


def provenance(mode: str) -> dict:
    """For the report: the mode, the functions it binds (the builder checks every worker ran exactly
    these) and, for fast3, the native build record (Python, SOABI, compiler, flags, source and module
    SHA256)."""
    out = {'requested': mode, 'active': expected_active(mode)}
    if mode in ('fast3', 'fast4'):
        from rt30.native import build

        out['native_build'] = build.build_info()
    return out
