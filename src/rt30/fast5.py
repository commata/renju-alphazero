"""Fifth exact speed-up (X2-F, P1a-2; docs/h6-value.md, "TB-V3 실행 최적화"): the whole VCF search in C.

fast5 = fast4 + the native VCF core (``rt30.vcfcore``, ``vcf_core.c``). Two drop-ins with the signatures and
return values of the frozen searches:
    ``find_vcf_with_stats``  ``search.mcts_v7._find_vcf_with_stats`` (V7, V8-A / V8-B, VCT2, the E4-v2 gate and
                             attack, the threat solvers)
    ``make_find``            ``analysis.vcf_tt_v1.make_find`` (V8-C's call-local no-win table): the same result,
                             nodes, cut, and the same table hits and stores in its ``TableStats``
The C recursion is the Python one step by step (same candidates, same order, one node per candidate before
its legality check, the frozen native forbidden classifier), so every caller sees the same results, witness,
charged nodes and cuts, hence makes the same calls and decisions. Only the speed differs.

``install`` puts this process on fast4 (``hybrid.h6_fast``) and then rebinds the two searches in every loaded
module; ``uninstall`` restores the frozen ones (leaving fast4). Strict: without a fresh native build it fails.
Nothing here edits a frozen file.
"""
from __future__ import annotations

import sys

from analysis import vcf_tt_v1
from search import mcts_v7 as v7

try:
    from .vcfcore import _vcf_core as _native
except ImportError:  # not built on this machine
    _native = None

FROZEN = v7._find_vcf_with_stats
FROZEN_MAKE_FIND = vcf_tt_v1.make_find
FROZEN_TABLE_FIND = vcf_tt_v1.find_vcf_with_stats
MODE = 'fast5'
BASE_MODE = 'fast4'


def available() -> bool:
    return _native is not None


def _validate(attacker, max_fours, node_limit) -> None:
    # the frozen search's own checks, in its order and with its messages
    if attacker not in (v7.BLACK, v7.WHITE):
        raise ValueError('attacker must be BLACK or WHITE')
    if type(max_fours) is not int or max_fours < 1:
        raise ValueError('max_fours must be a positive integer')
    if type(node_limit) is not int or node_limit < 1:
        raise ValueError('node_limit must be a positive integer')


def _result(found, moves, comps, fours, nodes):
    return v7.VCFResult(first_move=moves[0], fours=fours, nodes=nodes, attack_moves=moves, completion_points=comps)


def find_vcf_with_stats(game, attacker, *, max_fours, node_limit):
    """``search.mcts_v7._find_vcf_with_stats``, exact."""
    _validate(attacker, max_fours, node_limit)
    found, moves, comps, fours, nodes, exhausted, _, _ = _native.vcf(game.board, attacker, max_fours, node_limit, 0)
    if not found:
        return None, nodes, bool(exhausted)
    return _result(found, moves, comps, fours, nodes), nodes, bool(exhausted)


def make_find(stats=None):
    """``analysis.vcf_tt_v1.make_find``, exact: a fresh table per call; totals go to ``stats``."""

    def find(game, attacker, *, max_fours, node_limit):
        _validate(attacker, max_fours, node_limit)
        found, moves, comps, fours, nodes, exhausted, hits, stores = _native.vcf(game.board, attacker, max_fours,
                                                                                node_limit, 1)
        if stats is not None:
            stats.calls += 1
            stats.hits += hits
            stats.stores += stores
        if not found:
            return None, nodes, bool(exhausted)
        return _result(found, moves, comps, fours, nodes), nodes, bool(exhausted)
    return find


table_find_vcf_with_stats = make_find()


def _bindings(target):
    return [(module, name) for module in list(sys.modules.values())
            if module is not None and getattr(module, '__name__', None) != __name__
            for name, value in list(getattr(module, '__dict__', {}).items()) if value is target]


# The engine modules that must hold the native searches once installed (fail closed if any still holds a frozen one).
ENGINE_MODULES = ('search.mcts_v7', 'analysis.vcf_tt_v1', 'analysis.threats', 'analysis.mcts_v8',
                  'analysis.selective_vct', 'analysis.vct2_attack', 'analysis.e4_attack', 'analysis.e4v2_attack')


def frozen_aliases() -> list[str]:
    """Names in loaded modules (other than this one) that still point at a frozen search."""
    return sorted(f'{m.__name__}.{n}' for target in (FROZEN, FROZEN_MAKE_FIND, FROZEN_TABLE_FIND)
                  for m, n in _bindings(target))


def installed() -> bool:
    """The native searches bound where the engine looks them up, and no frozen alias left in a loaded module."""
    return (_native is not None and v7._find_vcf_with_stats is find_vcf_with_stats
            and vcf_tt_v1.make_find is make_find and vcf_tt_v1.find_vcf_with_stats is table_find_vcf_with_stats
            and not frozen_aliases())


def install() -> dict:
    """Put this process on fast4, then on the native VCF core; returns what was rebound."""
    from hybrid import h6_fast

    from .vcfcore import build

    if _native is None:
        raise RuntimeError('fast5 needs the native VCF core: python -m rt30.vcfcore.build (run in src/)')
    reason = build.stale()
    if reason is not None:
        raise RuntimeError(f'fast5: the native VCF core is stale ({reason}); '
                           'python -m rt30.vcfcore.build --if-stale (run in src/)')
    h6_fast.install(BASE_MODE)
    from importlib import import_module
    for name in ENGINE_MODULES:  # loaded first, so that every alias exists before the rebinding
        import_module(name)
    bound = {'find': _bindings(FROZEN), 'make_find': _bindings(FROZEN_MAKE_FIND),
             'table_find': _bindings(FROZEN_TABLE_FIND)}
    for module, name in bound['find']:
        setattr(module, name, find_vcf_with_stats)
    for module, name in bound['make_find']:
        setattr(module, name, make_find)
    for module, name in bound['table_find']:
        setattr(module, name, table_find_vcf_with_stats)
    left = frozen_aliases()
    if left:
        raise RuntimeError(f'fast5: frozen searches still bound after install: {left}')
    return {k: sorted(f'{m.__name__}.{n}' for m, n in v) for k, v in bound.items()}


def uninstall() -> int:
    """Restore the frozen searches (the process stays on whatever fast mode it was)."""
    pairs = ((find_vcf_with_stats, FROZEN), (make_find, FROZEN_MAKE_FIND),
             (table_find_vcf_with_stats, FROZEN_TABLE_FIND))
    count = 0
    for native, frozen in pairs:
        for module, name in _bindings(native):
            setattr(module, name, frozen)
            count += 1
    return count
