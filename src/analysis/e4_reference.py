"""H7-E4 reference solver: the event detector of the seed 8441 / 8442 arena (docs/h6-value.md, H7-E4).

A selective depth-2 attack wider than the E4 candidate, so that "punished" is not judged by the candidate's
own search:
    first moves     fours, open threes and the local VCF-threat moves (``e4_attack.vcf_threat_moves``)
    depth-1 quiet   our quiet moves inside the proof: fours, open threes and the local VCF-threat moves too
                    (``ReferenceSolver.quiet_moves``; the candidate keeps fours and open threes only)
    defender        every legal reply, as in every search here, so a WIN is a proof
    budget          ``BUDGETS['2m']`` (2M VCF nodes, 200k calls); ``'4m'`` once if the DEV check needs it
Frozen after its DEV check (``scripts/h7_e4_reference.py``): it must prove a WIN on every DEV position where
the selected candidate proved one, before any arena game is looked at.
"""
from __future__ import annotations

from time import perf_counter

from renju import Game
from search.mcts_v5 import _RootContext

from .e4_attack import _local, vcf_threat_moves
from .mcts_v8 import WIN, SearchDiagnostics, _attack_candidates, _first_proven
from .selective_vct import SelectiveSolver, threat_moves
from .threats import VCF_WIN, ordered_moves
from .vct2_attack import _depth2_attack_check

Move = tuple[int, int]
BUDGETS = {'2m': {'node_limit': 20_000, 'call_limit': 200_000, 'node_budget': 2_000_000},
           '4m': {'node_limit': 20_000, 'call_limit': 400_000, 'node_budget': 4_000_000}}


class ReferenceSolver(SelectiveSolver):
    """Our quiet moves: fours and open threes, then the local moves after which we have a VCF (passing)."""

    def quiet_moves(self, game: Game) -> list[Move]:
        base = threat_moves(game)
        me, skip, extra = game.to_play, set(base), []
        for move in ordered_moves(game):
            if move in skip or not _local(game, move):
                continue
            game.play(*move)
            try:
                threat = game.winner is not None if game.done else self.vcf(game, me)[0] == VCF_WIN
            finally:
                game.undo()
            if threat:
                extra.append(move)
        return base + extra


def reference_attack(game: Game, budget_name: str = '2m') -> tuple[Move | None, dict]:
    """The first proven depth-2 attack of ``game.to_play`` found by the reference solver (or None)."""
    budget = BUDGETS[budget_name]
    started = perf_counter()
    context = _RootContext(game.legal_moves(), SearchDiagnostics())
    solver = ReferenceSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    candidates = _attack_candidates(game, context)
    base = len(candidates)
    candidates = candidates + vcf_threat_moves(game, solver, candidates, context)

    def evaluate(g, move, share, call_share):
        return solver._bounded(g, move, share, call_share, lambda after: _depth2_attack_check(solver, after))

    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, evaluate, WIN) if candidates else None
    return chosen, {'budget': budget_name, 'status': WIN if chosen is not None else '',
                    'move': list(chosen) if chosen is not None else None, 'candidates': len(candidates),
                    'vcf_threat_candidates': len(candidates) - base, 'calls': solver.vcf_calls,
                    'nodes': solver.nodes_used, 'exhausted': solver.exhausted,
                    'seconds': round(perf_counter() - started, 4)}
