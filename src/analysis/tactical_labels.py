"""Proof-labelled tactical positions for the teacher arm (offline, torch-free).

A position gets a label only when its target follows from the Renju rules or from
a found VCF certificate, never from an agent's choice:

========================  ==============================  ==================
kind                      policy target                   value (side to move)
========================  ==============================  ==================
``immediate_win``         every legal winning point       +1
``forced_loss``           none                            -1  (>= 2 opponent
                                                               winning points, or
                                                               the only block is a
                                                               black forbidden point)
``must_block``            the single legal block          none
``unstoppable_four``      every such move                 +1
``vcf``                   every move that starts a VCF    +1
``vcf_loss`` (opt-in)     none                            -1  (every move proven
                                                               UNSAFE to VCF)
========================  ==============================  ==================

``must_block`` has no value label: blocking is forced but the result is unknown.
``vcf`` first moves are the fours after which (opponent's forced block) the
attacker still has a VCF, with the same solver class as ``analysis.threats`` (the
frozen V7 solver, budget-only cut); a position whose VCF probe is cut is skipped.
"""
from __future__ import annotations

from analysis.threats import UNSAFE, VCF_WIN, ThreatSolver, decision_status
from renju import Game
from search.mcts_v5 import _unstoppable_four_moves, _window_candidates, _winning_moves

KINDS = ('immediate_win', 'forced_loss', 'must_block', 'unstoppable_four', 'vcf', 'vcf_loss')


def vcf_first_moves(game: Game, solver: ThreatSolver) -> list:
    """Legal fours of ``game.to_play`` that begin a VCF (board restored)."""
    me = game.to_play
    legal = set(game.legal_moves())
    starts = []
    for move in _window_candidates(game, me, 3):
        if move not in legal:
            continue
        game.play(*move)
        try:
            if game.done:
                starts.append(move)  # an immediate five (callers handle these first)
                continue
            if _winning_moves(game, -me):
                continue  # the opponent answers our four with a five
            completions = _winning_moves(game, me)
            if not completions:
                continue  # not a four
            if len(completions) >= 2:
                starts.append(move)
                continue
            block = completions[0]
            if block not in game.legal_moves():
                starts.append(move)  # the only block is a black forbidden point
                continue
            game.play(*block)
            try:
                if _winning_moves(game, -me):
                    continue  # the block makes a four: outside the solver's class
                if solver.vcf(game, me)[0] == VCF_WIN:
                    starts.append(move)
            finally:
                game.undo()
        finally:
            game.undo()
    return sorted(starts)


def label_position(game: Game, solver: ThreatSolver, *, prove_losses: bool = False
                   ) -> dict | None:
    """Return ``{'kind', 'policy': [moves], 'value': +1/-1/None}`` or ``None``."""
    if game.done:
        return None
    me = game.to_play
    legal = game.legal_moves()
    legal_set = set(legal)
    wins = [m for m in _winning_moves(game, me) if m in legal_set]
    if wins:
        return {'kind': 'immediate_win', 'policy': sorted(wins), 'value': 1}
    threats = _winning_moves(game, -me)
    if len(threats) >= 2:
        return {'kind': 'forced_loss', 'policy': [], 'value': -1}
    if len(threats) == 1:
        if threats[0] not in legal_set:
            return {'kind': 'forced_loss', 'policy': [], 'value': -1}
        return {'kind': 'must_block', 'policy': [threats[0]], 'value': None}
    fours = _unstoppable_four_moves(game, me, legal=legal_set)
    if fours:
        return {'kind': 'unstoppable_four', 'policy': sorted(fours), 'value': 1}
    status, _ = solver.vcf(game, me)
    if status == VCF_WIN:
        starts = vcf_first_moves(game, solver)
        if starts:
            return {'kind': 'vcf', 'policy': starts, 'value': 1}
    if prove_losses:
        counts, _ = solver.decision(game, 0)
        if decision_status(counts) == UNSAFE:
            return {'kind': 'vcf_loss', 'policy': [], 'value': -1}
    return None
