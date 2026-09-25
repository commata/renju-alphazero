"""V5 forced policy followed by proactive root threat planning."""
from dataclasses import dataclass
from math import sqrt
from random import Random
from time import perf_counter

from renju import BLACK, WHITE, Game
from .mcts import Move
from .mcts_v5 import (
    SearchDiagnostics as V5Diagnostics, _RootContext, _forced_v5_move,
    _root_candidates_v5, _search_v5_tree, _validate_v5_config,
)
from .threat_patterns import compound_moves


V5_FINAL = dict(simulations=50, tactical_simulations=100, tactical_score_threshold=1800,
                exploration=sqrt(2), candidate_limit=20, initial_width=8,
                neighborhood_radius=2, priority_top_k=8)


@dataclass
class SearchDiagnostics(V5Diagnostics):
    black_43_candidates: int = 0
    white_43_candidates: int = 0
    white_44_candidates: int = 0
    white_33_candidates: int = 0
    black_43_defense_candidates: int = 0
    black_43_defense_injections: int = 0
    future_black_43_setups: int = 0
    future_white_43_setups: int = 0
    future_white_44_setups: int = 0
    future_white_33_setups: int = 0
    forbidden_defense_induction_count: int = 0
    v6_root_injection_count: int = 0
    v6_selected_threat_type: str | None = None
    v6_selected_reasons: tuple[str, ...] = ()
    v6_threat_planner_seconds: float = 0.0


# Ordinal root ordering, not arbitrary additions to the V3.2.1 score.
PRIORITY = {'white_44': 4, 'black_43': 3, 'white_43': 3, 'white_33': 2}


def plan_root(game: Game, context: _RootContext) -> dict[Move, set[str]]:
    diag = context.diagnostics
    own = compound_moves(game, game.to_play)
    reasons: dict[Move, set[str]] = {}
    color = 'black' if game.to_play == BLACK else 'white'
    for move, compound in own.items():
        for kind in sorted(compound.kinds):
            reason = f'{color}_{kind}'
            setattr(diag, reason + '_candidates', getattr(diag, reason + '_candidates') + 1)
            reasons.setdefault(move, set()).add(reason)
    return reasons


def _root_candidates_v6(game, context, limit, radius):
    # Select the complete original pool before injection; no generic widening.
    baseline, score = _root_candidates_v5(game, context, limit, radius)
    started = perf_counter()
    reasons = plan_root(game, context)
    legal = set(context.legal)
    reasons = {m: kinds for m, kinds in reasons.items() if m in legal}
    context.diagnostics.v6_threat_planner_seconds = perf_counter() - started
    context.diagnostics.v6_root_injection_count = len(reasons)
    moves = set(baseline).union(reasons)
    ranked = sorted(moves, key=lambda m: (
        -max((PRIORITY[k] for k in reasons.get(m, ())), default=0), context.key(game, m),
    ))
    # Budget selection uses original V5 scores, not planner ordering tiers.
    score = max(score, max(-context.key(game, m)[0] for m in ranked))
    return ranked, score, reasons


def mcts_search_v6(
    game: Game, *, simulations=50, tactical_simulations=100,
    tactical_score_threshold=1800, exploration=sqrt(2), candidate_limit=20,
    initial_width=8, neighborhood_radius=2, priority_top_k=8,
    random: Random | None = None, diagnostics: SearchDiagnostics | None = None,
) -> Move:
    _validate_v5_config(
        simulations=simulations, tactical_simulations=tactical_simulations,
        tactical_score_threshold=tactical_score_threshold, exploration=exploration,
        candidate_limit=candidate_limit, initial_width=initial_width,
        neighborhood_radius=neighborhood_radius, priority_top_k=priority_top_k,
    )
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.__dict__.update(vars(SearchDiagnostics()))
    context = _RootContext(game.legal_moves(), diag)
    forced = _forced_v5_move(game, context=context)
    if forced is not None:
        return forced
    moves, score, reasons = _root_candidates_v6(game, context, candidate_limit, neighborhood_radius)
    tactical = tactical_score_threshold is not None and score >= tactical_score_threshold
    budget = tactical_simulations if tactical else simulations
    diag.best_root_tactical_score = score
    diag.selected_simulations = budget
    diag.simulation_mode = 'tactical' if tactical else 'normal'
    diag.root_candidates = tuple(moves)
    chosen = _search_v5_tree(game, moves, budget, exploration, candidate_limit,
                             initial_width, neighborhood_radius, priority_top_k, random)
    selected = sorted(reasons.get(chosen, ()), key=lambda k: (-PRIORITY[k], k))
    diag.v6_selected_reasons = tuple(selected)
    diag.v6_selected_threat_type = selected[0] if selected else None
    return chosen
