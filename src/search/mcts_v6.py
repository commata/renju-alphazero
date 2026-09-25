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
from .threat_planning import future_setups, PlanningStats, forbidden_defense_attacks


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
    future_black_43_defense_candidates: int = 0
    planner_structural_candidates: int = 0
    planner_examined_setups: int = 0
    planner_setup_cap_hits: int = 0
    planner_continuation_cap_hits: int = 0
    planner_defense_cap_skips: int = 0
    planner_legal_responses: int = 0
    legal_defense_count: int = 0
    forbidden_defense_count: int = 0
    remaining_winning_continuations: int = 0


# Ordinal root ordering, not arbitrary additions to the V3.2.1 score.
PRIORITY = {'white_44': 4, 'black_43': 3, 'white_43': 3, 'white_33': 2,
            'black_43_defense': 3, 'future_black_43': 1,
            'future_white_43': 1, 'future_white_44': 1, 'future_white_33': 1,
            'future_black_43_defense': 1, 'forbidden_defense_induction': 3}


def plan_root(game: Game, context: _RootContext, *, response_counts=None) -> dict[Move, set[str]]:
    response_counts = response_counts if response_counts is not None else {}
    diag = context.diagnostics
    own = compound_moves(game, game.to_play)
    reasons: dict[Move, set[str]] = {}
    color = 'black' if game.to_play == BLACK else 'white'
    for move, compound in own.items():
        for kind in sorted(compound.kinds):
            reason = f'{color}_{kind}'
            setattr(diag, reason + '_candidates', getattr(diag, reason + '_candidates') + 1)
            reasons.setdefault(move, set()).add(reason)
    if game.to_play == WHITE:
        induction = forbidden_defense_attacks(game)
        diag.forbidden_defense_induction_count = len(induction)
        for move, profile in induction.items():
            reasons.setdefault(move, set()).add('forbidden_defense_induction')
            response_counts[move] = profile.legal_defense_count
            diag.legal_defense_count += profile.legal_defense_count
            diag.forbidden_defense_count += profile.forbidden_defense_count
            diag.remaining_winning_continuations += profile.remaining_winning_continuations
        danger = compound_moves(game, BLACK)
        diag.black_43_candidates = len(danger)
        defenses = set().union(*(t.defense_points for t in danger.values()))
        defenses.intersection_update(context.legal)
        diag.black_43_defense_candidates = len(defenses)
        diag.black_43_defense_injections = len(defenses)
        for move in sorted(defenses):
            reasons.setdefault(move, set()).add('black_43_defense')
    stats = PlanningStats()
    # Geometry is the gate: without a two-stone window future_setups does no work.
    # No new score threshold is substituted for the fixed V5 threshold.
    for player in ((BLACK, WHITE) if game.to_play == WHITE else (BLACK,)):
        setups = future_setups(game, player, stats=stats)
        setup_color = 'black' if player == BLACK else 'white'
        for move, setup in setups.items():
            diag.planner_legal_responses += setup.legal_defense_count
            for kind in sorted(setup.kinds):
                reason = f'future_{setup_color}_{kind}'
                setattr(diag, reason + '_setups', getattr(diag, reason + '_setups') + 1)
                if player == game.to_play:
                    reasons.setdefault(move, set()).add(reason)
                    response_counts[move] = setup.legal_defense_count
            if player != game.to_play:
                for defense in sorted(setup.defenses.intersection(context.legal)):
                    reasons.setdefault(defense, set()).add('future_black_43_defense')
    diag.future_black_43_defense_candidates = sum('future_black_43_defense' in r for r in reasons.values())
    for key, value in vars(stats).items():
        setattr(diag, 'planner_' + key, value)
    return reasons


def _root_candidates_v6(game, context, limit, radius):
    # Select the complete original pool before injection; no generic widening.
    baseline, score = _root_candidates_v5(game, context, limit, radius)
    started = perf_counter()
    response_counts = {}
    reasons = plan_root(game, context, response_counts=response_counts)
    legal = set(context.legal)
    reasons = {m: kinds for m, kinds in reasons.items() if m in legal}
    context.diagnostics.v6_threat_planner_seconds = perf_counter() - started
    context.diagnostics.v6_root_injection_count = len(reasons)
    if not reasons:
        return baseline, score, reasons
    moves = set(baseline).union(reasons)
    ranked = sorted(moves, key=lambda m: (
        -max([PRIORITY[k] for k in reasons.get(m, ())]
             + [3 if m in context.injected else 0]),
        response_counts.get(m, 0), context.key(game, m),
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
