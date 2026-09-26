"""MCTS-v7 frozen-benchmark candidate built on the unchanged V6 search stack."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random
from time import perf_counter

from renju import BLACK, WHITE, EMPTY, Game
from renju.rules import forbidden_reason
from .mcts import Move, _is_legal_for_player, _wins_for_player
from .mcts_v5 import (
    WINDOWS, _RootContext, _double_threat_moves, _forced_v5_move,
    _four_completions, _placed_completions, _search_v5_tree,
    _threat_windows, _unstoppable_four_moves, _validate_v5_config,
    _window_candidates, _winning_moves,
)
from .mcts_v6 import (
    PRIORITY, SearchDiagnostics as V6Diagnostics, V5_FINAL, _root_candidates_v6,
)
from .threat_patterns import compound_moves, placed


V7_FINAL = {
    **V5_FINAL,
    "own_vcf_max_fours": 10,
    "own_vcf_node_limit": 2000,
    "safety_vcf_max_fours": 10,
    "safety_vcf_node_limit": 1000,
    "self_forbidden_min_white": 3,
}


@dataclass(frozen=True)
class VCFResult:
    first_move: Move
    fours: int
    nodes: int
    attack_moves: tuple[Move, ...]
    completion_points: tuple[Move, ...]


@dataclass
class SearchDiagnostics(V6Diagnostics):
    v7_own_vcf_found: bool = False
    v7_own_vcf_length: int = 0
    v7_own_vcf_nodes: int = 0
    v7_safety_checked: int = 0
    v7_safety_removed: int = 0
    v7_safety_augmented: bool = False
    v7_safety_fallback: bool = False
    v7_self_forbidden_penalized: int = 0
    v7_stage4_tiebreak_applied: bool = False
    v7_module_seconds: float = 0.0


@dataclass
class _VCFState:
    node_limit: int
    nodes: int = 0
    exhausted: bool = False

    def take_node(self) -> bool:
        if self.nodes >= self.node_limit:
            self.exhausted = True
            return False
        self.nodes += 1
        return True


def _validate_v7_config(*, own_vcf_max_fours, own_vcf_node_limit,
                        safety_vcf_max_fours, safety_vcf_node_limit,
                        self_forbidden_min_white):
    for name, value in (
        ("own_vcf_max_fours", own_vcf_max_fours),
        ("own_vcf_node_limit", own_vcf_node_limit),
        ("safety_vcf_max_fours", safety_vcf_max_fours),
        ("safety_vcf_node_limit", safety_vcf_node_limit),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(self_forbidden_min_white) is not int or not 2 <= self_forbidden_min_white <= 4:
        raise ValueError("self_forbidden_min_white must be an integer in [2, 4]")


def _legal_completions(game: Game, player: int, anchor: Move) -> tuple[Move, ...]:
    return tuple(sorted(
        move for move in _placed_completions(game, player, anchor)
        if _is_legal_for_player(game, player, move)
    ))


def _reverse_four_after_block(game: Game, defender: int, block: Move) -> bool:
    return bool(_legal_completions(game, defender, block))


def _find_vcf_recursive(
    game: Game,
    attacker: int,
    *,
    max_fours: int,
    fours_used: int,
    state: _VCFState,
) -> tuple[tuple[Move, ...], tuple[Move, ...], int] | None:
    if state.exhausted:
        return None

    immediate = sorted(_winning_moves(game, attacker))
    if immediate:
        move = immediate[0]
        return (move,), (), fours_used

    if fours_used >= max_fours:
        return None

    defender = -attacker
    for move in _window_candidates(game, attacker, 3):
        if not state.take_node():
            return None
        if not _is_legal_for_player(game, attacker, move):
            continue
        if _wins_for_player(game, attacker, move):
            return (move,), (), fours_used

        with placed(game, attacker, move):
            completions = _legal_completions(game, attacker, move)
            if not completions:
                continue

            if _winning_moves(game, defender):
                continue

            if len(completions) >= 2:
                return (move,), completions, fours_used + 1

            block = completions[0]
            if not _is_legal_for_player(game, defender, block):
                return (move,), completions, fours_used + 1
            if _wins_for_player(game, defender, block):
                continue

            with placed(game, defender, block):
                if _reverse_four_after_block(game, defender, block):
                    continue
                child = _find_vcf_recursive(
                    game, attacker, max_fours=max_fours,
                    fours_used=fours_used + 1, state=state,
                )
                if child is None:
                    continue
                attack_moves, child_completions, child_fours = child
                return ((move,) + attack_moves,
                        tuple(sorted(set(completions).union(child_completions))),
                        child_fours)
    return None


def find_vcf(
    game: Game,
    attacker: int,
    *,
    max_fours: int,
    node_limit: int,
) -> VCFResult | None:
    """Return one deterministic conservative VCF line without changing game."""
    if attacker not in (BLACK, WHITE):
        raise ValueError("attacker must be BLACK or WHITE")
    if type(max_fours) is not int or max_fours < 1:
        raise ValueError("max_fours must be a positive integer")
    if type(node_limit) is not int or node_limit < 1:
        raise ValueError("node_limit must be a positive integer")

    state = _VCFState(node_limit=node_limit)
    found = _find_vcf_recursive(
        game, attacker, max_fours=max_fours, fours_used=0, state=state,
    )
    if found is None:
        return None
    attack_moves, completion_points, fours = found
    return VCFResult(
        first_move=attack_moves[0],
        fours=fours,
        nodes=state.nodes,
        attack_moves=attack_moves,
        completion_points=completion_points,
    )


def _has_four_material(game: Game, player: int) -> bool:
    return next(_threat_windows(game, player, 3), None) is not None


def _candidate_allows_vcf(
    game: Game, move: Move, player: int, opponent: int,
    *, max_fours: int, node_limit: int,
) -> tuple[bool, VCFResult | None]:
    if _wins_for_player(game, player, move):
        return False, None
    with placed(game, player, move):
        result = find_vcf(
            game, opponent, max_fours=max_fours, node_limit=node_limit,
        )
    return result is not None, result


def _own_four_creators(game: Game, player: int, legal: set[Move]) -> set[Move]:
    return {
        move for move in _window_candidates(game, player, 3)
        if move in legal and _four_completions(game, player, move)
    }


def _apply_vcf_safety(
    game: Game,
    context: _RootContext,
    moves: list[Move],
    diag: SearchDiagnostics,
    *,
    max_fours: int,
    node_limit: int,
) -> list[Move]:
    opponent = -game.to_play
    if not _has_four_material(game, opponent):
        return moves

    player = game.to_play
    safe: list[Move] = []
    removed: list[Move] = []
    for move in moves:
        diag.v7_safety_checked += 1
        unsafe, _ = _candidate_allows_vcf(
            game, move, player, opponent,
            max_fours=max_fours, node_limit=node_limit,
        )
        if unsafe:
            removed.append(move)
        else:
            safe.append(move)
    diag.v7_safety_removed += len(removed)
    if safe:
        return safe

    legal = set(context.legal)
    augmentation: set[Move] = set()
    opponent_line = find_vcf(
        game, opponent, max_fours=max_fours, node_limit=node_limit,
    )
    if opponent_line is not None:
        augmentation.update(opponent_line.attack_moves)
        augmentation.update(opponent_line.completion_points)
    augmentation.update(_own_four_creators(game, player, legal))
    augmentation.intersection_update(legal)
    augmented = [m for m in sorted(augmentation) if m not in moves]
    diag.v7_safety_augmented = bool(augmented)

    rescued: list[Move] = []
    for move in augmented:
        diag.v7_safety_checked += 1
        unsafe, _ = _candidate_allows_vcf(
            game, move, player, opponent,
            max_fours=max_fours, node_limit=node_limit,
        )
        if unsafe:
            diag.v7_safety_removed += 1
        else:
            rescued.append(move)
    if rescued:
        return rescued

    diag.v7_safety_fallback = True
    return moves


def _white_pressure_points(game: Game, minimum_white: int) -> set[Move]:
    result: set[Move] = set()
    for window in WINDOWS:
        values = [game.board[r][c] for r, c in window]
        if BLACK in values or values.count(WHITE) < minimum_white:
            continue
        result.update((r, c) for r, c in window if game.board[r][c] == EMPTY)
    return result


def _risky_black_forbidden_points(game: Game, minimum_white: int) -> set[Move]:
    points = set()
    for row, col in _white_pressure_points(game, minimum_white):
        if forbidden_reason(game.board, row, col) is not None:
            points.add((row, col))
    return points


def _apply_self_forbidden_penalty(
    game: Game,
    moves: list[Move],
    diag: SearchDiagnostics,
    *,
    minimum_white: int,
) -> list[Move]:
    if game.to_play != BLACK or len(moves) < 2:
        return moves
    before = _risky_black_forbidden_points(game, minimum_white)
    normal: list[Move] = []
    penalized: list[Move] = []
    for move in moves:
        with placed(game, BLACK, move):
            after = _risky_black_forbidden_points(game, minimum_white)
        if after - before:
            penalized.append(move)
        else:
            normal.append(move)
    diag.v7_self_forbidden_penalized = len(penalized)
    return normal + penalized


def _stage4_v7_move(
    game: Game,
    context: _RootContext,
    original: Move,
    diag: SearchDiagnostics,
    *,
    max_fours: int,
    node_limit: int,
) -> Move:
    """Refine only Stage-4 ties without changing V5/V6 forced-policy stages.

    V5 Stage 4 first minimizes the number of remaining unstoppable fours.
    Among candidates with that same best value, V7 prefers a defense that also
    removes opponent compound-threat creators (43/44/33), then one that leaves
    no opponent VCF. The final key is the unchanged V6/V5 root ordering.
    """
    player = game.to_play
    opponent = -player
    creators = _unstoppable_four_moves(game, opponent)
    if not creators:
        return original

    creator_set = set(creators)
    defenses = creator_set.copy()
    for window in _threat_windows(game, opponent):
        if creator_set.intersection(window):
            defenses.update(pos for pos in window if game.board[pos[0]][pos[1]] == EMPTY)
    defenses.intersection_update(context.legal)
    if len(defenses) < 2:
        return original

    rows = []
    for move in sorted(defenses):
        with placed(game, player, move):
            remaining = len(_unstoppable_four_moves(game, opponent))
            compound_count = len(compound_moves(game, opponent))
            has_vcf = find_vcf(
                game, opponent, max_fours=max_fours, node_limit=node_limit,
            ) is not None
        rows.append((remaining, compound_count, int(has_vcf),
                     context.key(game, move), move))

    best_remaining = min(row[0] for row in rows)
    tied = [row for row in rows if row[0] == best_remaining]
    chosen = min(tied, key=lambda row: row[1:])[-1]
    diag.v7_stage4_tiebreak_applied = chosen != original
    return chosen

def mcts_search_v7(
    game: Game, *, simulations=50, tactical_simulations=100,
    tactical_score_threshold=1800, exploration=sqrt(2), candidate_limit=20,
    initial_width=8, neighborhood_radius=2, priority_top_k=8,
    own_vcf_max_fours=10, own_vcf_node_limit=2000,
    safety_vcf_max_fours=10, safety_vcf_node_limit=1000,
    self_forbidden_min_white=3,
    random: Random | None = None, diagnostics: SearchDiagnostics | None = None,
) -> Move:
    """V6 search with fixed V7 root-only VCF and forbidden-point modules."""
    _validate_v5_config(
        simulations=simulations, tactical_simulations=tactical_simulations,
        tactical_score_threshold=tactical_score_threshold, exploration=exploration,
        candidate_limit=candidate_limit, initial_width=initial_width,
        neighborhood_radius=neighborhood_radius, priority_top_k=priority_top_k,
    )
    _validate_v7_config(
        own_vcf_max_fours=own_vcf_max_fours,
        own_vcf_node_limit=own_vcf_node_limit,
        safety_vcf_max_fours=safety_vcf_max_fours,
        safety_vcf_node_limit=safety_vcf_node_limit,
        self_forbidden_min_white=self_forbidden_min_white,
    )
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.__dict__.update(vars(SearchDiagnostics()))
    context = _RootContext(game.legal_moves(), diag)

    forced = _forced_v5_move(game, context=context)
    forced_stage = diag.forced_policy_stage
    if forced is not None and forced_stage in (1, 2, 3):
        return forced

    module_started = perf_counter()
    own_vcf = find_vcf(
        game, game.to_play,
        max_fours=own_vcf_max_fours, node_limit=own_vcf_node_limit,
    )
    if own_vcf is not None:
        diag.v7_own_vcf_found = True
        diag.v7_own_vcf_length = own_vcf.fours
        diag.v7_own_vcf_nodes = own_vcf.nodes
        diag.forced_policy_stage = None
        diag.v7_module_seconds = perf_counter() - module_started
        return own_vcf.first_move

    if forced is not None:
        if forced_stage == 4:
            forced = _stage4_v7_move(
                game, context, forced, diag,
                max_fours=safety_vcf_max_fours,
                node_limit=safety_vcf_node_limit,
            )
        diag.v7_module_seconds = perf_counter() - module_started
        return forced

    moves, score, reasons = _root_candidates_v6(
        game, context, candidate_limit, neighborhood_radius,
    )
    moves = _apply_vcf_safety(
        game, context, moves, diag,
        max_fours=safety_vcf_max_fours,
        node_limit=safety_vcf_node_limit,
    )
    moves = _apply_self_forbidden_penalty(
        game, moves, diag, minimum_white=self_forbidden_min_white,
    )
    diag.v7_module_seconds = perf_counter() - module_started

    tactical = tactical_score_threshold is not None and score >= tactical_score_threshold
    budget = tactical_simulations if tactical else simulations
    diag.best_root_tactical_score = score
    diag.selected_simulations = budget
    diag.simulation_mode = "tactical" if tactical else "normal"
    diag.root_candidates = tuple(moves)
    chosen = _search_v5_tree(
        game, moves, budget, exploration, candidate_limit,
        initial_width, neighborhood_radius, priority_top_k, random,
    )
    selected = sorted(reasons.get(chosen, ()), key=lambda k: (-PRIORITY[k], k))
    diag.v6_selected_reasons = tuple(selected)
    diag.v6_selected_threat_type = selected[0] if selected else None
    return chosen
