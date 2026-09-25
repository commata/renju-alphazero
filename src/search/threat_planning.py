"""Bounded threat-space probes, never a whole-board forced-win proof."""
from dataclasses import dataclass

from renju import BLACK, WHITE, Game
from .mcts import Move, _is_legal_for_player, _wins_for_player
from .mcts_v5 import _window_candidates, _winning_moves, _four_completions
from .threat_patterns import (
    Compound, compound_at, placed, structural_candidates, fours_at, threes_at,
    BY_CELL,
)


@dataclass(frozen=True)
class PlannerLimits:
    # Work caps, not tuned tactical-score thresholds. Expose saturation in diagnostics.
    setups: int = 6
    continuations: int = 8
    defenses: int = 24


@dataclass(frozen=True)
class Setup:
    move: Move
    kinds: frozenset[str]
    defenses: frozenset[Move]
    legal_defense_count: int
    surviving_response_count: int
    continuation_count: int


@dataclass
class PlanningStats:
    structural_candidates: int = 0
    examined_setups: int = 0
    setup_cap_hits: int = 0
    continuation_cap_hits: int = 0
    defense_cap_skips: int = 0


@dataclass(frozen=True)
class DefenseProfile:
    legal_defense_count: int
    forbidden_defense_count: int
    remaining_winning_continuations: int


def white_defense_profile(game: Game, move: Move) -> DefenseProfile:
    """Count exact single-move defenses to the four created by a white move.

    These are winning completion points, not speculative three endpoints.
    Forbidden defenses are classified on the board AFTER the white attack.
    """
    if not _is_legal_for_player(game, WHITE, move):
        return DefenseProfile(0, 0, 0)
    with placed(game, WHITE, move):
        completions = set().union(*(t.continuations for t in fours_at(game, WHITE, move)))
        legal, forbidden, remaining = 0, 0, 0
        for block in sorted(completions):
            if not _is_legal_for_player(game, BLACK, block):
                forbidden += 1
                continue
            with placed(game, BLACK, block):
                wins = _winning_moves(game, WHITE)
                remaining += len(wins)
                legal += not bool(wins)
        # Counter-wins are real defenses even when all blocking points are forbidden.
        legal += len(set(_winning_moves(game, BLACK)) - completions)
        return DefenseProfile(legal, forbidden, remaining if legal else len(completions))


def forbidden_defense_attacks(game: Game) -> dict[Move, DefenseProfile]:
    result = {}
    for move in _window_candidates(game, WHITE, 3):
        if _wins_for_player(game, WHITE, move):
            continue
        profile = white_defense_profile(game, move)
        if profile.forbidden_defense_count:
            result[move] = profile
    return result


def _structure_key(game: Game, player: int, move: Move):
    counts = [sum(game.board[r][c] == player for r, c in w)
              for w in BY_CELL[move] if all(game.board[r][c] != -player for r, c in w)]
    return (-max(counts, default=0), -sum(counts), move)


def _linked(compound: Compound, setup: Move) -> bool:
    return any(setup in t.stones for t in compound.fours + compound.threes)


def _continuations(game, player, setup, limits, stats):
    candidates = sorted(structural_candidates(game, player),
                        key=lambda m: _structure_key(game, player, m))
    stats.continuation_cap_hits += len(candidates) > limits.continuations
    result = {}
    for move in candidates[:limits.continuations]:
        compound = compound_at(game, player, move)
        if compound is not None and _linked(compound, setup):
            with placed(game, player, move):
                if _winning_moves(game, -player):
                    continue
            result[move] = compound
    return result


def _responses(game, player, setup, continuations):
    wins = _winning_moves(game, player)
    if wins:
        # The defender must occupy an immediate completion or win now.
        return set(wins).union(_winning_moves(game, -player))
    points = set().union(*(c.defense_points for c in continuations.values()))
    for t in fours_at(game, player, setup) + threes_at(game, player, setup):
        points.update(t.defenses)
    # Include remote counter-fours as well as local window blocks.
    for move in _window_candidates(game, -player, 3):
        if _four_completions(game, -player, move):
            points.add(move)
    points.update(_winning_moves(game, -player))
    return points


def future_setups(game: Game, player: int, *, limits=PlannerLimits(),
                  stats: PlanningStats | None = None) -> dict[Move, Setup]:
    """Require a linked compound after EVERY examined relevant legal reply.

    The response set is local threats plus global counter-fours, not all legal
    moves. Overflow skips a setup rather than silently dropping a defense.
    """
    if game.done:
        return {}
    stats = stats if stats is not None else PlanningStats()
    candidates = sorted(_window_candidates(game, player, 2),
                        key=lambda m: _structure_key(game, player, m))
    stats.structural_candidates += len(candidates)
    stats.setup_cap_hits += len(candidates) > limits.setups
    result = {}
    for move in candidates[:limits.setups]:
        if not _is_legal_for_player(game, player, move) or _wins_for_player(game, player, move):
            continue
        if compound_at(game, player, move) is not None:
            continue
        stats.examined_setups += 1
        with placed(game, player, move):
            if _winning_moves(game, -player):
                continue
            continuations = _continuations(game, player, move, limits, stats)
            if not continuations:
                continue
            responses = sorted(p for p in _responses(game, player, move, continuations)
                               if _is_legal_for_player(game, -player, p))
            if not responses:
                continue
            if len(responses) > limits.defenses:
                stats.defense_cap_skips += 1
                continue
            common_kinds = {'43'} if player == BLACK else {'33', '43', '44'}
            surviving = 0
            count = 0
            for reply in responses:
                if _wins_for_player(game, -player, reply):
                    common_kinds.clear()
                    break
                with placed(game, -player, reply):
                    following = _continuations(game, player, move, limits, stats)
                    kinds = set().union(*(c.kinds for c in following.values()))
                    common_kinds.intersection_update(kinds)
                    count += len(following)
                    surviving += bool(kinds)
                if not common_kinds:
                    break
            if common_kinds:
                defense_points = set(responses).union(continuations, {move})
                result[move] = Setup(move, frozenset(common_kinds), frozenset(defense_points),
                                     len(responses), surviving, count)
    return result
