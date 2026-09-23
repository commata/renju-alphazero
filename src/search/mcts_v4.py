"""MCTS V4: force defense against opponent open-three continuations."""
from __future__ import annotations

from copy import deepcopy
from math import sqrt
from random import Random

from renju import BLACK, EMPTY, WHITE, Game, IllegalMove
from renju.rules import DIRECTIONS, forbidden_reason, inside

from .mcts import Move, _immediate_wins
from .mcts_v321 import (
    _best_immediate_win_fast,
    _fast_winning_extensions_in_direction,
    _v321_move_key,
    mcts_search_v321,
)


def _creates_open_four(
    game: Game,
    player: int,
    move: Move,
    *,
    assume_legal: bool = False,
) -> bool:
    """Return whether move turns the player's current three into an open four.

    V4 defines the tactical trigger as a move that creates a four with at least
    two distinct immediate winning extensions in one direction. For black, the
    top-level creator move must be legal under Renju rules.
    """
    row, col = move
    if not inside(row, col) or game.board[row][col] != EMPTY:
        return False
    if player == BLACK and not assume_legal:
        if forbidden_reason(game.board, row, col) is not None:
            return False

    game.board[row][col] = player
    try:
        for dr, dc in DIRECTIONS:
            winning_extensions = _fast_winning_extensions_in_direction(
                game.board,
                player,
                move,
                dr,
                dc,
            )
            if len(winning_extensions) >= 2:
                return True
        return False
    finally:
        game.board[row][col] = EMPTY


def _open_three_creator_moves(
    game: Game,
    player: int,
    legal_moves: list[Move],
) -> list[Move]:
    """Legal moves that convert an existing open-three threat into open four."""
    return [
        move
        for move in legal_moves
        if _creates_open_four(
            game,
            player,
            move,
            assume_legal=True,
        )
    ]


def _remaining_open_three_creators_after_block(
    game: Game,
    block: Move,
) -> int:
    """Count opponent open-four creators after the current player blocks."""
    state = deepcopy(game)
    state.play(*block)
    if state.done:
        return 0

    opponent_legal = state.legal_moves()
    return len(
        _open_three_creator_moves(
            state,
            state.to_play,
            opponent_legal,
        )
    )


def _forced_v4_move(game: Game) -> Move | None:
    """Return a tactical move that must override MCTS at the real root.

    Priority:
    1. take an immediate win;
    2. block an opponent immediate win;
    3. if the opponent can turn an existing three into an open four, occupy a
       creator point immediately.

    When multiple open-three blocks exist, choose the legal block that leaves
    the fewest remaining open-four creator moves. V3.2.1's tactical ordering is
    used only as a deterministic tie-break.
    """
    legal = game.legal_moves()
    if not legal:
        raise IllegalMove("No legal moves available")

    own_wins = _immediate_wins(game, game.to_play, legal)
    if own_wins:
        return _best_immediate_win_fast(game, game.to_play, own_wins)

    opponent = deepcopy(game)
    opponent.to_play = -game.to_play
    opponent_legal = opponent.legal_moves()
    opponent_wins = _immediate_wins(
        opponent,
        opponent.to_play,
        opponent_legal,
    )

    legal_set = set(legal)
    immediate_blocks = [
        move
        for move in opponent_wins
        if move in legal_set
    ]
    if immediate_blocks:
        return min(
            immediate_blocks,
            key=lambda move: _v321_move_key(game, move),
        )

    creator_moves = _open_three_creator_moves(
        opponent,
        opponent.to_play,
        opponent_legal,
    )
    blocks = [
        move
        for move in creator_moves
        if move in legal_set
    ]
    if not blocks:
        return None

    return min(
        blocks,
        key=lambda move: (
            _remaining_open_three_creators_after_block(game, move),
            _v321_move_key(game, move),
        ),
    )


def _validate_v4_config(
    simulations: int,
    exploration: float,
    candidate_limit: int,
    initial_width: int,
    neighborhood_radius: int,
    priority_top_k: int,
) -> None:
    if type(simulations) is not int or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if exploration <= 0:
        raise ValueError("exploration must be positive")
    if type(candidate_limit) is not int or candidate_limit <= 0:
        raise ValueError("candidate_limit must be a positive integer")
    if type(initial_width) is not int or initial_width <= 0:
        raise ValueError("initial_width must be a positive integer")
    if initial_width > candidate_limit:
        raise ValueError("initial_width must not exceed candidate_limit")
    if type(neighborhood_radius) is not int or neighborhood_radius <= 0:
        raise ValueError("neighborhood_radius must be a positive integer")
    if type(priority_top_k) is not int or priority_top_k <= 0:
        raise ValueError("priority_top_k must be a positive integer")
    if priority_top_k > candidate_limit:
        raise ValueError("priority_top_k must not exceed candidate_limit")


def mcts_search_v4(
    game: Game,
    *,
    simulations: int = 50,
    exploration: float = sqrt(2.0),
    candidate_limit: int = 20,
    initial_width: int = 8,
    neighborhood_radius: int = 2,
    priority_top_k: int = 8,
    random: Random | None = None,
) -> Move:
    """V3.2.1 search plus a hard root policy for opponent open-three defense."""
    _validate_v4_config(
        simulations,
        exploration,
        candidate_limit,
        initial_width,
        neighborhood_radius,
        priority_top_k,
    )

    forced = _forced_v4_move(game)
    if forced is not None:
        return forced

    return mcts_search_v321(
        game,
        simulations=simulations,
        exploration=exploration,
        candidate_limit=candidate_limit,
        initial_width=initial_width,
        neighborhood_radius=neighborhood_radius,
        priority_top_k=priority_top_k,
        random=random,
    )
