"""MCTS V3.2.1: V3.2 policy with fast heuristic pattern scanning."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from random import Random

from renju import BLACK, EMPTY, SIZE, WHITE, Game, IllegalMove
from renju.rules import DIRECTIONS, forbidden_reason, inside, run_length

from .mcts import (
    MCTSNode,
    Move,
    _backpropagate,
    _immediate_wins,
    _is_legal_for_player,
    _move_score,
    _select_child,
    _wins_for_player,
)
from .mcts_v3 import (
    _can_expand,
    _local_legal_candidates,
    _pop_ranked_untried,
    _ranked_rollout_choice,
)
from .mcts_v32 import PatternFeatures, _local_components


_NO_PATTERN = PatternFeatures(False, False, 0, 0, 0, False)


def _is_winning_run(player: int, length: int) -> bool:
    return length == 5 if player == BLACK else length >= 5


def _run_info(
    board: list[list[int]],
    row: int,
    col: int,
    dr: int,
    dc: int,
    anchor: Move,
) -> tuple[int, bool]:
    """Return contiguous run length and whether that run contains anchor."""
    color = board[row][col]
    count = 1
    contains_anchor = (row, col) == anchor
    for sign in (-1, 1):
        rr, cc = row + sign * dr, col + sign * dc
        while inside(rr, cc) and board[rr][cc] == color:
            count += 1
            if (rr, cc) == anchor:
                contains_anchor = True
            rr += sign * dr
            cc += sign * dc
    return count, contains_anchor


def _black_overline_at(board: list[list[int]], row: int, col: int) -> bool:
    """Cheap hard rejection used only after a temporary black placement."""
    return any(
        run_length(board, row, col, dr, dc) >= 6
        for dr, dc in DIRECTIONS
    )


def _fast_winning_extensions_in_direction(
    board: list[list[int]],
    player: int,
    anchor: Move,
    dr: int,
    dc: int,
) -> set[Move]:
    """Find structural winning extensions without recursive forbidden checks.

    For black, an exact-five extension wins even if the same placement also
    creates an overline on another axis. Renju 3-3/4-4 checks are unnecessary
    after exact five, matching the rule engine's ordering.
    """
    row, col = anchor
    result: set[Move] = set()
    for step in range(-4, 5):
        if step == 0:
            continue
        rr, cc = row + step * dr, col + step * dc
        if not inside(rr, cc) or board[rr][cc] != EMPTY:
            continue

        board[rr][cc] = player
        try:
            length, contains_anchor = _run_info(
                board,
                rr,
                cc,
                dr,
                dc,
                anchor,
            )
            if not contains_anchor or not _is_winning_run(player, length):
                continue
            result.add((rr, cc))
        finally:
            board[rr][cc] = EMPTY
    return result


def _fast_open_three_in_direction(
    board: list[list[int]],
    player: int,
    anchor: Move,
    dr: int,
    dc: int,
) -> bool:
    """Approximate an open three using local geometry only.

    The top-level move is still checked by the real Renju rule engine. For
    hypothetical non-winning extensions we deliberately avoid recursive 3-3
    and 4-4 legality checks; this function is a ranking heuristic, not a rule
    authority. Obvious black overlines are still rejected cheaply.
    """
    row, col = anchor
    for step in range(-4, 5):
        if step == 0:
            continue
        rr, cc = row + step * dr, col + step * dc
        if not inside(rr, cc) or board[rr][cc] != EMPTY:
            continue

        board[rr][cc] = player
        try:
            # An extension that already wins is not a three -> straight-four
            # extension. Reject it before the cheaper overline heuristic.
            if any(
                _is_winning_run(player, run_length(board, rr, cc, vr, vc))
                for vr, vc in DIRECTIONS
            ):
                continue
            if player == BLACK and _black_overline_at(board, rr, cc):
                continue

            length, contains_anchor = _run_info(
                board,
                rr,
                cc,
                dr,
                dc,
                anchor,
            )
            if not contains_anchor:
                continue

            if len(
                _fast_winning_extensions_in_direction(
                    board,
                    player,
                    anchor,
                    dr,
                    dc,
                )
            ) >= 2:
                return True
        finally:
            board[rr][cc] = EMPTY
    return False


def _fast_pattern_features_for_move(
    game: Game,
    player: int,
    move: Move,
    *,
    assume_legal: bool = False,
) -> PatternFeatures:
    """Fast 4/3 feature scan with at most one exact top-level black legality check."""
    row, col = move
    if not inside(row, col) or game.board[row][col] != EMPTY:
        return _NO_PATTERN

    if not assume_legal and player == BLACK:
        # Exact legality is needed only for the candidate itself. Internal
        # hypothetical extensions use the fast structural scanner above.
        if forbidden_reason(game.board, row, col) is not None:
            return _NO_PATTERN

    game.board[row][col] = player
    try:
        lengths = [
            run_length(game.board, row, col, dr, dc)
            for dr, dc in DIRECTIONS
        ]
        max_run = max(lengths)
        immediate_win = any(
            _is_winning_run(player, length)
            for length in lengths
        )
        if immediate_win:
            return PatternFeatures(True, True, max_run, 0, 0, False)

        four_dirs: set[tuple[int, int]] = set()
        three_dirs: set[tuple[int, int]] = set()
        for dr, dc in DIRECTIONS:
            if _fast_winning_extensions_in_direction(
                game.board,
                player,
                move,
                dr,
                dc,
            ):
                four_dirs.add((dr, dc))
            if _fast_open_three_in_direction(
                game.board,
                player,
                move,
                dr,
                dc,
            ):
                three_dirs.add((dr, dc))

        has_four_three = any(
            four_direction != three_direction
            for four_direction in four_dirs
            for three_direction in three_dirs
        )
        return PatternFeatures(
            True,
            False,
            max_run,
            len(four_dirs),
            len(three_dirs),
            has_four_three,
        )
    finally:
        game.board[row][col] = EMPTY


def _v321_priority_score(game: Game, move: Move) -> int:
    """V3.2 weights with the V3.2.1 fast pattern detector."""
    player = game.to_play
    opponent = -player

    # _search_candidates_v321 receives moves from _local_legal_candidates, so
    # the current player's top-level move has already passed exact legality.
    own = _fast_pattern_features_for_move(
        game,
        player,
        move,
        assume_legal=True,
    )
    if not own.legal:
        return -10**9

    # If the opponent is black, reject a forbidden top-level threat exactly
    # once. White needs no legality check.
    opponent_pattern = _fast_pattern_features_for_move(
        game,
        opponent,
        move,
        assume_legal=opponent == WHITE,
    )
    attack, defense = _local_components(game, move, player)

    if player == BLACK:
        score = attack * 14 + defense * 9
        if own.immediate_win:
            score += 10_000
        if opponent_pattern.immediate_win:
            score += 9_000
        if own.has_four_three:
            score += 1_800
        if own.four_directions:
            score += 1_300
        if opponent_pattern.four_directions:
            score += 1_100
        if own.open_three_directions:
            score += 600
        return score

    score = attack * 10 + defense * 14
    if own.immediate_win:
        score += 10_000 + own.max_run * 10
    if opponent_pattern.immediate_win:
        score += 9_000
    if own.four_directions >= 2:
        score += 1_800
    if opponent_pattern.has_four_three:
        score += 1_500
    if opponent_pattern.four_directions:
        score += 1_400
    if own.four_directions:
        score += 1_200
    if own.open_three_directions >= 2:
        score += 900
    if opponent_pattern.open_three_directions:
        score += 700
    if own.open_three_directions:
        score += 500
    return score


def _v321_move_key(game: Game, move: Move) -> tuple[int, int, int, int, int]:
    base = _move_score(game, move)
    return (-_v321_priority_score(game, move), *base)


def _search_candidates_v321(game: Game, limit: int, radius: int) -> list[Move]:
    """Exact legal prefilter followed by fast bounded tactical ranking."""
    prefilter_limit = max(limit * 2, limit)
    moves = _local_legal_candidates(game, prefilter_limit, radius)

    scored = [
        (_v321_move_key(game, move), move)
        for move in moves
    ]
    scored.sort(key=lambda item: item[0])
    return [move for _, move in scored[:limit]]


def _max_run_after_move_fast(game: Game, player: int, move: Move) -> int:
    row, col = move
    game.board[row][col] = player
    try:
        return max(
            run_length(game.board, row, col, dr, dc)
            for dr, dc in DIRECTIONS
        )
    finally:
        game.board[row][col] = EMPTY


def _best_immediate_win_fast(game: Game, player: int, wins: list[Move]) -> Move:
    if player == WHITE:
        return max(
            wins,
            key=lambda move: (
                _max_run_after_move_fast(game, player, move),
                -move[0],
                -move[1],
            ),
        )
    return wins[0]


def _root_candidates_v321(
    game: Game,
    candidate_limit: int,
    radius: int,
) -> tuple[list[Move], Move | None]:
    """Full-board tactical safety followed by optimized V3.2 ranking."""
    legal = game.legal_moves()
    if not legal:
        raise IllegalMove("No legal moves available")

    own_wins = _immediate_wins(game, game.to_play, legal)
    if own_wins:
        return legal, _best_immediate_win_fast(game, game.to_play, own_wins)

    opponent = deepcopy(game)
    opponent.to_play = -game.to_play
    opponent_legal = opponent.legal_moves()
    opponent_wins = _immediate_wins(
        opponent,
        opponent.to_play,
        opponent_legal,
    )

    legal_set = set(legal)
    blocking = [move for move in opponent_wins if move in legal_set]
    if len(opponent_wins) == 1 and blocking:
        return legal, blocking[0]

    candidates = _local_legal_candidates(
        game,
        candidate_limit * 2,
        radius,
    )
    seen = set(candidates)
    for move in blocking:
        if move not in seen:
            candidates.append(move)
            seen.add(move)

    scored = [
        (_v321_move_key(game, move), move)
        for move in candidates
    ]
    scored.sort(key=lambda item: item[0])
    return [move for _, move in scored[:candidate_limit]], None


def _rollout_local_score(game: Game, move: Move, player: int) -> int:
    attack, defense = _local_components(game, move, player)
    if player == BLACK:
        return attack * 14 + defense * 9
    return attack * 10 + defense * 14


def _rollout_move_v321(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> Move | None:
    """Same rollout policy as V3.2 with duplicated locality work removed."""
    moves = _local_legal_candidates(game, candidate_limit, radius)
    if not moves:
        return None

    wins = _immediate_wins(game, game.to_play, moves)
    if wins:
        return _best_immediate_win_fast(game, game.to_play, wins)

    opponent = -game.to_play
    blocks = [
        move
        for move in moves
        # Run-length checking is much cheaper than exact black Renju legality.
        # Only an actual winning point needs the forbidden-move check.
        if _wins_for_player(game, opponent, move)
        and _is_legal_for_player(game, opponent, move)
    ]
    if blocks:
        return random.choice(blocks)

    player = game.to_play
    moves.sort(
        key=lambda move: (
            -_rollout_local_score(game, move, player),
            *_move_score(game, move),
        )
    )
    return _ranked_rollout_choice(moves, priority_top_k, random)


def _rollout_v321(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> int | None:
    while not game.done:
        move = _rollout_move_v321(
            game,
            random,
            candidate_limit,
            radius,
            priority_top_k,
        )
        if move is None:
            return None
        game.play(*move)
    return game.winner


def mcts_search_v321(
    game: Game,
    *,
    simulations: int = 25,
    exploration: float = sqrt(2.0),
    candidate_limit: int = 16,
    initial_width: int = 6,
    neighborhood_radius: int = 2,
    priority_top_k: int = 5,
    random: Random | None = None,
) -> Move:
    """V3.2 policy with recursive forbidden checks removed from heuristic extensions."""
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

    random = random or Random()
    root_moves, forced = _root_candidates_v321(
        game,
        candidate_limit,
        neighborhood_radius,
    )
    if forced is not None:
        return forced

    root = MCTSNode(
        parent=None,
        move=None,
        player_just_moved=None,
        untried_moves=root_moves[:],
    )
    state = deepcopy(game)
    root_history_length = len(state.history)

    for _ in range(simulations):
        node = root

        while (
            not state.done
            and not _can_expand(node, initial_width)
            and node.children
        ):
            node = _select_child(node, exploration)
            assert node.move is not None
            state.play(*node.move)

        if not state.done and _can_expand(node, initial_width):
            move = _pop_ranked_untried(
                node,
                priority_top_k,
                random,
            )
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node,
                move=move,
                player_just_moved=player,
                untried_moves=(
                    []
                    if state.done
                    else _search_candidates_v321(
                        state,
                        candidate_limit,
                        neighborhood_radius,
                    )
                ),
            )
            node.children.append(child)
            node = child

        winner = (
            state.winner
            if state.done
            else _rollout_v321(
                state,
                random,
                candidate_limit,
                neighborhood_radius,
                priority_top_k,
            )
        )
        _backpropagate(node, winner)

        while len(state.history) > root_history_length:
            state.undo()

    max_visits = max(child.visits for child in root.children)
    candidates = [
        child
        for child in root.children
        if child.visits == max_visits
    ]
    max_value = max(child.mean_value for child in candidates)
    candidates = [
        child
        for child in candidates
        if child.mean_value == max_value
    ]
    chosen = random.choice(candidates)
    assert chosen.move is not None
    return chosen.move
