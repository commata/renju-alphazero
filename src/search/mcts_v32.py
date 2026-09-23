"""MCTS V3.2: color-aware tactical candidate priorities on top of V3.1."""
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


@dataclass(frozen=True, slots=True)
class PatternFeatures:
    legal: bool
    immediate_win: bool
    max_run: int
    four_directions: int
    open_three_directions: int
    has_four_three: bool


_NO_PATTERN = PatternFeatures(False, False, 0, 0, 0, False)


def _legal_on_board(board: list[list[int]], player: int, move: Move) -> bool:
    row, col = move
    if not inside(row, col) or board[row][col] != EMPTY:
        return False
    return player == WHITE or forbidden_reason(board, row, col) is None


def _run_segment(
    board: list[list[int]],
    row: int,
    col: int,
    dr: int,
    dc: int,
) -> tuple[int, set[Move]]:
    """Return contiguous run length and coordinates containing (row, col)."""
    color = board[row][col]
    cells: set[Move] = {(row, col)}
    for sign in (-1, 1):
        rr, cc = row + sign * dr, col + sign * dc
        while inside(rr, cc) and board[rr][cc] == color:
            cells.add((rr, cc))
            rr += sign * dr
            cc += sign * dc
    return len(cells), cells


def _is_winning_run(player: int, length: int) -> bool:
    return length == 5 if player == BLACK else length >= 5


def _winning_extensions_in_direction(
    board: list[list[int]],
    player: int,
    anchor: Move,
    dr: int,
    dc: int,
) -> set[Move]:
    """Winning next moves in one direction whose winning run contains anchor."""
    row, col = anchor
    result: set[Move] = set()
    for step in range(-4, 5):
        if step == 0:
            continue
        rr, cc = row + step * dr, col + step * dc
        move = (rr, cc)
        if not inside(rr, cc) or not _legal_on_board(board, player, move):
            continue
        board[rr][cc] = player
        try:
            length, cells = _run_segment(board, rr, cc, dr, dc)
            if anchor in cells and _is_winning_run(player, length):
                result.add(move)
        finally:
            board[rr][cc] = EMPTY
    return result


def _open_three_in_direction(
    board: list[list[int]],
    player: int,
    anchor: Move,
    dr: int,
    dc: int,
) -> bool:
    """Whether one legal extension can create a straight four with two wins."""
    row, col = anchor
    for step in range(-4, 5):
        if step == 0:
            continue
        rr, cc = row + step * dr, col + step * dc
        extension = (rr, cc)
        if not inside(rr, cc) or not _legal_on_board(board, player, extension):
            continue

        board[rr][cc] = player
        try:
            length, cells = _run_segment(board, rr, cc, dr, dc)
            if anchor not in cells or _is_winning_run(player, length):
                continue
            if len(_winning_extensions_in_direction(board, player, anchor, dr, dc)) >= 2:
                return True
        finally:
            board[rr][cc] = EMPTY
    return False


def _pattern_features_for_move(game: Game, player: int, move: Move) -> PatternFeatures:
    """Evaluate 4/3 compound threats for one prospective legal move."""
    if not _legal_on_board(game.board, player, move):
        return _NO_PATTERN

    row, col = move
    game.board[row][col] = player
    try:
        lengths = [run_length(game.board, row, col, dr, dc) for dr, dc in DIRECTIONS]
        max_run = max(lengths)
        immediate_win = any(_is_winning_run(player, length) for length in lengths)
        if immediate_win:
            return PatternFeatures(True, True, max_run, 0, 0, False)

        four_dirs: set[tuple[int, int]] = set()
        three_dirs: set[tuple[int, int]] = set()
        for dr, dc in DIRECTIONS:
            if _winning_extensions_in_direction(game.board, player, move, dr, dc):
                four_dirs.add((dr, dc))
            if _open_three_in_direction(game.board, player, move, dr, dc):
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


def _local_components(game: Game, move: Move, player: int) -> tuple[int, int]:
    """Cheap attack/defense locality components around one move."""
    row, col = move
    attack = 0
    defense = 0
    for rr in range(max(0, row - 2), min(SIZE, row + 3)):
        for cc in range(max(0, col - 2), min(SIZE, col + 3)):
            stone = game.board[rr][cc]
            if stone == EMPTY:
                continue
            distance = max(abs(rr - row), abs(cc - col))
            if distance == 1:
                value = 6
            elif distance == 2:
                value = 2
            else:
                continue
            if stone == player:
                attack += value
            else:
                defense += value
    return attack, defense


def _v32_priority_score(game: Game, move: Move) -> int:
    """Color-aware tactical score. Larger is better."""
    player = game.to_play
    opponent = -player
    own = _pattern_features_for_move(game, player, move)
    if not own.legal:
        return -10**9

    opponent_pattern = _pattern_features_for_move(game, opponent, move)
    attack, defense = _local_components(game, move, player)

    if player == BLACK:
        # Black must already be legal, so 3-3, 4-4 and overline never enter.
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

    # White is defense-weighted but can legally exploit 4-4, 3-3 and overlines.
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


def _v32_move_key(game: Game, move: Move) -> tuple[int, int, int, int, int]:
    """Sort by V3.2 tactical score, then retain the V3.1 locality tie-break."""
    base = _move_score(game, move)
    return (-_v32_priority_score(game, move), *base)


def _search_candidates_v32(game: Game, limit: int, radius: int) -> list[Move]:
    """Prefilter cheaply, then do expensive tactical ranking on a bounded set."""
    prefilter_limit = max(limit * 2, limit)
    moves = _local_legal_candidates(game, prefilter_limit, radius)
    moves.sort(key=lambda move: _v32_move_key(game, move))
    return moves[:limit]


def _max_run_after_move(game: Game, player: int, move: Move) -> int:
    features = _pattern_features_for_move(game, player, move)
    return features.max_run if features.legal else -1


def _best_immediate_win(game: Game, player: int, wins: list[Move]) -> Move:
    """White prefers the longest winning run; black exact-five wins are tied."""
    if player == WHITE:
        return max(wins, key=lambda move: (_max_run_after_move(game, player, move), -move[0], -move[1]))
    return wins[0]


def _root_candidates_v32(
    game: Game,
    candidate_limit: int,
    radius: int,
) -> tuple[list[Move], Move | None]:
    """Full-board tactical safety followed by V3.2 color-aware ranking."""
    legal = game.legal_moves()
    if not legal:
        raise IllegalMove("No legal moves available")

    own_wins = _immediate_wins(game, game.to_play, legal)
    if own_wins:
        return legal, _best_immediate_win(game, game.to_play, own_wins)

    opponent = deepcopy(game)
    opponent.to_play = -game.to_play
    opponent_legal = opponent.legal_moves()
    opponent_wins = _immediate_wins(opponent, opponent.to_play, opponent_legal)

    legal_set = set(legal)
    blocking = [move for move in opponent_wins if move in legal_set]
    if len(opponent_wins) == 1 and blocking:
        return legal, blocking[0]

    candidates = _local_legal_candidates(game, candidate_limit * 2, radius)
    seen = set(candidates)
    for move in blocking:
        if move not in seen:
            candidates.append(move)
            seen.add(move)

    candidates.sort(key=lambda move: _v32_move_key(game, move))
    return candidates[:candidate_limit], None


def _rollout_move_v32(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> Move | None:
    """Keep rollout cheap while making locality attack/defense color-aware."""
    moves = _local_legal_candidates(game, candidate_limit, radius)
    if not moves:
        return None

    wins = _immediate_wins(game, game.to_play, moves)
    if wins:
        return _best_immediate_win(game, game.to_play, wins)

    opponent = -game.to_play
    blocks = [
        move
        for move in moves
        if _is_legal_for_player(game, opponent, move)
        and _wins_for_player(game, opponent, move)
    ]
    if blocks:
        return random.choice(blocks)

    player = game.to_play
    if player == BLACK:
        moves.sort(
            key=lambda move: (
                -(
                    _local_components(game, move, player)[0] * 14
                    + _local_components(game, move, player)[1] * 9
                ),
                *_move_score(game, move),
            )
        )
    else:
        moves.sort(
            key=lambda move: (
                -(
                    _local_components(game, move, player)[0] * 10
                    + _local_components(game, move, player)[1] * 14
                ),
                *_move_score(game, move),
            )
        )
    return _ranked_rollout_choice(moves, priority_top_k, random)


def _rollout_v32(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> int | None:
    while not game.done:
        move = _rollout_move_v32(
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


def mcts_search_v32(
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
    """V3.1 search structure with asymmetric Renju tactical priorities."""
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
    root_moves, forced = _root_candidates_v32(game, candidate_limit, neighborhood_radius)
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

        while not state.done and not _can_expand(node, initial_width) and node.children:
            node = _select_child(node, exploration)
            assert node.move is not None
            state.play(*node.move)

        if not state.done and _can_expand(node, initial_width):
            move = _pop_ranked_untried(node, priority_top_k, random)
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node,
                move=move,
                player_just_moved=player,
                untried_moves=(
                    []
                    if state.done
                    else _search_candidates_v32(state, candidate_limit, neighborhood_radius)
                ),
            )
            node.children.append(child)
            node = child

        winner = (
            state.winner
            if state.done
            else _rollout_v32(
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
    candidates = [child for child in root.children if child.visits == max_visits]
    max_value = max(child.mean_value for child in candidates)
    candidates = [child for child in candidates if child.mean_value == max_value]
    chosen = random.choice(candidates)
    assert chosen.move is not None
    return chosen.move
