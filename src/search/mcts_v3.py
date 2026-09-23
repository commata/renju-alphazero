"""Third pure-MCTS revision: wider candidate pool with progressive widening."""
from __future__ import annotations

from copy import deepcopy
from math import sqrt
from random import Random

from renju import EMPTY, SIZE, Game, IllegalMove

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


def _neighborhood_pool(game: Game, radius: int) -> list[Move]:
    """Return empty cells near existing stones; use a center window on an empty board."""
    occupied = [
        (row, col)
        for row in range(SIZE)
        for col in range(SIZE)
        if game.board[row][col] != EMPTY
    ]
    pool: set[Move] = set()

    if not occupied:
        center = SIZE // 2
        for row in range(max(0, center - radius), min(SIZE, center + radius + 1)):
            for col in range(max(0, center - radius), min(SIZE, center + radius + 1)):
                if game.board[row][col] == EMPTY:
                    pool.add((row, col))
        return list(pool)

    for base_row, base_col in occupied:
        for row in range(max(0, base_row - radius), min(SIZE, base_row + radius + 1)):
            for col in range(max(0, base_col - radius), min(SIZE, base_col + radius + 1)):
                if game.board[row][col] == EMPTY:
                    pool.add((row, col))
    return list(pool)


def _local_legal_candidates(
    game: Game,
    limit: int,
    radius: int,
) -> list[Move]:
    """Score only a local pool, then validate enough moves to fill the shortlist."""
    if game.done:
        return []

    pool = _neighborhood_pool(game, radius)
    ranked = sorted(pool, key=lambda move: _move_score(game, move))
    result: list[Move] = []
    seen: set[Move] = set()

    for move in ranked:
        seen.add(move)
        if _is_legal_for_player(game, game.to_play, move):
            result.append(move)
            if len(result) >= limit:
                return result

    if len(result) < limit:
        for move in game.legal_moves():
            if move in seen:
                continue
            result.append(move)
            if len(result) >= limit:
                break
    return result


def _root_candidates_v3(
    game: Game,
    candidate_limit: int,
    radius: int,
) -> tuple[list[Move], Move | None]:
    """Use full-board tactical safety, then keep a wider local search pool."""
    legal = game.legal_moves()
    if not legal:
        raise IllegalMove("No legal moves available")

    own_wins = _immediate_wins(game, game.to_play, legal)
    if own_wins:
        return legal, own_wins[0]

    opponent = deepcopy(game)
    opponent.to_play = -game.to_play
    opponent_legal = opponent.legal_moves()
    opponent_wins = _immediate_wins(opponent, opponent.to_play, opponent_legal)

    legal_set = set(legal)
    blocking = [move for move in opponent_wins if move in legal_set]
    if len(opponent_wins) == 1 and blocking:
        return legal, blocking[0]

    local = _neighborhood_pool(game, radius)
    local_legal = [move for move in local if move in legal_set]
    local_legal.sort(key=lambda move: _move_score(game, move))

    result: list[Move] = []
    seen: set[Move] = set()
    for move in blocking:
        if move not in seen:
            result.append(move)
            seen.add(move)

    for move in local_legal:
        if move not in seen:
            result.append(move)
            seen.add(move)
            if len(result) >= candidate_limit:
                return result, None

    center = SIZE // 2
    remaining = [move for move in legal if move not in seen]
    remaining.sort(key=lambda move: (
        abs(move[0] - center) + abs(move[1] - center),
        move[0],
        move[1],
    ))
    for move in remaining:
        result.append(move)
        if len(result) >= candidate_limit:
            break
    return result, None


def _allowed_children(node: MCTSNode, initial_width: int) -> int:
    """Progressively expose more of the candidate pool as a node is revisited."""
    total = len(node.children) + len(node.untried_moves)
    if total == 0:
        return 0
    return min(total, initial_width + int(sqrt(node.visits)))


def _can_expand(node: MCTSNode, initial_width: int) -> bool:
    return bool(node.untried_moves) and len(node.children) < _allowed_children(node, initial_width)


def _rank_weights(count: int) -> list[int]:
    """Return descending rank weights, e.g. 5, 4, 3, 2, 1."""
    return list(range(count, 0, -1))


def _ranked_choice_index(length: int, top_k: int, random: Random) -> int:
    """Choose only among the highest-ranked remaining candidates."""
    width = min(length, top_k)
    return random.choices(
        range(width),
        weights=_rank_weights(width),
        k=1,
    )[0]


def _pop_ranked_untried(node: MCTSNode, top_k: int, random: Random) -> Move:
    """Pop one untried move from the ranked top-k window."""
    index = _ranked_choice_index(len(node.untried_moves), top_k, random)
    return node.untried_moves.pop(index)


def _ranked_rollout_choice(moves: list[Move], top_k: int, random: Random) -> Move:
    """Choose a rollout move from the ranked top-k window without removing it."""
    index = _ranked_choice_index(len(moves), top_k, random)
    return moves[index]


def _rollout_move_v3(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> Move | None:
    """Tactical rollout from a local candidate pool."""
    moves = _local_legal_candidates(game, candidate_limit, radius)
    if not moves:
        return None

    wins = _immediate_wins(game, game.to_play, moves)
    if wins:
        return wins[0]

    opponent = -game.to_play
    blocks = [
        move
        for move in moves
        if _is_legal_for_player(game, opponent, move)
        and _wins_for_player(game, opponent, move)
    ]
    if blocks:
        return random.choice(blocks)
    return _ranked_rollout_choice(moves, priority_top_k, random)


def _rollout_v3(
    game: Game,
    random: Random,
    candidate_limit: int,
    radius: int,
    priority_top_k: int,
) -> int | None:
    while not game.done:
        move = _rollout_move_v3(
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


def mcts_search_v3(
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
    """Return a move using a wider pool and progressive widening.

    V3 uses a 25-simulation budget and up to 16 ranked candidates. It starts
    with a width of six and exposes more candidates as visits grow. New
    expansions and non-tactical rollout choices are restricted to the best
    five remaining candidates and sampled with descending rank weights.
    """
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
    root_moves, forced = _root_candidates_v3(game, candidate_limit, neighborhood_radius)
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
                    else _local_legal_candidates(state, candidate_limit, neighborhood_radius)
                ),
            )
            node.children.append(child)
            node = child

        winner = (
            state.winner
            if state.done
            else _rollout_v3(
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
