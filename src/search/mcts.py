"""Pure Monte Carlo Tree Search without a neural network."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import log, sqrt
from random import Random

from renju import BLACK, EMPTY, SIZE, WHITE, Game, IllegalMove
from renju.rules import DIRECTIONS, forbidden_reason, run_length


Move = tuple[int, int]


@dataclass(slots=True)
class MCTSNode:
    """One state in the search tree.

    value_sum is stored from the perspective of player_just_moved. This lets
    every player maximize child values on its own turn while signs alternate
    naturally during backpropagation.
    """

    parent: MCTSNode | None
    move: Move | None
    player_just_moved: int | None
    untried_moves: list[Move]
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _wins_for_player(game: Game, player: int, move: Move) -> bool:
    """Check a prospective already-legal move without changing Game state."""
    row, col = move
    game.board[row][col] = player
    try:
        lengths = (run_length(game.board, row, col, dr, dc) for dr, dc in DIRECTIONS)
        return any(length == 5 if player == BLACK else length >= 5 for length in lengths)
    finally:
        game.board[row][col] = EMPTY


def _is_legal_for_player(game: Game, player: int, move: Move) -> bool:
    row, col = move
    if game.board[row][col] != EMPTY:
        return False
    return player == WHITE or forbidden_reason(game.board, row, col) is None


def _move_score(game: Game, move: Move) -> tuple[int, int, int, int]:
    """Rank local moves without changing legality or game rules."""
    row, col = move
    score = 0
    for rr in range(max(0, row - 2), min(SIZE, row + 3)):
        for cc in range(max(0, col - 2), min(SIZE, col + 3)):
            stone = game.board[rr][cc]
            if stone == EMPTY:
                continue
            distance = max(abs(rr - row), abs(cc - col))
            if distance == 1:
                score += 6 if stone == game.to_play else 5
            elif distance == 2:
                score += 2 if stone == game.to_play else 1
    center_distance = abs(row - SIZE // 2) + abs(col - SIZE // 2)
    return (-score, center_distance, row, col)


def _shortlist_from_legal(
    game: Game,
    legal_moves: list[Move],
    limit: int,
    priority: tuple[Move, ...] = (),
) -> list[Move]:
    """Keep tactical priorities, then the strongest local legal candidates."""
    if len(legal_moves) <= limit:
        return legal_moves[:]

    legal = set(legal_moves)
    result: list[Move] = []
    seen: set[Move] = set()
    for move in priority:
        if move in legal and move not in seen:
            result.append(move)
            seen.add(move)

    for move in sorted(legal_moves, key=lambda item: _move_score(game, item)):
        if move in seen:
            continue
        result.append(move)
        seen.add(move)
        if len(result) >= max(limit, len(priority)):
            break
    return result


def _legal_candidates(game: Game, limit: int) -> list[Move]:
    """Find a small legal shortlist without generating every black legal move.

    Empty cells are ranked first, then legality is checked only until the
    shortlist is full. This is a search heuristic; Game.legal_moves() remains
    the source of truth for rules and is used as a defensive fallback.
    """
    if game.done:
        return []

    empties = [
        (row, col)
        for row in range(SIZE)
        for col in range(SIZE)
        if game.board[row][col] == EMPTY
    ]
    empties.sort(key=lambda item: _move_score(game, item))

    result: list[Move] = []
    for move in empties:
        if _is_legal_for_player(game, game.to_play, move):
            result.append(move)
            if len(result) >= limit:
                return result

    if result:
        return result
    return game.legal_moves()


def _immediate_wins(game: Game, player: int, moves: list[Move]) -> list[Move]:
    return [move for move in moves if _wins_for_player(game, player, move)]


def _root_candidates(game: Game, limit: int) -> tuple[list[Move], Move | None]:
    """Return root shortlist and an optional forced tactical move."""
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

    # A single immediate opponent win is a forced block when that point is
    # legal for us. Multiple distinct winning points cannot all be occupied by
    # one move, so keep them as high-priority candidates rather than pretending
    # there is a unique forced defense.
    legal_set = set(legal)
    blocking = tuple(move for move in opponent_wins if move in legal_set)
    if len(opponent_wins) == 1 and blocking:
        return legal, blocking[0]

    return _shortlist_from_legal(game, legal, limit, blocking), None


def _select_child(node: MCTSNode, exploration: float) -> MCTSNode:
    """Select a fully expanded child with UCT."""
    log_parent = log(node.visits)
    return max(
        node.children,
        key=lambda child: (
            child.mean_value
            + exploration * sqrt(log_parent / child.visits)
        ),
    )


def _rollout_move(game: Game, random: Random, candidate_limit: int) -> Move | None:
    """Tactical rollout: win, block a nearby immediate win, then random local play."""
    moves = _legal_candidates(game, candidate_limit)
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
    return random.choice(moves)


def _rollout(game: Game, random: Random, candidate_limit: int) -> int | None:
    """Finish one simulation with a small tactical/local rollout policy."""
    while not game.done:
        move = _rollout_move(game, random, candidate_limit)
        if move is None:
            return None
        game.play(*move)
    return game.winner


def _backpropagate(node: MCTSNode, winner: int | None) -> None:
    """Propagate a terminal result using each node's player perspective."""
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        if current.player_just_moved is not None and winner is not None:
            current.value_sum += 1.0 if winner == current.player_just_moved else -1.0
        current = current.parent


def mcts_search(
    game: Game,
    *,
    simulations: int = 10,
    exploration: float = sqrt(2.0),
    candidate_limit: int = 8,
    random: Random | None = None,
) -> Move:
    """Return a legal move using UCT MCTS with a local tactical shortlist.

    The caller's Game is never mutated. The default eight search candidates
    allow a ten-simulation budget to finish initial expansion and enter UCT
    selection instead of spending every simulation on a different root move.
    """
    if type(simulations) is not int or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if exploration <= 0:
        raise ValueError("exploration must be positive")
    if type(candidate_limit) is not int or candidate_limit <= 0:
        raise ValueError("candidate_limit must be a positive integer")

    random = random or Random()
    root_moves, forced = _root_candidates(game, candidate_limit)
    if forced is not None:
        return forced

    root = MCTSNode(
        parent=None,
        move=None,
        player_just_moved=None,
        untried_moves=root_moves[:],
    )

    # Reuse one private state and undo simulations back to the root instead of
    # deepcopying the whole Game for every simulation.
    state = deepcopy(game)
    root_history_length = len(state.history)

    for _ in range(simulations):
        node = root

        # Selection: follow UCT while this shortlisted node is fully expanded.
        while not state.done and not node.untried_moves and node.children:
            node = _select_child(node, exploration)
            assert node.move is not None
            state.play(*node.move)

        # Expansion: add one previously untried shortlisted legal move.
        if not state.done and node.untried_moves:
            index = random.randrange(len(node.untried_moves))
            move = node.untried_moves.pop(index)
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node,
                move=move,
                player_just_moved=player,
                untried_moves=[] if state.done else _legal_candidates(state, candidate_limit),
            )
            node.children.append(child)
            node = child

        winner = state.winner if state.done else _rollout(state, random, candidate_limit)
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
