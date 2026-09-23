"""Pure Monte Carlo Tree Search without a neural network."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import log, sqrt
from random import Random

from renju import Game, IllegalMove


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


def _rollout(game: Game, random: Random) -> int | None:
    """Finish one simulation with uniformly random legal moves."""
    while not game.done:
        moves = game.legal_moves()
        if not moves:
            # Game.play() normally marks this case done. Keep a defensive draw
            # fallback for externally constructed Game states used in tests.
            return None
        game.play(*random.choice(moves))
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
    random: Random | None = None,
) -> Move:
    """Return a legal move using pure UCT MCTS and random rollouts.

    The caller's Game is never mutated. Ten simulations is deliberately a
    small structural baseline; later stages can raise the budget after timing
    and strength are measured.
    """
    if type(simulations) is not int or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if exploration <= 0:
        raise ValueError("exploration must be positive")

    root_moves = game.legal_moves()
    if not root_moves:
        raise IllegalMove("No legal moves available")

    random = random or Random()
    root = MCTSNode(
        parent=None,
        move=None,
        player_just_moved=None,
        untried_moves=root_moves[:],
    )

    for _ in range(simulations):
        state = deepcopy(game)
        node = root

        # Selection: follow UCT while the node is fully expanded.
        while not state.done and not node.untried_moves and node.children:
            node = _select_child(node, exploration)
            state.play(*node.move)

        # Expansion: add one previously untried legal move.
        if not state.done and node.untried_moves:
            index = random.randrange(len(node.untried_moves))
            move = node.untried_moves.pop(index)
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node,
                move=move,
                player_just_moved=player,
                untried_moves=[] if state.done else state.legal_moves(),
            )
            node.children.append(child)
            node = child

        # Simulation and backpropagation.
        winner = state.winner if state.done else _rollout(state, random)
        _backpropagate(node, winner)

    # Prefer the robust child (most visits). mean_value breaks visit ties,
    # which is useful with the intentionally tiny initial budget.
    max_visits = max(child.visits for child in root.children)
    candidates = [child for child in root.children if child.visits == max_visits]
    max_value = max(child.mean_value for child in candidates)
    candidates = [child for child in candidates if child.mean_value == max_value]
    chosen = random.choice(candidates)
    assert chosen.move is not None
    return chosen.move
