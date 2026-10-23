"""H6-4: PUCT with a value leaf (docs/h6-value.md, H6-4; docs/mcts-v8-teacher.md §12.19).

A copy of ``analysis.puct_v8.search_tree_puct`` (frozen, unchanged) that differs in the leaf only; the
action sets, priors, selection rule and move choice are the same. Torch-free: the value comes in as a
callable ``leaf_value(game) -> v`` in [-1, 1] from the side to move's view (``hybrid.h6_value_leaf``).

Leaf of a simulation (λ = 1.0, fixed by the H6-3 gate: no rollout mix):
- the game is over: the result, as before (+1 / -1 for the player who moved into the node, 0 draw);
- exact shortcuts, before any value (solver beats value):
    the side to move can complete a five           -> the side to move wins
    otherwise the opponent has two or more fives   -> the side to move loses (one stone blocks one)
- otherwise v = leaf_value(state); the node's value (player who moved into it) is **-v**.
The node is expanded first in every case, as in ``puct_v8``, so the tree grows the same way.

Backup adds the value to every node on the path from the perspective of the player who moved into it,
so a parent still maximizes its children's mean value.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from time import perf_counter

from renju import Game
from search.mcts_v321 import _search_candidates_v321
from search.tactics import winning_points

from .puct_v8 import PUCT_PRIORS, PUCTNode, PUCTStats, _expand, _select

Move = tuple[int, int]


@dataclass
class PUCTValueStats(PUCTStats):
    value_calls: int = 0
    value_seconds: float = 0.0
    exact_leaves: int = 0     # leaves decided by the five / double-five shortcut
    terminal_leaves: int = 0  # leaves where the game was already over
    # Value of the evaluated leaves by the colour to move there (bias watch, H6-3 G5 note).
    black_leaves: int = 0
    black_value_sum: float = 0.0
    white_leaves: int = 0
    white_value_sum: float = 0.0


def exact_leaf(state: Game) -> float | None:
    """+1 / -1 (side to move's view) when a five settles the position at once, else None."""
    player = state.to_play
    if winning_points(state.board, player):
        return 1.0
    if len(winning_points(state.board, -player)) >= 2:
        return -1.0
    return None


def backup(node: PUCTNode, value: float, player: int) -> None:
    """Add ``value`` (from ``player``'s view) to every node up to the root, each from the view of the
    player who moved into it (``search.mcts._backpropagate`` with a real-valued result)."""
    current: PUCTNode | None = node
    while current is not None:
        current.visits += 1
        if current.player_just_moved is not None:
            current.value_sum += value if current.player_just_moved == player else -value
        current = current.parent


def leaf_evaluation(state: Game, leaf_value, stats: PUCTValueStats) -> tuple[float, int]:
    """(value, player whose view it is) of a non-terminal leaf: the exact shortcut, else the network."""
    player = state.to_play
    exact = exact_leaf(state)
    if exact is not None:
        stats.exact_leaves += 1
        return exact, player
    started = perf_counter()
    v = float(leaf_value(state))
    stats.value_seconds += perf_counter() - started
    stats.value_calls += 1
    if not -1.0 <= v <= 1.0:
        raise ValueError(f'leaf value {v} is outside [-1, 1]')
    if player > 0:
        stats.black_leaves += 1
        stats.black_value_sum += v
    else:
        stats.white_leaves += 1
        stats.white_value_sum += v
    return v, player


def search_tree_puct_value(game: Game, root_moves: list[Move], simulations: int, *, c_puct: float, prior: str,
                           policy=None, leaf_value=None, candidate_limit: int, neighborhood_radius: int,
                           priority_top_k: int, random: Random | None = None,
                           stats: PUCTValueStats | None = None):
    """Returns (chosen move, root children), like ``puct_v8.search_tree_puct``.

    ``random`` and ``priority_top_k`` are unused (no rollout); they are kept so the call matches the
    rollout tree's."""
    if prior not in PUCT_PRIORS:
        raise ValueError(f"prior must be one of {PUCT_PRIORS}")
    if prior == "policy" and policy is None:
        raise ValueError("the policy prior needs a policy callable")
    if leaf_value is None:
        raise ValueError("search_tree_puct_value needs a leaf_value callable")
    if not root_moves:
        raise ValueError("search_tree_puct_value needs at least one root move")
    stats = stats if stats is not None else PUCTValueStats()
    state = deepcopy(game)
    root_length = len(state.history)
    root = PUCTNode(parent=None, move=None, player_just_moved=None)
    _expand(root, state, list(root_moves), prior, policy, stats)
    stats.root_prior = {c.move: c.prior for c in root.children}
    for _ in range(simulations):
        node = root
        while node.expanded and node.children and not state.done:
            node = _select(node, c_puct)
            state.play(*node.move)
        if state.done:
            stats.terminal_leaves += 1
            if state.winner is None:
                backup(node, 0.0, state.to_play)
            else:
                backup(node, 1.0, state.winner)
        else:
            _expand(node, state, _search_candidates_v321(state, candidate_limit, neighborhood_radius),
                    prior, policy, stats)
            value, player = leaf_evaluation(state, leaf_value, stats)
            backup(node, value, player)
        while len(state.history) > root_length:
            state.undo()
        stats.simulations += 1
    best = max(root.children, key=lambda c: (c.visits, c.mean_value, c.prior, (-c.move[0], -c.move[1])))
    return best.move, root.children
