"""H5: PUCT tree for V8's tree route (docs/mcts-v8-teacher.md §12.16).

Replaces only the V5 widening tree (``mcts_v8._search_tree_v8``); every V8 module
around it (Stage 1-5, own VCF, V8-B, V8-A, V8-C) is unchanged. Torch-free: a policy
prior comes in as a callable ``game -> {move: probability}`` (``hybrid.h4_policy``).

Kept identical to the V5 tree, so the arms differ only in the selection rule and prior:
- action sets: the root children are V8's prepared root list (V6 candidates -> VCF
  safety tiers -> M3), an inner node's children are ``_search_candidates_v321``,
  the same lists the V5 tree widens over. All children are present from the start
  (no widening); the prior decides which ones get visits;
- leaf value: ``_rollout_v321`` to the end of the game (+1 / -1 / 0), backed up with
  ``search.mcts._backpropagate`` (value from the perspective of the player who moved
  into the node, so a parent maximizes its children's mean value).

Selection (Track A ``search.alphazero.puct_score``, FPU 0):

    Q(child) + c_puct * P(child) * sqrt(max(1, N(parent))) / (1 + N(child))

Priors over a node's children (they always sum to 1 over exactly those children):
- ``uniform``: 1 / n;
- ``heuristic``: proportional to 1 / rank in the list's own order (both lists are
  ranked by V8's heuristics; no temperature to tune);
- ``policy``: the H3 policy (masked softmax over legal moves) restricted to the
  children and renormalized; if it puts no mass on them the node falls back to
  uniform and the fallback is counted.

The move is the most visited root child, then the higher mean value, then the
higher prior, then the smaller move.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import log, sqrt
from random import Random

from renju import Game
from search.mcts import _backpropagate
from search.mcts_v321 import _rollout_v321, _search_candidates_v321

Move = tuple[int, int]
PUCT_PRIORS = ("uniform", "heuristic", "policy")


@dataclass(eq=False)
class PUCTNode:
    parent: PUCTNode | None
    move: Move | None
    player_just_moved: int | None
    prior: float = 1.0
    children: list[PUCTNode] = field(default_factory=list)
    expanded: bool = False
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class PUCTStats:
    simulations: int = 0
    nn_calls: int = 0
    prior_fallbacks: int = 0
    root_prior: dict = field(default_factory=dict)


def puct_score(parent: PUCTNode, child: PUCTNode, c_puct: float) -> float:
    return child.mean_value + c_puct * child.prior * sqrt(max(1, parent.visits)) / (1 + child.visits)


def _select(node: PUCTNode, c_puct: float) -> PUCTNode:
    return max(node.children, key=lambda c: (puct_score(node, c, c_puct), c.prior, (-c.move[0], -c.move[1])))


def child_priors(game: Game, moves: list[Move], prior: str, policy, stats: PUCTStats) -> list[float]:
    n = len(moves)
    if prior == "uniform":
        return [1.0 / n] * n
    if prior == "heuristic":
        weights = [1.0 / (i + 1) for i in range(n)]
    elif prior == "policy":
        scores = policy(game)
        stats.nn_calls += 1
        weights = [max(0.0, float(scores.get(m, 0.0))) for m in moves]
        if sum(weights) <= 0.0:
            stats.prior_fallbacks += 1
            return [1.0 / n] * n
    else:
        raise ValueError(f"unknown PUCT prior {prior!r}")
    total = sum(weights)
    return [w / total for w in weights]


def _expand(node: PUCTNode, state: Game, moves: list[Move], prior: str, policy, stats: PUCTStats) -> None:
    node.expanded = True
    if not moves:
        return
    player = state.to_play
    for move, p in zip(moves, child_priors(state, moves, prior, policy, stats)):
        node.children.append(PUCTNode(parent=node, move=move, player_just_moved=player, prior=p))


def prior_entropy(priors: list[float]) -> float:
    """Entropy normalized to [0, 1] (1 = uniform over the children)."""
    if len(priors) < 2:
        return 0.0
    return -sum(p * log(p) for p in priors if p > 0) / log(len(priors))


def search_tree_puct(game: Game, root_moves: list[Move], simulations: int, *, c_puct: float, prior: str,
                     policy=None, candidate_limit: int, neighborhood_radius: int, priority_top_k: int,
                     random: Random | None = None, stats: PUCTStats | None = None):
    """Returns (chosen move, root children); children expose ``move``, ``visits``, ``mean_value``."""
    if prior not in PUCT_PRIORS:
        raise ValueError(f"prior must be one of {PUCT_PRIORS}")
    if prior == "policy" and policy is None:
        raise ValueError("the policy prior needs a policy callable")
    if not root_moves:
        raise ValueError("search_tree_puct needs at least one root move")
    random = random or Random()
    stats = stats if stats is not None else PUCTStats()
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
            winner = state.winner
        else:
            _expand(node, state, _search_candidates_v321(state, candidate_limit, neighborhood_radius),
                    prior, policy, stats)
            winner = _rollout_v321(state, random, candidate_limit, neighborhood_radius, priority_top_k)
        _backpropagate(node, winner)
        while len(state.history) > root_length:
            state.undo()
        stats.simulations += 1
    best = max(root.children, key=lambda c: (c.visits, c.mean_value, c.prior, (-c.move[0], -c.move[1])))
    return best.move, root.children
