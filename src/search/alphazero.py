"""Stage 5 AlphaZero PUCT search (independent of the frozen Stage 3 MCTS-v6).

Shares only ``Game``/Renju legality and the torch-free Stage 4 action contract.
No rollouts, forced moves, tactical planners, candidate injection or widening:
children are exactly ``Game.legal_moves()`` with evaluator priors.

Statistics convention: a child's ``value_sum`` is stored from the perspective of
``player_who_moved`` (the parent's ``to_play``), so selection maximizes child Q.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, sqrt
from random import Random
from time import perf_counter

from model.config import ACTION_COUNT, action_to_coordinate, coordinate_to_action

from .evaluator import POLICY_SUM_TOLERANCE, EvaluationSnapshot, Evaluator, evaluate_validated

SEARCH_CONFIG_FORMAT = 'stage5-search-config-v1'


@dataclass(frozen=True)
class SearchConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    fpu: float = 0.0
    tau: float = 1.0
    temperature_moves: int = 10
    dirichlet_alpha: float = 0.05
    dirichlet_epsilon: float = 0.25
    noise_enabled: bool = True
    evaluator_batch_size: int = 1

    def __post_init__(self):
        def positive_int(name):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        def real(name):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
                raise ValueError(f'{name} must be a finite number')
            return value

        positive_int('num_simulations')
        if real('c_puct') <= 0:
            raise ValueError('c_puct must be positive')
        if real('fpu') != 0:
            raise ValueError('Stage 5 fixes FPU to 0')
        if real('tau') <= 0:
            raise ValueError('tau must be positive')
        if type(self.temperature_moves) is not int or self.temperature_moves < 0:
            raise ValueError('temperature_moves must be a non-negative integer')
        if real('dirichlet_alpha') <= 0:
            raise ValueError('dirichlet_alpha must be positive')
        if not 0 <= real('dirichlet_epsilon') <= 1:
            raise ValueError('dirichlet_epsilon must be in [0, 1]')
        if type(self.noise_enabled) is not bool:
            raise ValueError('noise_enabled must be a bool')
        positive_int('evaluator_batch_size')
        if self.evaluator_batch_size != 1:
            # Sequential PUCT produces one leaf at a time; no virtual loss in Stage 5.
            raise ValueError('Stage 5 search evaluates leaves with batch size 1')

    def to_dict(self) -> dict:
        """The exact field set that is hashed into ``config_hash``."""
        return {
            'format_version': SEARCH_CONFIG_FORMAT,
            'num_simulations': self.num_simulations,
            'c_puct': float(self.c_puct),
            'fpu': float(self.fpu),
            'tau': float(self.tau),
            'temperature_moves': self.temperature_moves,
            'dirichlet_alpha': float(self.dirichlet_alpha),
            'dirichlet_epsilon': float(self.dirichlet_epsilon),
            'noise_enabled': self.noise_enabled,
            'evaluator_batch_size': self.evaluator_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SearchConfig:
        data = dict(data)
        if data.pop('format_version', None) != SEARCH_CONFIG_FORMAT:
            raise ValueError('unsupported search config format')
        return cls(**data)


class Node:
    __slots__ = ('action', 'prior', 'visit_count', 'value_sum', 'player_who_moved', 'to_play',
                 'children', 'is_expanded', 'is_terminal', 'terminal_value')

    def __init__(self, action: int | None, prior: float, player_who_moved: int | None):
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0            # player_who_moved perspective
        self.player_who_moved = player_who_moved
        self.to_play: int | None = None  # recorded when the state is first reached
        self.children: dict[int, Node] = {}
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value: float | None = None  # to_play perspective when terminal

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0  # FPU = 0


@dataclass
class SearchTiming:
    legal_moves_s: float = 0.0
    inference_s: float = 0.0
    tree_s: float = 0.0
    total_s: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    visit_counts: tuple[int, ...]   # len 225; canonical policy target source
    priors: tuple[float, ...]       # root priors actually used (after noise when enabled)
    to_play: int
    fast_path: bool                 # single-legal root: no search, synthetic count 1
    evaluator_calls: int
    timing: SearchTiming


def terminal_value(game, perspective: int) -> float:
    """Terminal value by player identity (a winning move keeps to_play == winner)."""
    if not game.done:
        raise ValueError('game is not terminal')
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner == perspective else -1.0


def puct_score(parent: Node, child: Node, c_puct: float) -> float:
    return child.q + c_puct * child.prior * sqrt(max(1, parent.visit_count)) / (1 + child.visit_count)


def select_child(node: Node, c_puct: float) -> Node:
    """Highest PUCT; ties by larger prior, then smaller action index."""
    return max(node.children.values(),
               key=lambda child: (puct_score(node, child, c_puct), child.prior, -child.action))


def backup(root: Node, path: list[Node], leaf_value: float, leaf_player: int) -> None:
    root.visit_count += 1
    for child in reversed(path):
        child.visit_count += 1
        child.value_sum += leaf_value if child.player_who_moved == leaf_player else -leaf_value


def sample_dirichlet(rng: Random, alpha: float, count: int) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(count)]
    total = sum(samples)
    if not all(isfinite(s) for s in samples) or not isfinite(total) or total <= 0:
        raise ValueError('Dirichlet gamma samples must be finite with a positive total')
    return [sample / total for sample in samples]


def apply_root_noise(root: Node, rng: Random, alpha: float, epsilon: float) -> None:
    """Mix Dirichlet noise into legal root children only (ascending action order)."""
    actions = sorted(root.children)
    eta = sample_dirichlet(rng, alpha, len(actions))
    for action, noise in zip(actions, eta):
        child = root.children[action]
        child.prior = (1 - epsilon) * child.prior + epsilon * noise
    total = sum(root.children[a].prior for a in actions)
    if abs(total - 1.0) > POLICY_SUM_TOLERANCE:
        raise ValueError(f'noised root priors must sum to 1 (got {total!r})')


def _expand(node: Node, game, legal_moves, evaluator: Evaluator, timing: SearchTiming) -> float:
    """Create children for the supplied legal moves; return value from node.to_play view."""
    started = perf_counter()
    snapshot = EvaluationSnapshot.from_game(game, legal_moves)
    result = evaluate_validated(evaluator, [snapshot])[0]
    timing.inference_s += perf_counter() - started
    player = game.to_play
    for row, col in legal_moves:
        action = coordinate_to_action(row, col)
        node.children[action] = Node(action, float(result.priors[action]), player)
    node.is_expanded = True
    return float(result.value)


def _legal_moves(game, timing: SearchTiming):
    started = perf_counter()
    moves = game.legal_moves()
    timing.legal_moves_s += perf_counter() - started
    if not moves:
        raise RuntimeError('Game invariant violated: done=False but no legal moves')
    return moves


def run_search(game, evaluator: Evaluator, config: SearchConfig,
               rng: Random | None = None) -> SearchResult:
    return search_with_tree(game, evaluator, config, rng)[0]


def search_with_tree(game, evaluator: Evaluator, config: SearchConfig,
                     rng: Random | None = None) -> tuple[SearchResult, Node | None]:
    """Search one root and also return the tree (None on the fast path).

    ``game`` is never mutated; a single working copy is used.

    Root expansion is not a simulation and its value is not backed up; exactly
    ``config.num_simulations`` descents follow. Root Dirichlet noise is applied iff
    ``config.noise_enabled`` (self-play), consuming ``rng`` only for searched roots.
    """
    started = perf_counter()
    if game.done:
        raise ValueError('cannot search a finished game')
    if config.noise_enabled and rng is None:
        raise ValueError('root noise requires a caller-owned random.Random')
    timing = SearchTiming()
    legal_moves = _legal_moves(game, timing)
    to_play = game.to_play

    if len(legal_moves) == 1:
        action = coordinate_to_action(*legal_moves[0])
        one_hot = [0] * ACTION_COUNT
        one_hot[action] = 1
        timing.total_s = perf_counter() - started
        timing.tree_s = timing.total_s - timing.legal_moves_s
        return SearchResult(tuple(one_hot), tuple(float(c) for c in one_hot), to_play,
                            True, 0, timing), None

    work = deepcopy(game)
    root_length = len(work.history)
    root = Node(None, 1.0, None)
    root.to_play = to_play
    evaluator_calls = 1
    _expand(root, work, legal_moves, evaluator, timing)
    if config.noise_enabled:
        apply_root_noise(root, rng, config.dirichlet_alpha, config.dirichlet_epsilon)

    for _ in range(config.num_simulations):
        node = root
        path: list[Node] = []
        played = 0
        while node.is_expanded and not node.is_terminal:
            child = select_child(node, config.c_puct)
            path.append(child)
            node = child
            if child.is_terminal:
                break  # cached terminal: no play/legal_moves/evaluator
            work.play(*action_to_coordinate(child.action))
            played += 1
            if child.to_play is None:
                child.to_play = work.to_play
                if work.done:
                    child.is_terminal = True
                    child.is_expanded = True
                    child.terminal_value = terminal_value(work, work.to_play)
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = _expand(node, work, _legal_moves(work, timing), evaluator, timing)
            evaluator_calls += 1
        backup(root, path, value, node.to_play)
        for _ in range(played):
            work.undo()
        if len(work.history) != root_length:
            raise RuntimeError('working game was not restored to the root state')

    child_visits = sum(child.visit_count for child in root.children.values())
    if root.visit_count != config.num_simulations or child_visits != config.num_simulations:
        raise RuntimeError('root visit counts do not match num_simulations')
    counts = [0] * ACTION_COUNT
    priors = [0.0] * ACTION_COUNT
    for action, child in root.children.items():
        counts[action] = child.visit_count
        priors[action] = child.prior
    timing.total_s = perf_counter() - started
    timing.tree_s = timing.total_s - timing.legal_moves_s - timing.inference_s
    return SearchResult(tuple(counts), tuple(priors), to_play, False, evaluator_calls, timing), root


def visit_policy(visit_counts) -> tuple[float, ...]:
    """Derived training target pi = N / sum(N); never the canonical stored value."""
    total = sum(visit_counts)
    if total <= 0:
        raise ValueError('visit counts must have a positive total')
    return tuple(count / total for count in visit_counts)


def argmax_action(result: SearchResult) -> int:
    """Visit count, then the prior actually used at the root, then smaller action index."""
    candidates = [a for a, count in enumerate(result.visit_counts) if count > 0]
    return max(candidates, key=lambda a: (result.visit_counts[a], result.priors[a], -a))


def sample_action(visit_counts, tau: float, rng: Random) -> int:
    """One ``rng.random()`` draw; prob ∝ N^(1/tau) accumulated in ascending action order."""
    actions = [a for a, count in enumerate(visit_counts) if count > 0]
    peak = max(visit_counts[a] for a in actions)
    weights = [(visit_counts[a] / peak) ** (1.0 / tau) for a in actions]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight
        if threshold < cumulative:
            return action
    return actions[-1]  # floating-point edge: threshold == total


def select_action(result: SearchResult, ply: int, config: SearchConfig,
                  rng: Random | None) -> int:
    """Choose the move to play; never changes the stored visit counts.

    ``ply`` is ``len(game.history)`` before the move (0-based). Temperature sampling
    applies while ``ply < temperature_moves``; the single-legal fast path uses no RNG.
    """
    if result.fast_path:
        return result.visit_counts.index(1)
    if ply < config.temperature_moves:
        if rng is None:
            raise ValueError('temperature sampling requires a caller-owned random.Random')
        return sample_action(result.visit_counts, config.tau, rng)
    return argmax_action(result)
