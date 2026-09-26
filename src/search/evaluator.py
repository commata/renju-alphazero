"""Stage 5 evaluator boundary: immutable snapshots, result validation, fake evaluators.

This module must stay torch-free. The neural ``PolicyValueEvaluator`` lives in
``model.evaluator`` and satisfies the same ``Evaluator`` protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from model.config import ACTION_COUNT, BOARD_SIZE, coordinate_to_action

POLICY_SUM_TOLERANCE = 1e-5

Coordinate = tuple[int, int]


class EvaluatorOutputError(ValueError):
    """An evaluator returned a result that violates the Stage 5 search contract."""


@dataclass(frozen=True)
class EvaluationSnapshot:
    """Immutable evaluator input; never a reference to the live, mutating search Game."""

    board: tuple[tuple[int, ...], ...]      # 15x15, black 1 / white -1 / empty 0
    to_play: int
    last_move: Coordinate | None
    legal_moves: tuple[Coordinate, ...]     # computed once by search

    def __post_init__(self):
        if (len(self.board) != BOARD_SIZE
                or any(len(row) != BOARD_SIZE for row in self.board)):
            raise ValueError('snapshot board must be 15x15')
        if self.to_play not in (1, -1):
            raise ValueError('snapshot to_play must be 1 or -1')
        if not self.legal_moves:
            raise ValueError('snapshot must have at least one legal move')

    @classmethod
    def from_game(cls, game, legal_moves: Sequence[Coordinate]) -> EvaluationSnapshot:
        return cls(
            board=tuple(tuple(row) for row in game.board),
            to_play=game.to_play,
            last_move=tuple(game.history[-1]) if game.history else None,
            legal_moves=tuple((row, col) for row, col in legal_moves),
        )

    def legal_actions(self) -> tuple[int, ...]:
        return tuple(coordinate_to_action(row, col) for row, col in self.legal_moves)


@dataclass(frozen=True)
class EvaluationResult:
    priors: tuple[float, ...]   # len 225, probability over legal actions
    value: float                # snapshot.to_play perspective


class Evaluator(Protocol):
    def evaluate_batch(self, snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
        ...

    def evaluate(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        """Must use the same path as ``evaluate_batch([snapshot])[0]``."""
        ...


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_evaluation(result, snapshot: EvaluationSnapshot) -> EvaluationResult:
    """Reject (never repair) a result that breaks the search boundary contract."""
    if not isinstance(result, EvaluationResult):
        raise EvaluatorOutputError('evaluator must return EvaluationResult')
    priors = result.priors
    if not isinstance(priors, tuple) or len(priors) != ACTION_COUNT:
        raise EvaluatorOutputError(f'policy must be a tuple of length {ACTION_COUNT}')
    for prior in priors:
        if not _is_real(prior) or not isfinite(prior):
            raise EvaluatorOutputError('policy priors must be finite numbers')
        if prior < 0:
            raise EvaluatorOutputError('policy priors must be non-negative')
    legal = set(snapshot.legal_actions())
    if any(prior != 0 for action, prior in enumerate(priors) if action not in legal):
        raise EvaluatorOutputError('illegal action prior must be exactly 0')
    total = sum(priors[action] for action in legal)
    if abs(total - 1.0) > POLICY_SUM_TOLERANCE:
        raise EvaluatorOutputError(f'legal priors must sum to 1 (got {total!r})')
    value = result.value
    if not _is_real(value) or not isfinite(value):
        raise EvaluatorOutputError('value must be a finite number')
    if not -1.0 <= value <= 1.0:
        raise EvaluatorOutputError('value must be in [-1, 1]')
    return result


def evaluate_validated(evaluator: Evaluator,
                       snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
    results = evaluator.evaluate_batch(snapshots)
    if not isinstance(results, list) or len(results) != len(snapshots):
        raise EvaluatorOutputError('evaluate_batch must return one result per snapshot')
    return [validate_evaluation(result, snapshot)
            for result, snapshot in zip(results, snapshots)]


class UniformEvaluator:
    """Uniform prior over the supplied legal moves, value 0."""

    def __init__(self):
        self.calls = 0

    def evaluate_batch(self, snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
        results = []
        for snapshot in snapshots:
            self.calls += 1
            priors = [0.0] * ACTION_COUNT
            actions = snapshot.legal_actions()
            for action in actions:
                priors[action] = 1.0 / len(actions)
            results.append(EvaluationResult(tuple(priors), 0.0))
        return results

    def evaluate(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        return self.evaluate_batch([snapshot])[0]


ScriptOutput = EvaluationResult | tuple[Sequence[float], float] | None


class ScriptedEvaluator:
    """Deterministic test evaluator with call counting.

    Default behaviour: legal action ``a`` gets weight ``weights.get(a, default_weight)``,
    normalized over the snapshot's legal actions; value is ``value`` (a float, or a
    callable of the snapshot). ``script(snapshot)`` may override a snapshot by returning
    an ``EvaluationResult`` or a raw ``(priors, value)`` pair, which is passed through
    unmodified so invalid outputs can be exercised; ``None`` falls back to the default.
    """

    def __init__(self, *, weights: Mapping[int, float] | None = None,
                 value: float | Callable[[EvaluationSnapshot], float] = 0.0,
                 default_weight: float = 1.0,
                 script: Callable[[EvaluationSnapshot], ScriptOutput] | None = None):
        self.weights = dict(weights or {})
        self.value = value
        self.default_weight = default_weight
        self.script = script
        self.calls = 0
        self.batch_calls = 0
        self.snapshots: list[EvaluationSnapshot] = []

    def _default(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        actions = snapshot.legal_actions()
        raw = {action: float(self.weights.get(action, self.default_weight)) for action in actions}
        total = sum(raw.values())
        priors = [0.0] * ACTION_COUNT
        for action, weight in raw.items():
            priors[action] = weight / total
        value = self.value(snapshot) if callable(self.value) else self.value
        return EvaluationResult(tuple(priors), float(value))

    def evaluate_batch(self, snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
        self.batch_calls += 1
        results = []
        for snapshot in snapshots:
            self.calls += 1
            self.snapshots.append(snapshot)
            output = self.script(snapshot) if self.script is not None else None
            if output is None:
                output = self._default(snapshot)
            elif not isinstance(output, EvaluationResult):
                priors, value = output
                output = EvaluationResult(tuple(priors), value)
            results.append(output)
        return results

    def evaluate(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        return self.evaluate_batch([snapshot])[0]
