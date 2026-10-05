"""Lock-step batching of independent searches (Stage 8 G0, docs/stage8-plan.md §12.20).

Each search or self-play game is a generator (``search.alphazero.search_steps``,
``training.self_play.self_play_steps``) with exactly one leaf snapshot outstanding.
``run_lockstep`` advances all of them together: one ``evaluate_batch`` call per round
evaluates the current leaf of every unfinished generator. One leaf per game means no
virtual loss and no change to any tree: the results are those of running the
generators one after another with the same evaluator outputs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .evaluator import Evaluator, evaluate_validated


@dataclass
class LockstepStats:
    rounds: int = 0
    evaluations: int = 0
    batch_sizes: Counter = field(default_factory=Counter)

    @property
    def mean_batch(self) -> float:
        return self.evaluations / self.rounds if self.rounds else 0.0

    def to_dict(self) -> dict:
        return {'rounds': self.rounds, 'evaluations': self.evaluations,
                'mean_batch': self.mean_batch,
                'batch_sizes': {str(k): v for k, v in sorted(self.batch_sizes.items())}}


def run_lockstep(generators: Sequence, evaluator: Evaluator,
                 stats: LockstepStats | None = None) -> list:
    """Run the generators to completion; return their return values in input order."""
    results = [None] * len(generators)
    pending = {}
    for index, steps in enumerate(generators):
        try:
            pending[index] = next(steps)
        except StopIteration as stop:          # e.g. a single-legal fast path
            results[index] = stop.value
    while pending:
        order = sorted(pending)
        evaluations = evaluate_validated(evaluator, [pending[i] for i in order])
        if stats is not None:
            stats.rounds += 1
            stats.evaluations += len(order)
            stats.batch_sizes[len(order)] += 1
        for index, evaluation in zip(order, evaluations):
            try:
                pending[index] = generators[index].send(evaluation)
            except StopIteration as stop:
                results[index] = stop.value
                del pending[index]
    return results
