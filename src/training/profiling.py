"""Stage 7 PUCT cost breakdown (plan §4) without changing search results.

``SearchTiming.inference_s`` mixes snapshot construction, the evaluator and result
validation; ``tree_s`` mixes rule filtering, play/undo and selection/backup. This
module splits them:

- ``ProfiledEvaluator`` reproduces ``PolicyValueEvaluator.evaluate_batch`` exactly
  (same ops, same order) while timing encode (legal mask + planes), stack, NN forward,
  masked softmax, CPU transfer (``.cpu().tolist()``) and result construction;
- ``instrument_search()`` temporarily wraps ``search.alphazero._legal_moves`` (split by
  side to move), ``search.alphazero.tactical_filter`` and ``Game.play``/``Game.undo``.

Derived per searched move: snapshot+validation = inference_s - evaluator time, and
tree other (selection, backup, node creation) = tree_s - rule filter - play - undo.
Wrapper overhead is small but nonzero; compare configurations measured the same way.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import torch

from model.encoding import encode_game
from model.evaluator import PolicyValueEvaluator, _SnapshotView
from model.masking import legal_moves_to_mask, masked_softmax
from renju import BLACK, Game
from search.evaluator import EvaluationResult

EVALUATOR_PARTS = ('encode', 'stack', 'forward', 'softmax', 'transfer', 'results')


class ProfiledEvaluator(PolicyValueEvaluator):
    """``PolicyValueEvaluator`` with per-stage timers; outputs are identical."""

    def __init__(self, model, *, device='cpu'):
        super().__init__(model, device=device)
        self.seconds = defaultdict(float)
        self.batches = 0

    def evaluate_batch(self, snapshots):
        if not snapshots:
            return []
        if self.model.training:
            raise RuntimeError('PolicyValueEvaluator requires model.eval()')
        t0 = perf_counter()
        encoded = []
        for snapshot in snapshots:
            mask = legal_moves_to_mask(snapshot.legal_moves, device=self.device)
            encoded.append((encode_game(_SnapshotView(snapshot), mask), mask))
        t1 = perf_counter()
        planes = torch.stack([x for x, _ in encoded])
        masks = torch.stack([m for _, m in encoded])
        t2 = perf_counter()
        with torch.inference_mode():
            logits, values = self.model(planes)
            t3 = perf_counter()
            priors = masked_softmax(logits, masks)
        t4 = perf_counter()
        self.calls += len(snapshots)
        prior_rows = priors.cpu().tolist()
        value_rows = values.reshape(-1).cpu().tolist()
        t5 = perf_counter()
        results = [EvaluationResult(tuple(row), value)
                   for row, value in zip(prior_rows, value_rows)]
        t6 = perf_counter()
        for name, seconds in zip(EVALUATOR_PARTS, (t1 - t0, t2 - t1, t3 - t2, t4 - t3,
                                                   t5 - t4, t6 - t5)):
            self.seconds[name] += seconds
        self.batches += 1
        return results


@contextmanager
def instrument_search():
    """Time legal_moves (by side to move), the rule filter and Game.play/undo."""
    import search.alphazero as alphazero

    seconds = defaultdict(float)
    counts = defaultdict(int)
    original_legal = alphazero._legal_moves
    original_filter = alphazero.tactical_filter
    original_play, original_undo = Game.play, Game.undo

    def legal_moves(game, timing):
        key = 'legal_moves_black' if game.to_play == BLACK else 'legal_moves_white'
        started = perf_counter()
        try:
            return original_legal(game, timing)
        finally:
            seconds[key] += perf_counter() - started
            counts[key] += 1

    def tactical_filter(game, legal):
        started = perf_counter()
        try:
            return original_filter(game, legal)
        finally:
            seconds['rule_filter'] += perf_counter() - started
            counts['rule_filter'] += 1

    def play(self, row, col):
        started = perf_counter()
        try:
            return original_play(self, row, col)
        finally:
            seconds['play'] += perf_counter() - started
            counts['play'] += 1

    def undo(self):
        started = perf_counter()
        try:
            return original_undo(self)
        finally:
            seconds['undo'] += perf_counter() - started
            counts['undo'] += 1

    alphazero._legal_moves = legal_moves
    alphazero.tactical_filter = tactical_filter
    Game.play, Game.undo = play, undo
    try:
        yield seconds, counts
    finally:
        alphazero._legal_moves = original_legal
        alphazero.tactical_filter = original_filter
        Game.play, Game.undo = original_play, original_undo


def profile_searches(model, games, search_config) -> dict:
    """Search every game once; return per-move mean milliseconds per component."""
    from search.alphazero import run_search

    evaluator = ProfiledEvaluator(model)
    timing_totals = defaultdict(float)
    searched = 0
    calls = 0
    with instrument_search() as (seconds, counts):
        for game in games:
            result = run_search(game, evaluator, search_config, None)
            if result.fast_path:
                continue
            searched += 1
            calls += result.evaluator_calls
            for name in ('legal_moves_s', 'inference_s', 'tree_s', 'total_s'):
                timing_totals[name] += getattr(result.timing, name)
    if not searched:
        raise ValueError('no searched (non fast-path) positions')
    evaluator_total = sum(evaluator.seconds.values())
    parts = {
        'legal_moves_black': seconds['legal_moves_black'],
        'legal_moves_white': seconds['legal_moves_white'],
        'snapshot_and_validation': timing_totals['inference_s'] - evaluator_total,
        **{f'nn_{name}': evaluator.seconds[name] for name in EVALUATOR_PARTS},
        'rule_filter': seconds['rule_filter'],
        'play': seconds['play'],
        'undo': seconds['undo'],
    }
    parts['tree_other'] = (timing_totals['tree_s'] - seconds['rule_filter']
                           - seconds['play'] - seconds['undo'])
    per_move_ms = {name: 1000 * value / searched for name, value in parts.items()}
    total_ms = 1000 * timing_totals['total_s'] / searched
    return {
        'searched_positions': searched,
        'search': search_config.to_dict(),
        'evaluator_calls_per_move': calls / searched,
        'total_ms_per_move': total_ms,
        'per_move_ms': per_move_ms,
        'share': {name: value / total_ms for name, value in per_move_ms.items()},
        'per_call_us': {
            'nn_total': 1e6 * evaluator_total / calls,
            **{f'nn_{n}': 1e6 * evaluator.seconds[n] / calls for n in EVALUATOR_PARTS},
        },
        'counts_per_move': {name: counts[name] / searched for name in counts},
        'stage5_timing_ms_per_move': {k: 1000 * v / searched for k, v in timing_totals.items()},
    }
