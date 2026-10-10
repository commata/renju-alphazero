"""Track A GPU plan: SearchSession steps and the batching scheduler (torch-free)."""
from random import Random
import unittest

from search.alphazero import SearchConfig, run_search, search_steps
from search.evaluator import (EvalRequest, EvaluationResult, EvaluatorOutputError,
                              ScriptedEvaluator, UniformEvaluator, drive_requests, drive_steps,
                              tag_requests)
from search.scheduler import SchedulerStats, run_tasks
from training.self_play import (play_self_play_game, play_self_play_games, record_hash,
                                 self_play_game_steps)
from renju import Game


def board_value(snapshot):
    """Deterministic, position-dependent value so different games diverge."""
    score = sum((r * 15 + c) * cell for r, row in enumerate(snapshot.board)
                for c, cell in enumerate(row))
    return ((score % 17) - 8) / 10.0


def scripted():
    return ScriptedEvaluator(weights={112: 3.0, 96: 2.0, 128: 2.0}, value=board_value)


CONFIG = SearchConfig(num_simulations=6, temperature_moves=4, tactical_rules=True)
SEEDS = [11, 12, 13, 14]


class AsyncEvaluator:
    """submit/collect stub whose batches finish only after a few polls."""

    def __init__(self, inner, polls=2):
        self.inner = inner
        self.polls = polls
        self.outstanding = 0
        self.max_outstanding = 0

    class Handle:
        def __init__(self, results, polls):
            self.results = results
            self.left = polls

        def done(self):
            self.left -= 1
            return self.left < 0

    def submit(self, snapshots):
        self.outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self.outstanding)
        return self.Handle(self.inner.evaluate_batch(snapshots), self.polls)

    def collect(self, handle):
        self.outstanding -= 1
        return handle.results


class SearchStepsTest(unittest.TestCase):
    def test_driver_equals_run_search(self):
        game = Game()
        for move in ((7, 7), (7, 8), (8, 8)):
            game.play(*move)
        a = run_search(game, scripted(), CONFIG, Random(3))
        b, _ = drive_steps(search_steps(game, CONFIG, Random(3)), scripted())
        self.assertEqual((a.visit_counts, a.priors, a.evaluator_calls),
                         (b.visit_counts, b.priors, b.evaluator_calls))

    def test_fast_path_yields_nothing(self):
        game = Game()
        steps = search_steps(game, SearchConfig(noise_enabled=False, temperature_moves=0))
        with self.assertRaises(StopIteration) as stop:
            next(steps)
        self.assertTrue(stop.exception.value[0].fast_path)

    def test_one_outstanding_request(self):
        game = Game()
        game.play(7, 7)
        steps = tag_requests(search_steps(game, SearchConfig(noise_enabled=False, temperature_moves=0,
                                                              num_simulations=3)), 'key')
        request = next(steps)
        self.assertIsInstance(request, EvalRequest)
        self.assertEqual(request.evaluator, 'key')


class SchedulerTest(unittest.TestCase):
    def serial_hashes(self):
        return [record_hash(g.record) for g in
                [play_self_play_game(scripted(), CONFIG, seed) for seed in SEEDS]]

    def test_batched_self_play_equals_serial(self):
        expected = self.serial_hashes()
        for kwargs in ({}, {'max_batch': 2}, {'max_batch': 1}, {'eager': True}):
            with self.subTest(**kwargs):
                stats = SchedulerStats()
                games = play_self_play_games(scripted(), CONFIG, SEEDS, batched=True,
                                             stats=stats, **kwargs)
                self.assertEqual([record_hash(g.record) for g in games], expected)
                summary = stats.summary()
                self.assertEqual(summary['tasks'], len(SEEDS))
                self.assertLessEqual(summary['max_batch'], kwargs.get('max_batch', len(SEEDS)))
                self.assertEqual(sum(int(k) * v for k, v in summary['histogram'].items()),
                                 summary['requests'])

    def test_lockstep_batches_fill_to_active_games(self):
        stats = SchedulerStats()
        play_self_play_games(scripted(), CONFIG, SEEDS, batched=True, stats=stats)
        self.assertEqual(stats.batch_sizes[0], len(SEEDS))

    def test_async_evaluator_same_games(self):
        expected = self.serial_hashes()
        for eager in (False, True):
            with self.subTest(eager=eager):
                evaluator = AsyncEvaluator(scripted())
                games = play_self_play_games(evaluator, CONFIG, SEEDS, batched=True, eager=eager)
                self.assertEqual([record_hash(g.record) for g in games], expected)
                self.assertEqual(evaluator.max_outstanding, 1)
                self.assertEqual(evaluator.outstanding, 0)

    def test_refill_keeps_order_and_bound(self):
        expected = self.serial_hashes()
        stats = SchedulerStats()
        evaluator = scripted()
        tasks = [lambda s=s: tag_requests(self_play_game_steps(CONFIG, s), evaluator)
                 for s in SEEDS]
        games = run_tasks(tasks, max_active=2, stats=stats)
        self.assertEqual([record_hash(g.record) for g in games], expected)
        self.assertLessEqual(max(stats.active_at_submit), 2)

    def test_two_evaluators_batch_separately(self):
        first, second = UniformEvaluator(), scripted()

        game = Game()
        game.play(7, 7)

        def searched(evaluator, seed):
            return tag_requests(search_steps(game, CONFIG, Random(seed)), evaluator)

        tasks = [lambda: searched(first, 1), lambda: searched(second, 2), lambda: searched(first, 3)]
        batched = run_tasks(tasks)
        serial = [drive_requests(t()) for t in tasks]
        self.assertEqual([r[0].visit_counts for r in batched], [r[0].visit_counts for r in serial])

    def test_bad_evaluator_output_is_rejected(self):
        class Short:
            def evaluate_batch(self, snapshots):
                return [EvaluationResult(tuple([0.0] * 225), 0.0)]

        with self.assertRaises(EvaluatorOutputError):
            play_self_play_games(Short(), CONFIG, SEEDS[:2], batched=True)

    def test_tasks_must_yield_requests(self):
        def bad():
            yield 'not a request'

        with self.assertRaises(TypeError):
            run_tasks([bad])

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            run_tasks([], max_active=0)
        with self.assertRaises(ValueError):
            run_tasks([], max_batch=0)
        self.assertEqual(run_tasks([]), [])


if __name__ == '__main__':
    unittest.main()
