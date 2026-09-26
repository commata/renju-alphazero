import math
import unittest

from renju import BLACK, WHITE, Game
from search.evaluator import (EvaluationResult, EvaluationSnapshot, EvaluatorOutputError,
                              ScriptedEvaluator, UniformEvaluator, evaluate_validated,
                              validate_evaluation)


def opened_game():
    game = Game()
    for move in ((7, 7), (7, 8), (8, 8)):
        game.play(*move)
    return game


def snapshot_of(game):
    return EvaluationSnapshot.from_game(game, game.legal_moves())


class FakeEvaluatorTest(unittest.TestCase):
    def test_uniform_prior_legal_only(self):
        game = opened_game()
        snapshot = snapshot_of(game)
        result = validate_evaluation(UniformEvaluator().evaluate(snapshot), snapshot)
        legal = set(snapshot.legal_actions())
        self.assertEqual(len(result.priors), 225)
        for action, prior in enumerate(result.priors):
            if action in legal:
                self.assertAlmostEqual(prior, 1 / len(legal))
            else:
                self.assertEqual(prior, 0)
        self.assertAlmostEqual(sum(result.priors), 1.0)
        self.assertEqual(result.value, 0.0)

    def test_uniform_initial_position_center_only(self):
        snapshot = snapshot_of(Game())
        result = UniformEvaluator().evaluate(snapshot)
        self.assertEqual(result.priors[112], 1.0)
        self.assertEqual(sum(result.priors), 1.0)

    def test_scripted_deterministic_and_counts_calls(self):
        game = opened_game()
        snapshot = snapshot_of(game)
        evaluator = ScriptedEvaluator(weights={0: 3.0}, default_weight=1.0, value=0.25)
        first = evaluator.evaluate(snapshot)
        second = evaluator.evaluate_batch([snapshot, snapshot])
        self.assertEqual(evaluator.calls, 3)
        self.assertEqual(evaluator.batch_calls, 2)
        self.assertEqual(first, second[0])
        self.assertEqual(first, second[1])
        n = len(snapshot.legal_moves)
        self.assertAlmostEqual(first.priors[0], 3 / (n + 2))
        self.assertAlmostEqual(first.priors[1], 1 / (n + 2))
        self.assertEqual(first.value, 0.25)
        validate_evaluation(first, snapshot)

    def test_evaluate_uses_batch_path(self):
        snapshot = snapshot_of(opened_game())
        evaluator = ScriptedEvaluator()
        evaluator.evaluate(snapshot)
        self.assertEqual((evaluator.batch_calls, evaluator.calls), (1, 1))

    def test_batch_length_mismatch_rejected(self):
        class Short(UniformEvaluator):
            def evaluate_batch(self, snapshots):
                return []
        with self.assertRaises(EvaluatorOutputError):
            evaluate_validated(Short(), [snapshot_of(opened_game())])


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.game = opened_game()
        self.snapshot = snapshot_of(self.game)
        self.legal = self.snapshot.legal_actions()
        self.illegal = next(a for a in range(225) if a not in set(self.legal))

    def uniform(self):
        priors = [0.0] * 225
        for action in self.legal:
            priors[action] = 1 / len(self.legal)
        return priors

    def assert_rejected(self, priors, value=0.0):
        with self.assertRaises(EvaluatorOutputError):
            validate_evaluation(EvaluationResult(tuple(priors), value), self.snapshot)

    def test_valid_passes(self):
        validate_evaluation(EvaluationResult(tuple(self.uniform()), -1.0), self.snapshot)
        validate_evaluation(EvaluationResult(tuple(self.uniform()), 1), self.snapshot)

    def test_wrong_length(self):
        self.assert_rejected(self.uniform()[:-1])
        self.assert_rejected(self.uniform() + [0.0])

    def test_non_tuple_or_wrong_type(self):
        with self.assertRaises(EvaluatorOutputError):
            validate_evaluation(EvaluationResult(self.uniform(), 0.0), self.snapshot)
        with self.assertRaises(EvaluatorOutputError):
            validate_evaluation((tuple(self.uniform()), 0.0), self.snapshot)

    def test_nan_inf(self):
        for bad in (math.nan, math.inf, -math.inf):
            priors = self.uniform()
            priors[self.legal[0]] = bad
            self.assert_rejected(priors)

    def test_negative_prior(self):
        priors = self.uniform()
        priors[self.legal[0]] += 0.5
        priors[self.legal[1]] = -0.5 + priors[self.legal[1]]
        self.assert_rejected(priors)

    def test_illegal_prior(self):
        priors = self.uniform()
        priors[self.legal[0]] -= 1e-3
        priors[self.illegal] = 1e-3
        self.assert_rejected(priors)
        stone = 7 * 15 + 7
        priors = self.uniform()
        priors[stone] = 1e-9
        self.assert_rejected(priors)

    def test_sum_not_one(self):
        priors = [p * 1.01 for p in self.uniform()]
        self.assert_rejected(priors)
        priors = [p * (1 - 1e-6) for p in self.uniform()]
        validate_evaluation(EvaluationResult(tuple(priors), 0.0), self.snapshot)

    def test_invalid_value(self):
        for value in (1.0001, -1.5, math.nan, math.inf, True, None, '0'):
            self.assert_rejected(self.uniform(), value)

    def test_does_not_normalize(self):
        priors = tuple(p * 2 for p in self.uniform())
        result = EvaluationResult(priors, 0.0)
        with self.assertRaises(EvaluatorOutputError):
            validate_evaluation(result, self.snapshot)
        self.assertEqual(result.priors, priors)


class SnapshotTest(unittest.TestCase):
    def test_independent_from_live_game(self):
        game = opened_game()
        legal = game.legal_moves()
        snapshot = EvaluationSnapshot.from_game(game, legal)
        board_before = snapshot.board
        copy_board = [list(row) for row in board_before]
        game.play(6, 6)
        game.play(9, 9)
        game.undo()
        legal.clear()
        self.assertEqual([list(row) for row in snapshot.board], copy_board)
        self.assertEqual(snapshot.to_play, WHITE)
        self.assertEqual(snapshot.last_move, (8, 8))
        self.assertTrue(snapshot.legal_moves)
        self.assertEqual(game.board[6][6], WHITE)
        with self.assertRaises(Exception):
            snapshot.board = ()

    def test_no_legal_moves_method(self):
        snapshot = snapshot_of(opened_game())
        self.assertFalse(hasattr(snapshot, 'legal_moves') and callable(snapshot.legal_moves))

    def test_rejects_bad_snapshot(self):
        game = opened_game()
        with self.assertRaises(ValueError):
            EvaluationSnapshot.from_game(game, [])
        with self.assertRaises(ValueError):
            EvaluationSnapshot(((0,) * 15,) * 14, BLACK, None, ((7, 7),))
        with self.assertRaises(ValueError):
            EvaluationSnapshot(((0,) * 15,) * 15, 0, None, ((7, 7),))


if __name__ == '__main__':
    unittest.main()
