from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from agents import MCTSV321Agent
from renju import BLACK, WHITE, Game
from search.mcts_v321 import (
    _fast_pattern_features_for_move,
    _v321_priority_score,
)


class MCTSV321OptimizationTest(unittest.TestCase):
    def test_defaults(self):
        agent = MCTSV321Agent()
        self.assertEqual(agent.simulations, 25)
        self.assertEqual(agent.candidate_limit, 16)
        self.assertEqual(agent.initial_width, 6)
        self.assertEqual(agent.neighborhood_radius, 2)
        self.assertEqual(agent.priority_top_k, 5)

    def test_fast_black_four_three_is_detected_without_recursive_legality(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK

        with patch(
            "search.mcts_v321.forbidden_reason",
            side_effect=AssertionError("unexpected recursive legality check"),
        ):
            features = _fast_pattern_features_for_move(
                game,
                BLACK,
                (7, 7),
                assume_legal=True,
            )

        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.four_directions, 1)
        self.assertGreaterEqual(features.open_three_directions, 1)
        self.assertTrue(features.has_four_three)

    def test_fast_white_double_four_is_detected(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (5, 7), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = WHITE
        game.to_play = WHITE

        features = _fast_pattern_features_for_move(
            game,
            WHITE,
            (7, 7),
            assume_legal=True,
        )
        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.four_directions, 2)

    def test_fast_white_double_three_is_detected(self):
        game = Game()
        for move in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = WHITE
        game.to_play = WHITE

        features = _fast_pattern_features_for_move(
            game,
            WHITE,
            (7, 7),
            assume_legal=True,
        )
        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.open_three_directions, 2)

    def test_exact_top_level_black_forbidden_is_still_rejected(self):
        game = Game()
        for move in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK

        features = _fast_pattern_features_for_move(
            game,
            BLACK,
            (7, 7),
        )
        self.assertFalse(features.legal)

    def test_white_overline_is_immediate_win(self):
        game = Game()
        for col in range(2, 7):
            game.board[7][col] = WHITE

        features = _fast_pattern_features_for_move(
            game,
            WHITE,
            (7, 7),
            assume_legal=True,
        )
        self.assertTrue(features.immediate_win)
        self.assertEqual(features.max_run, 6)

    def test_black_four_three_scores_above_quiet_move(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK

        self.assertGreater(
            _v321_priority_score(game, (7, 7)),
            _v321_priority_score(game, (0, 0)),
        )

    def test_search_is_seeded_legal_and_preserves_state(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()

        first = MCTSV321Agent(seed=123, simulations=1).select_move(game)
        second = MCTSV321Agent(seed=123, simulations=1).select_move(game)

        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)


if __name__ == "__main__":
    unittest.main()
