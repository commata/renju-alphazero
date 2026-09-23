from copy import deepcopy
import random
import unittest

from agents import MCTSV32Agent
from renju import BLACK, WHITE, Game
from search.mcts_v32 import (
    _pattern_features_for_move,
    _v32_priority_score,
)


class MCTSV32PolicyTest(unittest.TestCase):
    def test_defaults(self):
        agent = MCTSV32Agent()
        self.assertEqual(agent.simulations, 25)
        self.assertEqual(agent.candidate_limit, 16)
        self.assertEqual(agent.initial_width, 6)
        self.assertEqual(agent.neighborhood_radius, 2)
        self.assertEqual(agent.priority_top_k, 5)

    def test_black_four_three_is_detected(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK

        features = _pattern_features_for_move(game, BLACK, (7, 7))
        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.four_directions, 1)
        self.assertGreaterEqual(features.open_three_directions, 1)
        self.assertTrue(features.has_four_three)

    def test_white_double_four_is_detected(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (5, 7), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = WHITE
        game.to_play = WHITE

        features = _pattern_features_for_move(game, WHITE, (7, 7))
        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.four_directions, 2)

    def test_white_double_three_is_detected(self):
        game = Game()
        for move in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = WHITE
        game.to_play = WHITE

        features = _pattern_features_for_move(game, WHITE, (7, 7))
        self.assertTrue(features.legal)
        self.assertGreaterEqual(features.open_three_directions, 2)

    def test_same_shape_is_illegal_double_three_for_black(self):
        game = Game()
        for move in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK

        features = _pattern_features_for_move(game, BLACK, (7, 7))
        self.assertFalse(features.legal)

    def test_white_overline_is_immediate_win_and_tracks_run_length(self):
        game = Game()
        for col in range(2, 7):
            game.board[7][col] = WHITE
        game.to_play = WHITE

        features = _pattern_features_for_move(game, WHITE, (7, 7))
        self.assertTrue(features.legal)
        self.assertTrue(features.immediate_win)
        self.assertEqual(features.max_run, 6)

    def test_black_four_three_scores_above_quiet_move(self):
        game = Game()
        for move in ((7, 5), (7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK

        self.assertGreater(
            _v32_priority_score(game, (7, 7)),
            _v32_priority_score(game, (0, 0)),
        )

    def test_search_is_seeded_legal_and_preserves_state(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()
        first = MCTSV32Agent(seed=123, simulations=1).select_move(game)
        second = MCTSV32Agent(seed=123, simulations=1).select_move(game)
        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)


if __name__ == "__main__":
    unittest.main()
