from copy import deepcopy
import random
import unittest

from agents import MCTSV41Agent, MCTSV42Agent
from renju import BLACK, WHITE, Game
from search.mcts_v4 import _creates_open_four


class MCTSV4PolicyTest(unittest.TestCase):
    def test_v41_defaults(self):
        agent = MCTSV41Agent()
        self.assertEqual(agent.simulations, 50)
        self.assertEqual(agent.candidate_limit, 20)
        self.assertEqual(agent.initial_width, 8)
        self.assertEqual(agent.neighborhood_radius, 2)
        self.assertEqual(agent.priority_top_k, 8)

    def test_v42_defaults(self):
        agent = MCTSV42Agent()
        self.assertEqual(agent.simulations, 50)
        self.assertEqual(agent.candidate_limit, 24)
        self.assertEqual(agent.initial_width, 10)
        self.assertEqual(agent.neighborhood_radius, 2)
        self.assertEqual(agent.priority_top_k, 10)

    def test_open_three_creator_is_detected(self):
        game = Game()
        for col in (6, 7, 8):
            game.board[7][col] = WHITE

        self.assertTrue(
            _creates_open_four(
                game,
                WHITE,
                (7, 5),
                assume_legal=True,
            )
        )
        self.assertTrue(
            _creates_open_four(
                game,
                WHITE,
                (7, 9),
                assume_legal=True,
            )
        )

    def test_v41_forces_block_against_white_open_three(self):
        game = Game()
        for col in (6, 7, 8):
            game.board[7][col] = WHITE
        game.to_play = BLACK

        move = MCTSV41Agent(seed=42, simulations=1).select_move(game)
        self.assertIn(move, {(7, 5), (7, 9)})

    def test_v42_forces_block_against_white_open_three(self):
        game = Game()
        for col in (6, 7, 8):
            game.board[7][col] = WHITE
        game.to_play = BLACK

        move = MCTSV42Agent(seed=42, simulations=1).select_move(game)
        self.assertIn(move, {(7, 5), (7, 9)})

    def test_immediate_win_stays_above_open_three_defense(self):
        game = Game()
        for col in range(3, 7):
            game.board[4][col] = BLACK
        for col in (6, 7, 8):
            game.board[7][col] = WHITE
        game.to_play = BLACK

        move = MCTSV41Agent(seed=42, simulations=1).select_move(game)
        self.assertIn(move, {(4, 2), (4, 7)})

    def test_search_is_seeded_legal_and_preserves_state(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()

        first = MCTSV41Agent(seed=123, simulations=1).select_move(game)
        second = MCTSV41Agent(seed=123, simulations=1).select_move(game)

        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)


if __name__ == "__main__":
    unittest.main()
