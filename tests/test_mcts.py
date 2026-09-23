from copy import deepcopy
import random
import unittest

from agents import MCTSAgent
from renju import BLACK, EMPTY, WHITE, Game, IllegalMove


def forced_win_position() -> Game:
    """Black has two legal moves: one wins now and the other loses next ply."""
    game = Game()
    game.board = [[WHITE] * 15 for _ in range(15)]
    for col in range(4):
        game.board[7][col] = BLACK
    game.board[7][4] = EMPTY
    game.board[0][0] = EMPTY
    game.to_play = BLACK
    return game


def forced_block_position() -> Game:
    """Black must occupy (7, 4) or white wins on the next move."""
    game = Game()
    for col in range(4):
        game.board[7][col] = WHITE
    game.to_play = BLACK
    return game


class MCTSTest(unittest.TestCase):
    def test_default_budget_and_candidate_limit(self):
        agent = MCTSAgent()
        self.assertEqual(agent.simulations, 10)
        self.assertEqual(agent.candidate_limit, 8)

    def test_forced_win_selected_even_with_one_simulation(self):
        game = forced_win_position()
        before = deepcopy(vars(game))
        move = MCTSAgent(seed=7, simulations=1).select_move(game)
        self.assertEqual(move, (7, 4))
        self.assertIn(move, game.legal_moves())
        self.assertEqual(vars(game), before)

    def test_forced_block_selected_even_with_one_simulation(self):
        game = forced_block_position()
        before = deepcopy(vars(game))
        move = MCTSAgent(seed=7, simulations=1).select_move(game)
        self.assertEqual(move, (7, 4))
        self.assertIn(move, game.legal_moves())
        self.assertEqual(vars(game), before)

    def test_seeded_search_is_reproducible_and_does_not_touch_global_random(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()
        first = MCTSAgent(seed=123, simulations=10).select_move(game)
        second = MCTSAgent(seed=123, simulations=10).select_move(game)
        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)

    def test_selected_move_is_legal_from_double_three_position(self):
        game = Game()
        for move in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[move[0]][move[1]] = BLACK
        game.to_play = BLACK
        before = deepcopy(vars(game))
        move = MCTSAgent(seed=21, simulations=10).select_move(game)
        self.assertIn(move, game.legal_moves())
        self.assertNotEqual(move, (7, 7))
        self.assertEqual(vars(game), before)

    def test_no_legal_moves_raise_without_mutation(self):
        game = Game()
        game.board = [[WHITE] * 15 for _ in range(15)]
        before = deepcopy(vars(game))
        with self.assertRaisesRegex(IllegalMove, "No legal moves"):
            MCTSAgent().select_move(game)
        self.assertEqual(vars(game), before)

    def test_invalid_configuration(self):
        for simulations in (0, -1, 1.5, True):
            with self.subTest(simulations=simulations), self.assertRaises(ValueError):
                MCTSAgent(simulations=simulations)
        for candidate_limit in (0, -1, 1.5, True):
            with self.subTest(candidate_limit=candidate_limit), self.assertRaises(ValueError):
                MCTSAgent(candidate_limit=candidate_limit)
        for exploration in (0, -0.1):
            with self.subTest(exploration=exploration), self.assertRaises(ValueError):
                MCTSAgent(exploration=exploration)


if __name__ == "__main__":
    unittest.main()
