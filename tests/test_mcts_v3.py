from copy import deepcopy
import random
import unittest

from agents import MCTSV2Agent, MCTSV3Agent
from renju import BLACK, EMPTY, WHITE, Game, IllegalMove
from search.mcts import MCTSNode
from search.mcts_v3 import _allowed_children, _root_candidates_v3


def forced_win_position() -> Game:
    game = Game()
    game.board = [[WHITE] * 15 for _ in range(15)]
    for col in range(4):
        game.board[7][col] = BLACK
    game.board[7][4] = EMPTY
    game.board[0][0] = EMPTY
    game.to_play = BLACK
    return game


def forced_block_position() -> Game:
    game = Game()
    for col in range(4):
        game.board[7][col] = WHITE
    game.to_play = BLACK
    return game


class MCTSV3Test(unittest.TestCase):
    def test_revision_defaults_are_kept_separate(self):
        v2 = MCTSV2Agent()
        self.assertEqual(v2.simulations, 10)
        self.assertEqual(v2.candidate_limit, 8)

        v3 = MCTSV3Agent()
        self.assertEqual(v3.simulations, 25)
        self.assertEqual(v3.candidate_limit, 16)
        self.assertEqual(v3.initial_width, 6)
        self.assertEqual(v3.neighborhood_radius, 2)

    def test_progressive_widening_keeps_revisits_with_larger_pool(self):
        node = MCTSNode(
            parent=None,
            move=None,
            player_just_moved=None,
            untried_moves=[(0, col) for col in range(15)] + [(1, 0)],
        )
        self.assertEqual(_allowed_children(node, 6), 6)
        node.visits = 9
        self.assertEqual(_allowed_children(node, 6), 9)
        node.visits = 25
        self.assertEqual(_allowed_children(node, 6), 11)

    def test_root_candidates_always_returns_pair(self):
        game = Game()
        moves, forced = _root_candidates_v3(game, candidate_limit=16, radius=2)
        self.assertIsInstance(moves, list)
        self.assertEqual(len(moves), 16)
        self.assertIsNone(forced)

    def test_v3_forced_win_and_block_with_one_simulation(self):
        for game, expected in (
            (forced_win_position(), (7, 4)),
            (forced_block_position(), (7, 4)),
        ):
            with self.subTest(expected=expected):
                before = deepcopy(vars(game))
                move = MCTSV3Agent(seed=7, simulations=1).select_move(game)
                self.assertEqual(move, expected)
                self.assertEqual(vars(game), before)

    def test_v3_seeded_reproducible_and_preserves_state(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()
        first = MCTSV3Agent(seed=123).select_move(game)
        second = MCTSV3Agent(seed=123).select_move(game)
        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)

    def test_v3_avoids_black_double_three(self):
        game = Game()
        for row, col in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[row][col] = BLACK
        game.to_play = BLACK
        before = deepcopy(vars(game))
        move = MCTSV3Agent(seed=21).select_move(game)
        self.assertIn(move, game.legal_moves())
        self.assertNotEqual(move, (7, 7))
        self.assertEqual(vars(game), before)

    def test_v3_no_legal_moves_raise(self):
        game = Game()
        game.board = [[WHITE] * 15 for _ in range(15)]
        before = deepcopy(vars(game))
        with self.assertRaisesRegex(IllegalMove, "No legal moves"):
            MCTSV3Agent().select_move(game)
        self.assertEqual(vars(game), before)

    def test_v3_invalid_configuration(self):
        for kwargs in (
            {"simulations": 0},
            {"candidate_limit": 0},
            {"initial_width": 0},
            {"candidate_limit": 4, "initial_width": 5},
            {"neighborhood_radius": 0},
            {"exploration": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MCTSV3Agent(**kwargs)


if __name__ == "__main__":
    unittest.main()
