from copy import deepcopy
import random
import unittest

from agents import RandomAgent, TacticalAgent
from renju import BLACK, WHITE, Game, IllegalMove


def position(black=(), white=(), turn=BLACK):
    game = Game()
    for row, col in black:
        game.board[row][col] = BLACK
    for row, col in white:
        game.board[row][col] = WHITE
    game.to_play = turn
    return game


class AgentTest(unittest.TestCase):
    def test_random_legal_seeded_and_no_global_random_changes(self):
        game = Game()
        before = deepcopy(vars(game))
        global_state = random.getstate()
        first = RandomAgent(42)
        second = RandomAgent(42)
        legal = game.legal_moves()
        for _ in range(5):
            move = first.select_move(game)
            self.assertIn(move, legal)
            self.assertEqual(move, second.select_move(game))
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)

    def test_no_legal_moves_raise(self):
        for done in (False, True):
            game = Game()
            game.done = done
            if not done:
                game.board = [[WHITE] * 15 for _ in range(15)]
            for agent in (RandomAgent(), TacticalAgent()):
                with self.subTest(done=done, agent=agent.name):
                    before = deepcopy(vars(game))
                    with self.assertRaisesRegex(IllegalMove, "No legal moves"):
                        agent.select_move(game)
                    self.assertEqual(vars(game), before)

    def assert_choice_preserves_state(self, game, expected):
        before = deepcopy(vars(game))
        move = TacticalAgent().select_move(game)
        self.assertEqual(move, expected)
        self.assertIn(move, game.legal_moves())
        self.assertEqual(vars(game), before)
        return move

    def test_immediate_win_both_colors_and_white_overline(self):
        for color, cols in ((BLACK, (0, 1, 2, 3)), (WHITE, (0, 1, 2, 3)),
                            (WHITE, (0, 1, 2, 3, 5))):
            with self.subTest(color=color, cols=cols):
                stones = [(7, col) for col in cols]
                game = position(black=stones if color == BLACK else (),
                                white=stones if color == WHITE else (), turn=color)
                move = self.assert_choice_preserves_state(game, (7, 4))
                game.play(*move)
                self.assertEqual(game.winner, color)

    def test_blocks_immediate_win_for_both_colors(self):
        for color in (BLACK, WHITE):
            stones = [(7, col) for col in range(4)]
            game = position(black=stones if color == WHITE else (),
                            white=stones if color == BLACK else (), turn=color)
            self.assert_choice_preserves_state(game, (7, 4))

    def test_win_precedes_block(self):
        game = position(black=[(7, col) for col in range(4)],
                        white=[(0, col) for col in range(4)])
        self.assert_choice_preserves_state(game, (7, 4))

    def test_forbidden_black_move_is_not_chosen_or_treated_as_threat(self):
        # Filling (7, 4) makes a black overline, but (10, 4) is a real win.
        stones = [(7, col) for col in (0, 1, 2, 3, 5)]
        stones += [(10, col) for col in range(4)]
        for turn in (BLACK, WHITE):
            with self.subTest(turn=turn):
                self.assert_choice_preserves_state(position(black=stones, turn=turn), (10, 4))

    def test_forbidden_block_is_not_selected(self):
        game = position(black=[(7, 6), (7, 8), (6, 7), (8, 7)],
                        white=[(r, r) for r in (3, 4, 5, 6)])
        before = deepcopy(vars(game))
        move = TacticalAgent().select_move(game)
        self.assertNotEqual(move, (7, 7))
        self.assertIn(move, game.legal_moves())
        self.assertEqual(vars(game), before)

    def test_fallback_seed_and_state_with_real_history(self):
        game = Game()
        game.play(7, 7)
        game.play(0, 0)
        before = deepcopy(vars(game))
        global_state = random.getstate()
        move = TacticalAgent(42).select_move(game)
        self.assertEqual(move, TacticalAgent(42).select_move(game))
        self.assertEqual(move, RandomAgent(42).select_move(game))
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), global_state)


if __name__ == '__main__':
    unittest.main()
