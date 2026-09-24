from copy import deepcopy
import unittest

from renju import BLACK, WHITE, Game
from renju.rules import forbidden_reason
from search.mcts_v5 import (
    _four_completions, _is_unstoppable_four, _unstoppable_four_moves,
    _forced_v5_move, _RootContext, SearchDiagnostics,
)


def position(black=(), white=(), player=BLACK):
    game = Game()
    for color, moves in ((BLACK, black), (WHITE, white)):
        for r, c in moves:
            game.board[r][c] = color
    game.to_play = player
    return game


def trap(player):
    return position([(7,3),(7,4),(7,5),(7,7),(7,8),(3,6)],
                    [(4,6),(5,6),(6,6)], player)


class V5PolicyTest(unittest.TestCase):
    def test_a_forbidden_completion_attack_and_defense(self):
        for player in (WHITE, BLACK):
            game = trap(player)
            before = deepcopy(vars(game))
            self.assertIsNotNone(forbidden_reason(game.board, 7, 6))
            self.assertEqual(_unstoppable_four_moves(game, WHITE), [(8,6)])
            self.assertEqual(_forced_v5_move(game), (8,6))
            self.assertEqual(vars(game), before)

    def test_c_white_double_three(self):
        game = position([(10,10),(10,11),(11,10)], [(7,5),(7,6),(5,7),(6,7)])
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        self.assertEqual(_forced_v5_move(game, context=context), (7,7))
        self.assertEqual(context.diagnostics.forced_policy_stage, 5)

    def test_e_own_open_four(self):
        game = position([(7,6),(7,7),(7,8)], [(3,3),(3,4),(4,3)])
        self.assertIn(_forced_v5_move(game), {(7,5),(7,9)})

    def test_v4_open_three_defense(self):
        game = position(white=[(7,6),(7,7),(7,8)])
        self.assertIn(_forced_v5_move(game), {(7,5),(7,9)})

    def test_v4_immediate_win_priority(self):
        game = position([(4,c) for c in range(3,7)], [(7,6),(7,7),(7,8)])
        self.assertIn(_forced_v5_move(game), {(4,2),(4,7)})

    def test_opponent_immediate_win_overrides_four(self):
        game = position([(7,6),(7,7),(7,8)], [(3,c) for c in range(3,7)])
        self.assertFalse(_is_unstoppable_four(game, BLACK, (7,5)))
        self.assertIn(_forced_v5_move(game), {(3,2),(3,7)})

    def test_white_double_four_black_creator_illegal(self):
        stones = [(7,4),(7,5),(7,6),(4,7),(5,7),(6,7)]
        white = position(white=stones, player=WHITE)
        self.assertTrue(_is_unstoppable_four(white, WHITE, (7,7)))
        black = position(black=stones)
        self.assertFalse(_is_unstoppable_four(black, BLACK, (7,7)))

    def test_black_completion_cross_overline_rejected(self):
        game = position([(7,3),(7,4),(7,5),(4,7),(5,7),(6,7),(8,7),(9,7)])
        self.assertNotIn((7,7), _four_completions(game, BLACK, (7,6)))
        white = position(white=[(7,3),(7,4),(7,5),(7,8)])
        self.assertIn((7,7), _four_completions(white, WHITE, (7,6)))

    def test_closed_four_is_blockable(self):
        game = position([(7,4),(7,5),(7,6)], [(7,3)])
        self.assertFalse(_is_unstoppable_four(game, BLACK, (7,7)))


if __name__ == '__main__':
    unittest.main()
