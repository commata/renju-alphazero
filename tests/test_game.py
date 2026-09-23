import unittest
from renju import BLACK, WHITE, Game, IllegalMove
from renju.rules import forbidden_reason


def position(black=(), white=(), turn=BLACK):
    g = Game()
    for r, c in black:
        g.board[r][c] = BLACK
    for r, c in white:
        g.board[r][c] = WHITE
    g.to_play = turn
    return g


class RulesTest(unittest.TestCase):
    def test_turn_undo_and_occupied(self):
        g = Game()
        g.play(7, 7)
        self.assertEqual(g.to_play, WHITE)
        with self.assertRaises(IllegalMove):
            g.play(7, 7)
        g.play(7, 8)
        self.assertEqual(g.undo(), (7, 8))
        self.assertEqual(g.to_play, WHITE)
        g.undo()
        self.assertEqual(g.to_play, BLACK)
        self.assertEqual(g.board[7][7], 0)

    def test_black_exact_five(self):
        g = position(black=[(7, c) for c in range(4, 8)])
        g.play(7, 8)
        self.assertTrue(g.done)
        self.assertEqual(g.winner, BLACK)
        self.assertEqual(g.legal_moves(), [])
        with self.assertRaises(IllegalMove):
            g.play(0, 0)
        g.undo()
        self.assertFalse(g.done)

    def test_overline_precedes_five_and_restores_board(self):
        g = position(black=[(7, c) for c in (3, 4, 5, 6, 8)] +
                     [(r, 7) for r in (3, 4, 5, 6)])
        before = [row[:] for row in g.board]
        self.assertEqual(forbidden_reason(g.board, 7, 7), '장목')
        self.assertEqual(g.board, before)
        with self.assertRaisesRegex(IllegalMove, '장목'):
            g.play(7, 7)
        self.assertNotIn((7, 7), g.legal_moves())

    def test_white_overline_wins(self):
        g = position(white=[(7, c) for c in (3, 4, 5, 6, 8)], turn=WHITE)
        g.play(7, 7)
        self.assertEqual(g.winner, WHITE)

    def test_double_four(self):
        g = position(black=[(7, c) for c in (5, 6, 8)] + [(r, 7) for r in (5, 6, 8)])
        self.assertEqual(forbidden_reason(g.board, 7, 7), '사사')
        g = position(black=[(7, c) for c in (5, 6, 8)])
        self.assertIsNone(forbidden_reason(g.board, 7, 7))

    def test_double_three_and_blocked_three(self):
        stones = [(7, 6), (7, 8), (6, 7), (8, 7)]
        g = position(black=stones)
        self.assertEqual(forbidden_reason(g.board, 7, 7), '삼삼')
        with self.assertRaisesRegex(IllegalMove, '삼삼'):
            g.play(7, 7)
        blocked = position(black=stones, white=[(7, 4), (7, 10)])
        self.assertIsNone(forbidden_reason(blocked.board, 7, 7))

    def test_five_precedes_forks(self):
        g = position(black=[(7, c) for c in (3, 4, 5, 6)] +
                     [(6, 7), (8, 7), (6, 6), (8, 8)])
        self.assertIsNone(forbidden_reason(g.board, 7, 7))
        g.play(7, 7)
        self.assertEqual(g.winner, BLACK)

    def test_invalid_coordinate(self):
        with self.assertRaises(IllegalMove):
            Game().play(-1, 0)


if __name__ == '__main__':
    unittest.main()
