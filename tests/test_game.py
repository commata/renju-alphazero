import unittest
from unittest.mock import patch

from renju import BLACK, WHITE, Game, IllegalMove
from renju.rules import _open_three, forbidden_reason


def position(black=(), white=(), turn=BLACK):
    g = Game()
    for r, c in black:
        g.board[r][c] = BLACK
    for r, c in white:
        g.board[r][c] = WHITE
    g.to_play = turn
    return g


class RulesTest(unittest.TestCase):
    def test_opening_move_is_forced_to_center(self):
        g = Game()
        self.assertEqual(g.legal_moves(), [(7, 7)])
        self.assertTrue(g.has_legal_move())
        with self.assertRaisesRegex(IllegalMove, '정중앙'):
            g.play(7, 8)
        self.assertEqual(g.history, [])
        g.play(7, 7)
        self.assertEqual(g.board[7][7], BLACK)
        self.assertEqual(g.to_play, WHITE)

    def test_direct_board_fixture_is_not_treated_as_initial_position(self):
        g = Game()
        g.board[0][0] = BLACK
        g.to_play = WHITE
        self.assertNotEqual(g.legal_moves(), [(7, 7)])
        g.play(0, 1)
        self.assertEqual(g.board[0][1], WHITE)

        empty_white = Game()
        empty_white.to_play = WHITE
        self.assertGreater(len(empty_white.legal_moves()), 1)

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

    def test_exact_five_precedes_simultaneous_overline_and_restores_board(self):
        g = position(black=[(7, c) for c in (3, 4, 5, 6, 8)] +
                     [(r, 7) for r in (3, 4, 5, 6)])
        before = [row[:] for row in g.board]
        self.assertIsNone(forbidden_reason(g.board, 7, 7))
        self.assertEqual(g.board, before)
        self.assertIn((7, 7), g.legal_moves())
        g.play(7, 7)
        self.assertTrue(g.done)
        self.assertEqual(g.winner, BLACK)

    def test_white_overline_wins(self):
        g = position(white=[(7, c) for c in (3, 4, 5, 6, 8)], turn=WHITE)
        g.play(7, 7)
        self.assertEqual(g.winner, WHITE)

    def test_double_four(self):
        g = position(black=[(7, c) for c in (5, 6, 8)] + [(r, 7) for r in (5, 6, 8)])
        self.assertEqual(forbidden_reason(g.board, 7, 7), '사사')
        g = position(black=[(7, c) for c in (5, 6, 8)])
        self.assertIsNone(forbidden_reason(g.board, 7, 7))

        # RIF FOUR requires a completion to five. The horizontal apparent
        # four below can only complete at (7,8), which would make six because
        # (7,9) is already black; it therefore must not become a second four.
        pseudo = position(
            black=[(7, c) for c in (4, 5, 6, 9)] + [(r, 7) for r in (5, 6, 8)],
            white=[(7, 3)],
        )
        self.assertIsNone(forbidden_reason(pseudo.board, 7, 7))

    def test_double_three_and_blocked_three(self):
        stones = [(7, 6), (7, 8), (6, 7), (8, 7)]
        g = position(black=stones)
        self.assertEqual(forbidden_reason(g.board, 7, 7), '삼삼')
        with self.assertRaisesRegex(IllegalMove, '삼삼'):
            g.play(7, 7)
        blocked = position(black=stones, white=[(7, 4), (7, 10)])
        self.assertIsNone(forbidden_reason(blocked.board, 7, 7))

    def test_recursive_double_three_extension_does_not_count_as_open_three(self):
        black = [
            (7, 6), (7, 8), (6, 7), (8, 7),
            (6, 5), (8, 5), (6, 4), (8, 6),
        ]
        g = position(black=black, white=[(7, 10)])
        before = [row[:] for row in g.board]

        # The horizontal apparent three can only be extended at (7, 5).
        # That extension itself is a forbidden double-three, so the original
        # move at (7, 7) has only one valid three and must be legal.
        self.assertIsNone(forbidden_reason(g.board, 7, 7))
        self.assertEqual(g.board, before)

        g.board[7][7] = BLACK
        try:
            self.assertEqual(forbidden_reason(g.board, 7, 5), '삼삼')
        finally:
            g.board[7][7] = 0
        self.assertEqual(g.board, before)

        g.play(7, 7)
        self.assertEqual(g.board[7][7], BLACK)
        self.assertEqual(g.to_play, WHITE)

    def test_three_extension_that_already_makes_five_is_not_three(self):
        g = position(
            black=[(7, 6), (7, 7), (7, 8), (3, 5), (4, 5), (5, 5), (6, 5)],
            white=[(7, 10)],
        )
        self.assertFalse(_open_three(g.board, (7, 7), 0, 1))

    def test_has_legal_move_matches_full_generation(self):
        cases = [
            Game(),
            position(black=[(7, 6), (7, 8), (6, 7), (8, 7)]),
            position(black=[(7, 7)], white=[(0, 0)], turn=BLACK),
            position(black=[(7, 7)], turn=WHITE),
        ]
        for game in cases:
            with self.subTest(turn=game.to_play, history=len(game.history)):
                self.assertEqual(game.has_legal_move(), bool(game.legal_moves()))

        full = Game()
        full.board = [[WHITE] * 15 for _ in range(15)]
        full.to_play = BLACK
        self.assertFalse(full.has_legal_move())
        self.assertEqual(full.legal_moves(), [])

    def test_play_terminal_check_does_not_build_full_legal_list(self):
        game = Game()
        with patch.object(Game, "legal_moves", side_effect=AssertionError("full list generated")):
            game.play(7, 7)
            game.play(0, 0)
        self.assertEqual(game.history, [(7, 7), (0, 0)])

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
