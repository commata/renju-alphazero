from copy import deepcopy
import unittest
from unittest.mock import patch

from renju import BLACK, WHITE, Game
from renju.rules import DIRECTIONS, forbidden_reason
from search.threat_patterns import (
    compound_at, compound_moves, placed, fours_at, black_legal_43_moves,
    white_33_moves, white_43_moves, white_44_moves,
)


def position(stones=(), player=BLACK, opponents=()):
    game = Game()
    game.to_play = player
    for color, cells in ((player, stones), (-player, opponents)):
        for r, c in cells:
            game.board[r][c] = color
    return game


def cross(axis=(0, 1), kind='43', center=(7, 7)):
    dr, dc = axis
    er, ec = -dc, dr
    r, c = center
    first = (-2, -1, 1) if kind != '33' else (-1, 1)
    second = (-2, -1, 1) if kind == '44' else (-1, 1)
    return [(r+i*dr, c+i*dc) for i in first] + [(r+i*er, c+i*ec) for i in second]


class PatternTest(unittest.TestCase):
    def test_black_43_all_axes(self):
        for axis in DIRECTIONS:
            with self.subTest(axis=axis):
                game = position(cross(axis))
                before = deepcopy(vars(game))
                self.assertIsNone(forbidden_reason(game.board, 7, 7))
                self.assertIn((7, 7), black_legal_43_moves(game))
                self.assertEqual(vars(game), before)

    def test_white_patterns_all_axes(self):
        for kind, detector in [('33', white_33_moves), ('43', white_43_moves), ('44', white_44_moves)]:
            for axis in DIRECTIONS:
                with self.subTest(kind=kind, axis=axis):
                    self.assertIn((7, 7), detector(position(cross(axis, kind), WHITE)))

    def test_edge_closed_four_and_open_three(self):
        for player in (BLACK, WHITE):
            game = position([(1,0),(1,1),(1,2),(0,3),(2,3)], player)
            # At row 1 the vertical three cannot grow to an open four.
            self.assertIsNone(compound_at(game, player, (1,3)))
            game = position([(2,0),(2,1),(2,2),(1,3),(3,3)], player)
            self.assertIn('43', compound_at(game, player, (2,3)).kinds)

    def test_white_edge_33_and_44(self):
        for kind in ('33', '44'):
            game = position(cross(kind=kind, center=(2,2)), WHITE)
            self.assertIn(kind, compound_at(game, WHITE, (2,2)).kinds)
        game = position(cross(kind='33', center=(1,1)), WHITE)
        self.assertIsNone(compound_at(game, WHITE, (1,1)))

    def test_forbidden_creator_33_44_overline(self):
        for stones, reason in [(cross(kind='33'), '삼삼'), (cross(kind='44'), '사사'),
                               ([(7,c) for c in (2,3,4,5,6)] + [(6,7),(8,7)], '장목')]:
            game = position(stones)
            self.assertEqual(forbidden_reason(game.board, 7,7), reason)
            self.assertIsNone(compound_at(game, BLACK, (7,7)))

    def test_windows_are_canonical(self):
        game = position([(7,5),(7,6),(7,8)], WHITE)
        with placed(game, WHITE, (7,7)):
            fours = fours_at(game, WHITE, (7,7))
            self.assertEqual(len(fours), 1)
            self.assertEqual(fours[0].continuations, {(7,4),(7,9)})
        self.assertIsNone(compound_at(game, WHITE, (7,7)))

    def test_independent_same_axis_white_fours_are_not_duplicates(self):
        game = position([(7,7+i) for i in (-4,-3,-1,1,3,4)], WHITE)
        compound = compound_at(game, WHITE, (7,7))
        self.assertIn('44', compound.kinds)
        self.assertIn((7,7), white_44_moves(game))
        self.assertEqual({t.axis for t in compound.fours}, {(0,1)})
        self.assertEqual({tuple(sorted(t.continuations)) for t in compound.fours},
                         {((7,5),), ((7,9),)})

    def test_completion_cross_overline_included_when_it_makes_exact_five(self):
        game = position([(7,3),(7,4),(7,5),(4,7),(5,7),(6,7),(8,7),(9,7)])
        with placed(game, BLACK, (7,6)):
            self.assertTrue(any((7,7) in f.continuations for f in fours_at(game, BLACK, (7,6))))

    def test_fake_43_with_forbidden_three_extensions(self):
        stones = [(7,4),(7,5),(7,6),(6,7),(8,7)]
        stones += [(r,c) for r in (5,9) for c in (4,5,6)]
        game = position(stones, opponents=[(7,3)])
        self.assertIsNone(forbidden_reason(game.board, 7,7))
        self.assertIsNone(compound_at(game, BLACK, (7,7)))
        with placed(game, BLACK, (7,7)), placed(game, WHITE, (7,8)):
            for extension in ((5,7),(9,7)):
                self.assertEqual(forbidden_reason(game.board, *extension), '사사')

    def test_fake_43_with_double_three_extensions(self):
        stones = [(7,4),(7,5),(7,6),(6,7),(8,7)]
        for row in (5,9):
            stones += [(row,6),(row,8),(row-1,6),(row+1,8)]
        game = position(stones, opponents=[(7,3),(4,10),(10,4)])
        self.assertIsNone(forbidden_reason(game.board, 7,7))
        self.assertIsNone(compound_at(game, BLACK, (7,7)))
        with placed(game, BLACK, (7,7)), placed(game, WHITE, (7,8)):
            self.assertEqual(forbidden_reason(game.board, 5,7), '삼삼')
            self.assertEqual(forbidden_reason(game.board, 9,7), '사사')

    def test_structural_narrowing_matches_full_board_detector(self):
        from random import Random
        rng = Random(81)
        for player in (BLACK, WHITE):
            for _ in range(3):
                cells = rng.sample([(r,c) for r in range(3,12) for c in range(3,12)], 24)
                game = position(cells[:17], player, cells[17:])
                expected = {(r,c) for r in range(15) for c in range(15)
                            if compound_at(game, player, (r,c)) is not None}
                self.assertEqual(set(compound_moves(game, player)), expected)

    def test_43_three_is_validated_after_actual_four_reply(self):
        game = position([(7,5),(7,6),(7,8),(6,7),(8,7),(3,5),(4,6),(6,8)],
                        opponents=[(7,4),(2,4)])
        self.assertIn('43', compound_at(game, BLACK, (7,7)).kinds)
        with placed(game, BLACK, (7,7)):
            self.assertEqual(forbidden_reason(game.board, 5,7), '사사')
            with placed(game, WHITE, (7,9)):
                self.assertIsNone(forbidden_reason(game.board, 5,7))

    def test_engine_verifies_each_completion(self):
        for color in (BLACK, WHITE):
            for axis in DIRECTIONS:
                game = position(cross(axis), color)
                compound = compound_at(game, color, (7,7))
                game.play(7,7)
                for threat in compound.fours:
                    for point in threat.continuations:
                        state = deepcopy(game)
                        state.to_play = color
                        state.play(*point)
                        self.assertEqual(state.winner, color)

    def test_no_persistent_board_cache(self):
        game = position(cross(), WHITE)
        self.assertIn((7,7), white_43_moves(game))
        game.board[7][7] = BLACK
        self.assertNotIn((7,7), white_43_moves(game))

    def test_exception_state_restoration(self):
        game = position(cross(), WHITE)
        before = deepcopy(vars(game))
        with patch('search.threat_patterns.fours_at', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                compound_at(game, WHITE, (7,7))
        self.assertEqual(vars(game), before)

    def test_terminal_and_invalid_coordinates(self):
        game = Game()
        self.assertIsNone(compound_at(game, BLACK, (-1,0)))
        game.done = True
        self.assertEqual(compound_moves(game, WHITE), {})


if __name__ == '__main__':
    unittest.main()
