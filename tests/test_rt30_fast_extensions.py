"""rt30.fast_extensions: exact replacement of search.mcts_v321._fast_winning_extensions_in_direction."""
import unittest

from renju import BLACK, EMPTY, SIZE, WHITE
from renju.rules import DIRECTIONS
from rt30 import fast, fast_extensions
from search import mcts_v4, mcts_v5, mcts_v321

from test_rt30_fast_windows import SameSearchTest, random_positions

FROZEN = fast_extensions.FROZEN
NEW = fast_extensions.winning_extensions_in_direction


def every_case(board):
    for r in range(SIZE):
        for c in range(SIZE):
            for player in (BLACK, WHITE):
                for dr, dc in DIRECTIONS:
                    yield player, (r, c), dr, dc


class SameSetTest(unittest.TestCase):
    def test_every_cell_direction_and_colour_of_random_positions(self):
        for board in random_positions(250, seed=7):
            for player, anchor, dr, dc in every_case(board):
                self.assertEqual(NEW(board, player, anchor, dr, dc), FROZEN(board, player, anchor, dr, dc),
                                 (player, anchor, dr, dc))

    def test_lines_overlines_edges_and_gaps(self):
        # Hand-made rows: fours, broken fours, exact five vs overline for black, blocked ends, board edges.
        patterns = ['BBBB.', '.BBBB', 'BB.BB', 'B.BBB', 'BBBBB.', '.BBBB.B', 'B.BBBB', 'BBB.BB', 'WWWW.W',
                    'W.WWWW', '.WWWWW', 'WBBBB.', '.BBBBW', 'BBBB', 'B.B.B.B', '..BBB..', 'W.W.W.W.W']
        for pattern in patterns:
            for offset in (0, 3, SIZE - len(pattern)):
                for row in (0, 7, SIZE - 1):
                    board = [[EMPTY] * SIZE for _ in range(SIZE)]
                    for i, ch in enumerate(pattern):
                        board[row][offset + i] = {'B': BLACK, 'W': WHITE, '.': EMPTY}[ch]
                    for transpose in (False, True):
                        b = [list(x) for x in zip(*board)] if transpose else board
                        for player, anchor, dr, dc in every_case(b):
                            self.assertEqual(NEW(b, player, anchor, dr, dc), FROZEN(b, player, anchor, dr, dc),
                                             (pattern, offset, row, transpose, player, anchor, dr, dc))

    def test_board_is_unchanged(self):
        board = random_positions(60, seed=3)[-1]
        before = [row[:] for row in board]
        for player, anchor, dr, dc in every_case(board):
            NEW(board, player, anchor, dr, dc)
        self.assertEqual(board, before)


class InstallTest(unittest.TestCase):
    def tearDown(self):
        fast.uninstall_all()

    def test_install_rebinds_every_importer(self):
        self.assertGreaterEqual(fast_extensions.install(), 3)
        for module in (mcts_v321, mcts_v5, mcts_v4):
            self.assertIs(module._fast_winning_extensions_in_direction, NEW)
        fast_extensions.uninstall()
        for module in (mcts_v321, mcts_v5, mcts_v4):
            self.assertIs(module._fast_winning_extensions_in_direction, FROZEN)

    def test_install_all(self):
        fast.install_all()
        self.assertEqual(fast.installed(), {'windows': True, 'extensions': True})
        fast.uninstall_all()
        self.assertEqual(fast.installed(), {'windows': False, 'extensions': False})


class SameSearchAllFastTest(SameSearchTest):
    """V8-B end to end with every exact speed-up installed."""

    def test_v8b_identical(self):
        from test_rt30_fast_windows import recorded_position

        game = recorded_position(8415, 11, 15)
        expected = self.run_attack(game)
        with fast.all_fast():
            self.assertEqual(self.run_attack(game), expected)


if __name__ == '__main__':
    unittest.main()
