"""Known defects of frozen proof code, kept as expected failures until a new proof cycle fixes them.

``analysis.threats.ThreatSolver._after_move``, forced block of our only four: when the defender's block ends the
game without a winner (it fills the last legal point: a full board, or Black left with forbidden points only),
``decision`` sees no legal move and ``decision_status(Counter())`` reads UNSAFE: "the side that just moved is
lost" in a drawn position (docs/h6-value.md, "코드 감사"). ``threats.py`` is frozen (CLAUDE.md); no game in the
corpora gets near a full board, so the fix belongs to the next proof cycle. When it is fixed, this test passes
unexpectedly: then remove the decorator.
"""
import random
import unittest

from analysis.threats import SAFE, ThreatSolver
from renju import BLACK, EMPTY, SIZE, WHITE, Game


def _no_five(board) -> bool:
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == EMPTY:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                n, rr, cc = 0, r, c
                while 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == board[r][c]:
                    n, rr, cc = n + 1, rr + dr, cc + dc
                if n >= 5:
                    return False
    return True


def _exact_five_through(board, r, c, color) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == color:
                n, rr, cc = n + 1, rr + dr * s, cc + dc * s
        if n == 5:
            return True
    return False


def last_point_block():
    """A full board but one point W: Black's four completes at W, White (to move) must block it there."""
    pattern = [[BLACK if (r + 2 * c) % 4 < 2 else WHITE for c in range(SIZE)] for r in range(SIZE)]
    rng = random.Random(0)
    while True:
        board = [row[:] for row in pattern]
        r, c = rng.randrange(SIZE), rng.randrange(SIZE)
        dr, dc = rng.choice(((0, 1), (1, 0), (1, 1), (1, -1)))
        k = rng.randrange(5)
        cells = [(r + dr * (i - k), c + dc * (i - k)) for i in range(5)]
        if not all(0 <= x < SIZE and 0 <= y < SIZE for x, y in cells):
            continue
        for x, y in cells:
            board[x][y] = BLACK
        w = cells[k]
        board[w[0]][w[1]] = EMPTY
        if _no_five(board) and _exact_five_through(board, *w, BLACK) and not _exact_five_through(board, *w, WHITE):
            game = Game()
            game.board, game.to_play, game.history = board, WHITE, [(0, 0) if w != (0, 0) else (0, 1)]
            return game, w


class ForcedBlockDrawTest(unittest.TestCase):
    def test_the_fixture(self):
        game, w = last_point_block()
        self.assertEqual(game.legal_moves(), [w])
        game.play(*w)
        self.assertTrue(game.done)
        self.assertIsNone(game.winner)  # the block draws: Black is not lost

    @unittest.expectedFailure
    def test_a_drawing_block_is_not_a_loss(self):
        game, _ = last_point_block()
        self.assertEqual(ThreatSolver().after_move(game, 0)[0], SAFE)  # today: UNSAFE ('block', w)


class LegalMoveInvariantTest(unittest.TestCase):
    def test_legal_moves_and_has_legal_move_agree(self):
        rng = random.Random(7)
        checked = 0
        for _ in range(30):
            game = Game()
            while not game.done:
                legal = game.legal_moves()
                self.assertEqual(bool(legal), game.has_legal_move())
                checked += 1
                game.play(*rng.choice(legal))
            self.assertEqual(game.legal_moves(), [])
            self.assertFalse(game.has_legal_move())
        self.assertGreater(checked, 300)


if __name__ == '__main__':
    unittest.main()
