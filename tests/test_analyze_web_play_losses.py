"""Three-state classification of scripts/analyze_web_play_losses.py."""
from collections import Counter
import sys
import unittest
from pathlib import Path

from renju import Game

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from analyze_web_play_losses import (  # noqa: E402
    SAFE, UNKNOWN, UNSAFE, after_move_status, decision_status,
)


def play(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


# Black (7,5)(7,6)(7,7), white blocks (7,4); white to move.
CLOSED_THREE = [(7, 7), (0, 0), (7, 6), (7, 4), (7, 5)]


class AnalyzeWebPlayLossesTest(unittest.TestCase):
    def test_opponent_open_four_is_unsafe(self):
        game = play([(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0), (7, 8), (14, 14)])
        self.assertEqual(after_move_status(game, node_limit=1000, four_chain=6), UNSAFE)

    def test_completed_search_without_vcf_is_safe(self):
        game = play(CLOSED_THREE + [(14, 14)])
        self.assertEqual(after_move_status(game, node_limit=100_000, four_chain=6), SAFE)

    def test_exhausted_budget_is_unknown_not_safe(self):
        game = play(CLOSED_THREE + [(14, 14)])
        self.assertEqual(after_move_status(game, node_limit=1, four_chain=6), UNKNOWN)

    def test_own_four_chain_beyond_limit_is_unknown(self):
        # White makes a closed four (black holds (3,2)); with no chain budget the
        # position after black's forced block is not explored.
        game = play([(7, 7), (3, 3), (3, 2), (3, 4), (7, 6), (3, 5), (12, 12), (3, 6)])
        self.assertEqual(after_move_status(game, node_limit=100_000, four_chain=0), UNKNOWN)

    def test_decision_status_needs_no_safe_and_no_unknown_for_loss(self):
        self.assertEqual(decision_status(Counter({UNSAFE: 3})), UNSAFE)
        self.assertEqual(decision_status(Counter({UNSAFE: 3, UNKNOWN: 1})), UNKNOWN)
        self.assertEqual(decision_status(Counter({UNSAFE: 3, UNKNOWN: 1, SAFE: 1})), SAFE)


if __name__ == '__main__':
    unittest.main()
