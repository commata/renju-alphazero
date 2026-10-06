import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from summarize_h4 import safety  # noqa: E402

# White (moves 1, 3, 5, 7) has a four on row 3 blocked at (3, 2); black to move at ply 8.
MOVES = [[7, 7], [3, 3], [3, 2], [3, 4], [11, 11], [3, 5], [11, 1], [3, 6]]


def run(checked):
    move = {'ply': 8, 'route': 'stage4', 'played': [0, 0], 'vct': {'checked': checked},
            'root': {'checked': []}}
    return {'games': [{'key': 'g', 'moves': MOVES + [[0, 0]], 'v8_moves': [move]}]}


class SafetyInvariantTest(unittest.TestCase):
    def test_unrefuted_alternative_is_a_violation(self):
        report = safety(run([[[0, 0], 'UNSAFE'], [[3, 7], 'UNKNOWN']]))
        self.assertEqual(len(report['violations']), 1)

    def test_alternative_that_loses_at_once_is_not_an_alternative(self):
        # (14, 14) leaves white's five at (3, 7): V8's fallback skips it by design.
        report = safety(run([[[0, 0], 'UNSAFE'], [[14, 14], 'UNKNOWN']]))
        self.assertEqual(report, {'violations': [], 'proven_loss_in_lost_position': 1})

    def test_all_unsafe_is_a_lost_position(self):
        report = safety(run([[[0, 0], 'UNSAFE'], [[3, 7], 'UNSAFE']]))
        self.assertEqual(report, {'violations': [], 'proven_loss_in_lost_position': 1})


if __name__ == '__main__':
    unittest.main()
