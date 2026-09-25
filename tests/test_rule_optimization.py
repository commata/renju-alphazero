import unittest
from renju import Game
from renju.rules import forbidden_reason
import reference_rules
from rule_validation import positions, validate


class RuleDifferentialTest(unittest.TestCase):
    def test_seeded_all_empty_cells_and_ordered_legal_lists(self):
        stats = validate(positions(42, 240))
        self.assertEqual(stats['positions'], 240)
        for key in ('mismatches', 'legal_mismatches', 'restoration_failures'):
            self.assertEqual(stats[key], 0)
        for reason in ('삼삼', '사사', '장목'):
            self.assertGreater(stats[reason], 0)

    def test_invalid_and_occupied_behavior(self):
        game = Game()
        game.board[7][7] = 1
        before = [row[:] for row in game.board]
        for point in ((-1,0),(15,0),(0,-1),(0,15),(7,7)):
            messages = []
            for function in (reference_rules.forbidden_reason, forbidden_reason):
                with self.assertRaises(ValueError) as caught:
                    function(game.board, *point)
                messages.append(str(caught.exception))
                self.assertEqual(game.board, before)
            self.assertEqual(*messages)
