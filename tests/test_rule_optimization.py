import unittest
from renju import Game
from renju.rules import forbidden_reason, legal_black_points
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

    def test_stage7_batched_black_legal_list_matches_reference(self):
        # Stage 7 criterion 5: dense tactical probe positions plus a second seeded corpus.
        import json
        from pathlib import Path
        root = Path(__file__).parent / 'fixtures'
        boards = []
        for name in ('stage7_probes_v1.json', 'stage7_probes_defense_v1.json'):
            for probe in json.loads((root / name).read_text(encoding='utf-8'))['probes']:
                game = Game()
                for move in probe['moves']:
                    game.play(*move)
                boards.append(game.board)
        boards += [game.board for _, game in positions(7, 400)]
        for board in boards:
            before = [row[:] for row in board]
            expected = [(r, c) for r in range(15) for c in range(15)
                        if board[r][c] == 0 and reference_rules.forbidden_reason(board, r, c) is None]
            self.assertEqual(legal_black_points(board), expected)
            self.assertEqual(board, before)
