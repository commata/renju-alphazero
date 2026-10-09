"""E1-S stage-route layer (scripts/e1s_build_suite.py)."""
import json
from pathlib import Path
import unittest

from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, UNKNOWN
from scripts.e1s_build_suite import (
    STAGE_ROUTES, _engine_budget, classify, counterfactual, defense_set, outcome, screen_items,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = (ROOT / 'docs' / 'mcts-v8-results' / 'web_stress_20261009'
       / '20261009-155734-130351_s3_human-black_seed43535' / 'game.json')
L, C, U = {'status': PROVEN_LOSS}, {'status': CLEAR}, {'status': UNKNOWN}


def p14_history():
    """Web game 155734 before white 14 (§12.27): V7 chose (6,10), V8-A played (9,7) (a depth-2 loss)."""
    moves = json.loads(WEB.read_text(encoding='utf-8'))['moves']
    return [[m['row0'], m['col0']] for m in moves[:13]]


class ClassTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(classify({'a': L, 'b': C, 'c': U}), 'ROUTE_COVERAGE_MISS')
        self.assertEqual(classify({'a': L, 'b': U}), 'ROUTE_COVERAGE_UNRESOLVED')
        self.assertEqual(classify({'a': L, 'b': L}), 'ROUTE_NO_RESCUE')

    def test_outcomes(self):
        truths = {'1,1': L, '2,2': C, '3,3': L, '4,4': U}

        def cf(final, switched, first):
            return {'final': list(final), 'switched': switched, 'checked': [[[1, 1], first]]}
        self.assertEqual(outcome(cf((1, 1), False, 'UNKNOWN'), truths), 'DETECT_MISS')
        self.assertEqual(outcome(cf((2, 2), True, PROVEN_LOSS), truths), 'RESCUED')
        self.assertEqual(outcome(cf((1, 1), False, PROVEN_LOSS), truths), 'KEPT')
        self.assertEqual(outcome(cf((3, 3), True, PROVEN_LOSS), truths), 'SWITCH_TO_LOSS')
        self.assertEqual(outcome(cf((4, 4), True, PROVEN_LOSS), truths), 'SWITCH_TO_UNRESOLVED')


class SourceTest(unittest.TestCase):
    def test_screen_items_are_stage_moves_not_already_lost(self):
        items, skipped = screen_items()
        self.assertTrue(items)
        self.assertEqual(skipped['v8a_proven_loss'], 26)  # 4 + 9 + 3 + 10 in the four logs
        for item in items:
            self.assertIn(item['source_route'], STAGE_ROUTES)
            self.assertNotEqual(item['v8a_status'], 'UNSAFE')
            self.assertEqual((item['vct2_eligible'], item['vct2_checked']), (False, False))
            self.assertEqual(len(item['history']), item['ply'])


class P14Test(unittest.TestCase):
    def test_defense_set_and_s3_rule(self):
        history = p14_history()
        defenses = defense_set(history, 'stage4', [6, 10])
        forced = [tuple(d['move']) for d in defenses if d['tier'] == 'forced']
        self.assertEqual(forced[0], (6, 10))  # V7's choice first
        self.assertIn((9, 7), forced)
        self.assertEqual(len({tuple(d['move']) for d in defenses}), len(defenses))
        with self.assertRaises(ValueError):
            defense_set(history, 'stage5', [6, 10])
        # §12.27: the 10k check proves (9,7) lost (7,779 nodes) and leaves (6,10) UNKNOWN -> switch.
        cf = counterfactual(history, [9, 7], defenses, [[[6, 10], 'UNKNOWN'], [[9, 7], 'SAFE']],
                            _engine_budget(), 4)
        self.assertEqual(cf['checked'], [[[9, 7], PROVEN_LOSS], [[6, 10], 'UNKNOWN']])
        self.assertEqual((cf['final'], cf['switched']), ([6, 10], True))


if __name__ == '__main__':
    unittest.main()
