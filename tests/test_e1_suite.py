"""E1 rescue suite: selection rules and run outcomes (scripts/e1_build_suite.py, e1_evaluate.py)."""
import unittest

from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, UNKNOWN, canonical_hash, classify, select
from scripts.e1_evaluate import outcome

L, C, U = {'status': PROVEN_LOSS}, {'status': CLEAR}, {'status': UNKNOWN}


def run(pre, played, checked, children, switched=None, route='tree'):
    return {'route': route, 'pre': list(pre), 'played': list(played), 'checked': [[list(m), s] for m, s in checked],
            'switched': played != pre if switched is None else switched,
            'children': [[list(m), v, 0.0] for m, v in children]}


CHILDREN = [((1, 1), 9), ((2, 2), 5), ((3, 3), 4), ((4, 4), 3), ((5, 5), 2)]


class OutcomeTest(unittest.TestCase):
    def test_rescued_and_avoided(self):
        truths = {'1,1': L, '2,2': C}
        r = run((1, 1), (2, 2), [((1, 1), 'UNSAFE'), ((2, 2), 'SAFE')], CHILDREN)
        self.assertEqual(outcome('E1-P', truths, r), 'RESCUED')
        r = run((2, 2), (2, 2), [((2, 2), 'SAFE')], CHILDREN)
        self.assertEqual(outcome('E1-P', truths, r), 'TREE_AVOIDED')

    def test_detect_miss(self):
        r = run((1, 1), (1, 1), [((1, 1), 'UNKNOWN')], CHILDREN)
        self.assertEqual(outcome('E1-P', {'1,1': L, '2,2': C}, r), 'DETECT_MISS')

    def test_budget_ambiguity_is_the_8411_p8_type(self):
        # pre lost and detected; the next child is UNKNOWN to the 10k check but lost; a clear move follows.
        truths = {'1,1': L, '2,2': L, '3,3': C}
        r = run((1, 1), (2, 2), [((1, 1), 'UNSAFE'), ((2, 2), 'UNKNOWN')], CHILDREN)
        self.assertEqual(outcome('E1-P', truths, r), 'BUDGET_AMBIGUITY')

    def test_k4_and_pool_miss(self):
        truths = {'1,1': L, '2,2': L, '3,3': L, '4,4': L, '5,5': C}
        r = run((1, 1), (1, 1), [((1, 1), 'UNSAFE'), ((2, 2), 'UNSAFE'), ((3, 3), 'UNSAFE'), ((4, 4), 'UNSAFE')],
                CHILDREN)
        self.assertEqual(outcome('E1-P', truths, r), 'K4_MISS')
        r2 = run((1, 1), (1, 1), r['checked'] and [((1, 1), 'UNSAFE')], CHILDREN[:4])
        self.assertEqual(outcome('E1-P', truths, r2), 'POOL_MISS')

    def test_route_other_no_rescue_and_controls(self):
        r = run((1, 1), (1, 1), [], CHILDREN, route='stage4')
        self.assertEqual(outcome('E1-P', {'1,1': L, '2,2': C}, r), 'ROUTE_OTHER')
        r = run((1, 1), (2, 2), [((1, 1), 'UNSAFE'), ((2, 2), 'UNKNOWN')], CHILDREN)
        self.assertEqual(outcome('E1-N', {'1,1': L, '2,2': L}, r), 'LOSS_TO_LOSS_SWITCH')
        self.assertEqual(outcome('E1-C', {'1,1': C}, run((1, 1), (1, 1), [((1, 1), 'SAFE')], CHILDREN)), 'NO_VETO')
        self.assertEqual(outcome('E1-C', {'1,1': C, '2,2': C},
                                 run((1, 1), (2, 2), [((1, 1), 'UNSAFE'), ((2, 2), 'SAFE')], CHILDREN)), 'FALSE_VETO')


class SelectionTest(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(classify({'a': L, 'b': C}), 'E1-P')
        self.assertEqual(classify({'a': L, 'b': L}), 'E1-N')
        self.assertEqual(classify({'a': L, 'b': U}), 'E1-UNRESOLVED')

    def test_d4_hash(self):
        a = canonical_hash([[7, 7], [6, 8], [8, 9]])
        b = canonical_hash([[7, 7], [8, 6], [6, 5]])  # the same shape rotated by 180 degrees
        self.assertEqual(a, b)
        self.assertNotEqual(a, canonical_hash([[7, 7], [6, 8], [8, 10]]))

    def test_one_position_per_episode_and_controls(self):
        def row(pair, ply, status, moves):
            return {'source': 'h5_policy_8401.json', 'pair': pair, 'v8_color': 'white', 'ply': ply,
                    'history': moves, 'screen': {'status': status}}
        rows = [row(1, 11, PROVEN_LOSS, [[7, 7], [6, 8], [8, 9]]),
                row(1, 13, PROVEN_LOSS, [[7, 7], [6, 8], [8, 9], [5, 5], [9, 9]]),  # same episode
                row(2, 11, PROVEN_LOSS, [[7, 7], [8, 6], [6, 5]]),                   # D4 duplicate of pair 1
                row(3, 11, CLEAR, [[7, 7], [6, 6]])]
        chosen = select(rows)
        self.assertEqual([(r['pair'], r['ply']) for r in chosen['losses']], [(1, 11)])
        self.assertEqual(chosen['dropped'], {'same_episode': 1, 'd4_duplicate': 1})
        self.assertEqual([r['pair'] for r in chosen['controls']], [3])


if __name__ == '__main__':
    unittest.main()
