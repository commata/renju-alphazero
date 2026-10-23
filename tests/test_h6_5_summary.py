"""scripts/summarize_h6_5.py: the H6-5 decision table and guards, fixed before the games."""
import unittest

from scripts.summarize_h6_5 import decide


def row(seeds, score=0.6, low=0.52, black=0.6, white=0.6, not_safe=0.05, time_ratio=1.0, violations=()):
    return {'seeds': seeds, 'score': score, 'ci95_pairs': [low, 0.7], 'by_colour': {'black': black, 'white': white},
            'tree_not_safe_rate': not_safe, 'time_ratio': time_ratio, 'safety_violations': list(violations)}


class DecideTest(unittest.TestCase):
    def run_case(self, seeds, p92='PASS', aux_not_safe=0.05, base_not_safe=0.05, **main):
        return decide(row(seeds, **main), row(seeds, not_safe=aux_not_safe), row(seeds, not_safe=base_not_safe),
                      p92)['decision']

    def test_stages(self):
        self.assertEqual(self.run_case([8403], score=0.5), 'RUN_8404')
        self.assertEqual(self.run_case([8403], score=0.49), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], score=0.55, low=0.51), 'ADOPT')
        self.assertEqual(self.run_case([8403, 8404], score=0.56, low=0.5), 'INCONCLUSIVE')
        self.assertEqual(self.run_case([8403, 8404], score=0.54, low=0.51), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404, 8405], score=0.56, low=0.5), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404, 8405], score=0.56, low=0.51), 'ADOPT')
        self.assertEqual(self.run_case([8404]), 'INCOMPLETE')

    def test_guards(self):
        self.assertEqual(self.run_case([8403, 8404], violations=[{'key': 'k'}]), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], aux_not_safe=0.09, base_not_safe=0.05), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], aux_not_safe=0.08, base_not_safe=0.05), 'ADOPT')
        self.assertEqual(self.run_case([8403, 8404], time_ratio=1.51), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], black=0.44), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], p92='FAIL'), 'REJECT')
        self.assertEqual(self.run_case([8403, 8404], p92='NOT_DISCRIMINATING'), 'ADOPT')
        self.assertEqual(self.run_case([8403], p92=None), 'INCOMPLETE')
        self.assertEqual(decide(row([8403]), None, None, 'PASS')['decision'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
