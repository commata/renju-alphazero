"""H7-R opening suites (hybrid.h7r_openings): the rules, legality, ids, symmetry, repeats and overlap."""
import unittest

from hybrid import h7r_openings as op
from renju import Game


class SuiteTest(unittest.TestCase):
    def test_rules_and_lengths(self):
        for name, count in (('O0', 60), ('O1', 60), ('O2', None)):
            body = op.build_suite(name, count, 8781)
            op.check_suite(body)
            self.assertEqual(body['next_to_move'], 'white')
            for o in body['openings']:
                self.assertEqual(len(o['moves']), op.SUITES[name]['moves'])
                self.assertEqual(o['moves'][0], [7, 7])
                g = Game()
                for m in o['moves']:
                    g.play(*m)                                  # every move legal, black forbidden points refused
                self.assertFalse(g.done)
                self.assertEqual(g.to_play, -1)                 # white to move
                radius = [2, 2] if name != 'O1' else [2, 2, 3, 3]
                for m, r in zip(o['moves'][1:], radius):
                    self.assertLessEqual(max(abs(m[0] - 7), abs(m[1] - 7)), r)
            self.assertEqual(body['distinct']['history_id'], body['count'])  # no exact repeat

    def test_o2_is_the_26_standard_openings_in_8_orientations(self):
        std = op.standard26()
        self.assertEqual(len(std), 26)
        self.assertEqual(sum(o[1] == [6, 7] for o in std), 13)  # direct
        self.assertEqual(sum(o[1] == [6, 8] for o in std), 13)  # indirect
        self.assertEqual(len({op.ids(o)['class_id'] for o in std}), 26)
        body = op.build_suite('O2', None, None)
        self.assertEqual(body['distinct']['class_id'], 26)
        self.assertEqual(body['count'], 184)  # 26 x 8 less the orientations that coincide
        with self.assertRaisesRegex(ValueError, 'no repeats'):
            op.build_suite('O2', 185, None)

    def test_seeded_and_reproducible(self):
        a, b = op.build_suite('O1', 30, 8782), op.build_suite('O1', 30, 8782)
        self.assertEqual(a['openings_sha256'], b['openings_sha256'])
        self.assertNotEqual(a['openings_sha256'], op.build_suite('O1', 30, 8783)['openings_sha256'])
        with self.assertRaises(ValueError):
            op.build_suite('O0', 600, 1, max_attempts=5000)  # only 552 ordered O0 openings exist

    def test_rejections(self):
        self.assertIn('centre', op.check_opening([[7, 8], [7, 7]]))
        self.assertIn('illegal', op.check_opening([[7, 7], [7, 7]]))
        # black has four in a row and is to move after white's reply: settled at once
        four = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [14, 14]]
        self.assertIn('settled', op.check_opening(four))
        self.assertIsNone(op.check_opening([[7, 7], [6, 7], [8, 8]]))

    def test_ids_symmetry_and_order(self):
        a = [[7, 7], [6, 7], [8, 8]]
        rotated = [list(op.symmetry(m, 1)) for m in a]
        ia, ir = op.ids(a), op.ids(rotated)
        self.assertNotEqual(ia['history_id'], ir['history_id'])
        self.assertEqual(ia['class_id'], ir['class_id'])
        swapped = [[7, 7], [6, 7], [8, 8], [5, 5], [9, 9]]
        other = [[7, 7], [5, 5], [9, 9], [6, 7], [8, 8]]
        self.assertEqual(op.ids(swapped)['position_id'], op.ids(other)['position_id'])  # same stones
        self.assertNotEqual(op.ids(swapped)['history_id'], op.ids(other)['history_id'])  # another history

    def test_check_suite_refuses_edits(self):
        body = op.build_suite('O0', 20, 8781)
        edited = {**body, 'openings': [dict(o) for o in body['openings']]}
        edited['openings'][1] = {**edited['openings'][1], 'moves': edited['openings'][0]['moves'],
                                 **op.ids(edited['openings'][0]['moves'])}
        with self.assertRaisesRegex(ValueError, 'repeats'):
            op.check_suite(edited)
        with self.assertRaises(ValueError):
            op.check_suite({**body, 'openings_sha256': '0' * 64})

    def test_pick_and_overlap(self):
        body = op.build_suite('O0', 50, 8781)
        chosen = op.pick(body, 20, 8784)
        self.assertEqual(len({o['history_id'] for o in chosen}), 20)
        self.assertEqual(chosen, op.pick(body, 20, 8784))
        with self.assertRaises(ValueError):
            op.pick(body, 51, 1)
        previous = {'earlier': [chosen[0]['moves'], [list(op.symmetry(m, 2)) for m in chosen[1]['moves']]]}
        out = op.overlap(chosen, previous)['earlier']
        self.assertEqual(out['history_id'], 1)
        self.assertGreaterEqual(out['class_id'], 2)


if __name__ == '__main__':
    unittest.main()
