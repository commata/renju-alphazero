"""Sealed Final-1 / Final-2 generator (scripts/make_final_sets.py)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from renju import Game
from scripts.make_final_sets import build, canonical_hash, known_openings, main

SMALL = {'a': 6, 'b': 6}
QUOTA = {'immediate_win': 2, 'must_block': 2, 'black_forbidden': 1}


class FinalSetsTest(unittest.TestCase):
    def test_deterministic_disjoint_and_excluding_known_openings(self):
        sets, info = build('s1', counts=SMALL, quota=QUOTA)
        again, _ = build('s1', counts=SMALL, quota=QUOTA)
        self.assertEqual(sets, again)
        self.assertNotEqual(build('s2', counts=SMALL, quota=QUOTA)[0], sets)
        known = known_openings(Path(__file__).resolve().parents[1])
        self.assertGreater(info['known_benchmark_openings_excluded'], 0)
        keys = [canonical_hash(o['moves']) for name in SMALL for o in sets[name]['items']]
        self.assertEqual(len(keys), len(set(keys)))      # no D4 duplicate across or within sets
        self.assertFalse(set(keys) & known)              # none seen in any benchmark
        for item in sets['final1_tactical']['items']:
            game = Game()
            for m in item['moves']:
                game.play(*m)                            # every position replays legally
            self.assertIn(item['kind'], QUOTA)
        kinds = [i['kind'] for i in sets['final1_tactical']['items']]
        self.assertEqual({k: kinds.count(k) for k in QUOTA}, QUOTA)

    def test_sets_never_go_into_the_repo(self):
        with self.assertRaises(SystemExit):
            main(['--out-dir', 'runs/finals', '--manifest', 'x.json'])


if __name__ == '__main__':
    unittest.main()
