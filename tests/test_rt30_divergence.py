"""scripts/run_rt30_divergence.py and the A3 manifests it replays (docs/web-ser5.md §9.5)."""
import json
from pathlib import Path
import unittest

from scripts import run_rt30_divergence as dv

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [ROOT / 'docs' / 'rt30-eval' / name for name in ('a3-divergence-v1.json', 'a3-root-control-v1.json')]


class ManifestTest(unittest.TestCase):
    def test_cases_point_at_tb1_moves_of_the_records(self):
        for path in MANIFESTS:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['keys'], [c['key'] for c in manifest['cases']])
            for case in manifest['cases']:
                record = dv.record_of(case['key'])
                ply = int(case['key'].split('/')[-1])
                move = next(m for m in record['v8_moves'] if m['ply'] == ply)  # an engine move
                self.assertEqual(record['moves'][ply], case['tb1_move'], case['key'])
                self.assertEqual(f"{case['key'].split('_')[-1].split('/')[0]}/{record['pair']}/{record['v8_color']}",
                                 case['game'])
                self.assertEqual(move['route'], case['tb1_route'])

    def test_divergence_set(self):
        manifest = json.loads(MANIFESTS[0].read_text(encoding='utf-8'))
        kinds = [c['kind'] for c in manifest['cases']]
        self.assertEqual((kinds.count('attack'), kinds.count('root')), (6, 28))
        self.assertEqual(sum(dv.delta(c['result']) for c in manifest['cases']), -6.0)  # A3: 0.75 - 0.81


class SummaryTest(unittest.TestCase):
    def row(self, kind, result, recovered):
        return {'kind': kind, 'result': result, 'recovered': recovered, 'prefix_mismatches': [], 'tree_move': None,
                'tree_move_rec': None, 'seconds': 10.0, 'phases': {'attack': 1.0, 'root': 2.0}}

    def test_points(self):
        rows = [self.row('attack', 'win>loss', True), self.row('attack', 'win>draw', False),
                self.row('root', 'draw>win', True), self.row('root', 'win>win', True)]
        s = dv.summarize(rows)
        self.assertEqual(s['by_kind']['attack'], {'cases': 2, 'recovered': 1, 'a3_points': -1.5,
                                                  'points_if_recovered': 1.0})
        self.assertEqual(s['loss_making']['points_if_recovered'], 1.0)
        self.assertEqual(s['gain_making']['points_if_recovered'], -0.5)  # recovering TB1's move gives the luck back
        self.assertEqual(s['result_changed']['cases'], 3)


if __name__ == '__main__':
    unittest.main()
