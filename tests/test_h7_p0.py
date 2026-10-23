"""scripts/h7_p0_profile.py: transparent boundary timers, shares and the P1 branch rule."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import h7_p0_profile as p0
from scripts import h7_timing
from tests.test_h7_timing import FACTORIES, MOVES


def positions():
    return [{'id': f'e3a|k|{ply}', 'source': 'e3a', 'moves': MOVES[:ply]} for ply in (20, 92, 94)]


class TimersTest(unittest.TestCase):
    def setUp(self):
        h7_timing._init(None, None, 'cpu', 1, 'off', FACTORIES)

    def test_transparent_and_counted(self):
        from scripts.run_mcts_v8_benchmark import derive_seed
        from analysis import mcts_v8, vcf_tt_v1
        before = (mcts_v8._verify_root_with, vcf_tt_v1.make_find, vcf_tt_v1._key, mcts_v8._BudgetedSolver.vcf)
        for task in positions():
            plain = h7_timing.decide(task['moves'], derive_seed(h7_timing.SEED, 'h7-timing', task['id']))
            row = p0.p0_task({**task, 'groups': ['slow']})
            self.assertTrue(row['identical'])
            self.assertEqual(row['coarse']['decision'], plain['decision'])
            c = row['coarse']
            if plain['decision']['route'] == 'tree':
                self.assertGreater(c['root_seconds'], 0)
                self.assertLessEqual(c['vcf_seconds'], c['root_seconds'])
                self.assertEqual(c['calls'], plain['decision']['root_calls'])
                self.assertEqual(c['nodes'], plain['decision']['root_nodes'])
                self.assertEqual(c['requests'] - c['cache_hits'], c['calls'])
                self.assertTrue(any(k.startswith('inside:') for k in row['fine']['parts']))
        after = (mcts_v8._verify_root_with, vcf_tt_v1.make_find, vcf_tt_v1._key, mcts_v8._BudgetedSolver.vcf)
        self.assertEqual(before, after)  # every timer removed


def row(share, groups=('slow',)):
    c = {'root_seconds': 10.0, 'vcf_seconds': 10.0 * share, 'vcf_share': share, 'requests': 2, 'calls': 1,
         'cache_hits': 1, 'nodes': 5, 'cut_calls': 0, 'table': {'hits': 1, 'stores': 2}, 'tree_status': 'SAFE',
         'decision': {}}
    return {'id': str(share), 'groups': list(groups), 'coarse': c, 'fine': {**c, 'parts': {}}, 'identical': True}


class SummaryTest(unittest.TestCase):
    def test_branch(self):
        self.assertEqual(p0.summarize([row(0.7), row(0.6)], None)['decision']['branch'], 'P1a')
        out = p0.summarize([row(0.9), row(0.2), row(0.2)], None)
        self.assertEqual(out['decision']['branch'], 'P1b')  # aggregate 0.433
        self.assertEqual(out['pooled']['median_share'], 0.2)

    def test_select(self):
        corpus = {'ids': {'a': ['slow'], 'b': ['near_tie'], 'c': ['random'], 'd': ['v8c_not_safe_or_switch', 'slow']}}
        pos = [{'id': i, 'source': 'e3a', 'moves': []} for i in 'abcd']
        got = {p['id']: p['groups'] for p in p0.select(pos, corpus)}
        self.assertEqual(got, {'a': ['slow'], 'c': ['random'], 'd': ['slow', 'v8c_not_safe_or_switch']})


class MainTest(unittest.TestCase):
    def test_end_to_end(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'pos.json').write_text(json.dumps({'positions': positions()}), encoding='utf-8')
            (tmp / 'corpus.json').write_text(json.dumps({'ids': {p['id']: ['slow'] for p in positions()}}),
                                             encoding='utf-8')
            out = tmp / 'p0.json'
            self.assertEqual(p0.main(['--positions', str(tmp / 'pos.json'), '--corpus', str(tmp / 'corpus.json'),
                                      '--output', str(out)], factories=FACTORIES), 0)
            s = json.loads(out.read_text(encoding='utf-8'))['summary']
            self.assertTrue(s['timers_transparent'])
            self.assertIn(s['decision']['branch'], ('P1a', 'P1b'))
            self.assertIn('outside:solver_rest', s['fine'])


if __name__ == '__main__':
    unittest.main()
