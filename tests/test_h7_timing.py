"""scripts/h7_timing.py: positions, measure (frozen TB-V2-Value), corpus rules and the equivalence compare."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.tb_v2_value import VALUE_CHECKPOINT_SHA256
from scripts import h7_timing as t
from tests.tb_engine_v1_regression import WEB, hashed_policy

MOVES = json.loads(WEB.read_text(encoding='utf-8'))['moves']


class Leaf:
    """Torch-free stand-in for ValueLeaf (the frozen checkpoint's sha, a deterministic value)."""
    sha256 = VALUE_CHECKPOINT_SHA256

    def __init__(self):
        self._cache = {}

    def __call__(self, game):
        return ((len(game.history) * 7) % 11 - 5) / 10


FACTORIES = {'policy': hashed_policy, 'leaf': Leaf()}


def e3_run():
    move = {'route': 'tree', 'seconds': 1.0}
    return {'arm': t.CANDIDATE_ARM, 'games': [{'key': 'k/8407/0/black', 'moves': MOVES[:30],
                                               'v8_moves': [{**move, 'ply': p} for p in (8, 20, 28)]}]}


class PositionsTest(unittest.TestCase):
    def test_sources(self):
        positions = t.build_positions([e3_run()])
        counts = {s: sum(p['source'] == s for p in positions) for s in ('e3a', 'probe', 'h6_4b')}
        self.assertEqual(counts, {'e3a': 3, 'probe': 3, 'h6_4b': 298})
        self.assertEqual(positions[0]['moves'], MOVES[:8])
        with self.assertRaises(ValueError):
            t.build_positions([{**e3_run(), 'arm': 'puct_policy_vct2_50k_tt'}])


class EndToEndTest(unittest.TestCase):
    def test_measure_corpus_compare(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / 'run.json'
            run.write_text(json.dumps(e3_run()), encoding='utf-8')
            pos = tmp / 'positions.json'
            self.assertEqual(t.main(['positions', '--runs', str(run), '--output', str(pos)]), 0)
            outs = []
            for name in ('a', 'b'):
                out = tmp / f'{name}.json'
                self.assertEqual(t.main(['measure', '--positions', str(pos), '--limit', '3', '--output', str(out)],
                                        factories=FACTORIES), 0)
                outs.append(out)
            a = json.loads(outs[0].read_text(encoding='utf-8'))
            self.assertEqual(len(a['rows']), 3)
            self.assertEqual(a['summary']['all']['n'], 3)
            self.assertTrue(all(r['decision']['route'] for r in a['rows']))
            corpus = tmp / 'corpus.json'
            self.assertEqual(t.main(['corpus', '--measure', str(outs[0]), '--output', str(corpus)]), 0)
            c = json.loads(corpus.read_text(encoding='utf-8'))
            self.assertEqual(c['positions'], 3)  # every e3a position is in a group or the random sample
            for mode in ('search', 'value'):  # the same frozen engine twice: identical decisions
                res = tmp / f'cmp_{mode}.json'
                self.assertEqual(t.main(['compare', '--mode', mode, '--corpus', str(corpus), '--baseline', str(outs[0]),
                                         '--candidate', str(outs[1]), '--output', str(res)]), 0)
                self.assertEqual(json.loads(res.read_text(encoding='utf-8'))['verdict'], 'PASS')
            b = json.loads(outs[1].read_text(encoding='utf-8'))
            b['rows'][0]['decision']['root_nodes'] += 1  # a search-layer difference with the same move
            bad = tmp / 'bad.json'
            bad.write_text(json.dumps(b), encoding='utf-8')
            self.assertEqual(t.compare(c, a, b, 'search')['verdict'], 'FAIL')
            self.assertEqual(t.compare(c, a, b, 'value')['verdict'], 'PASS')
            b['rows'][0]['move'] = [0, 0]
            self.assertEqual(t.compare(c, a, b, 'value')['verdict'], 'FAIL')


class CorpusRulesTest(unittest.TestCase):
    def row(self, visits, checked='SAFE', seconds=1.0, source='e3a', switch=None, vct2=False):
        return {'id': 'x', 'source': source, 'seconds': seconds,
                'decision': {'route': 'tree', 'tree_move': [7, 7], 'changed': False, 'root_switch': switch,
                             'root_visits': visits, 'root_checked': [[[7, 7], checked]], 'vct2_switched': vct2}}

    def test_groups(self):
        far = [[[7, 7], 40, 0.5], [[7, 8], 10, 0.1]]
        self.assertEqual(t.groups_of(self.row(far), 9.0), [])
        self.assertEqual(t.groups_of(self.row([[[7, 7], 40, 0.5], [[7, 8], 37, 0.1]]), 9.0), ['near_tie'])
        self.assertEqual(t.groups_of(self.row([[[7, 7], 40, 0.5], [[7, 8], 10, 0.49]]), 9.0), ['near_tie'])
        self.assertEqual(t.groups_of(self.row(far, checked='UNKNOWN'), 9.0), ['v8c_not_safe_or_switch'])
        self.assertEqual(t.groups_of(self.row(far, vct2=True, seconds=10.0, source='probe'), 9.0),
                         ['vct2_switch', 'probe', 'slow'])


if __name__ == '__main__':
    unittest.main()
