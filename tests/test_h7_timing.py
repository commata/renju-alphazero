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
            first = a['rows'][0]['id']
            e = t.explain(a, b, [first])['positions'][first]
            self.assertEqual(e['differing_fields'], ['root_nodes'])
            self.assertTrue(e['same_root_children'])
            self.assertEqual(e['max_abs_q_diff_common_children'] or 0.0, 0.0)


class CorpusRulesTest(unittest.TestCase):
    def row(self, visits, checked='SAFE', seconds=1.0, source='e3a', switch=None, vct2=False):
        return {'id': 'x', 'source': source, 'seconds': seconds,
                'decision': {'route': 'tree', 'tree_move': [7, 7], 'changed': False, 'root_switch': switch,
                             'root_visits': visits, 'root_checked': [[[7, 7], checked]], 'vct2_switched': vct2}}

    def test_measure_tb_v3(self):
        from rt30 import fast4_forbidden
        from analysis import tb_v3, tb_v3_runtime
        if not fast4_forbidden.available():
            self.skipTest('python -m rt30.native.build first')
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / 'run.json'
            run.write_text(json.dumps(e3_run()), encoding='utf-8')
            pos = tmp / 'positions.json'
            self.assertEqual(t.main(['positions', '--runs', str(run), '--output', str(pos)]), 0)
            argv = ['measure', '--positions', str(pos), '--limit', '2', '--engine', 'tb-v3']
            with self.assertRaises(SystemExit):  # fast4 only
                t.main(argv + ['--output', str(tmp / 'x.json')], factories={**FACTORIES, 'stand_in': True})
            jsonl = tmp / 'v3.jsonl'
            out = tmp / 'v3.json'
            self.assertEqual(t.main(argv + ['--fast', 'fast4', '--jsonl', str(jsonl), '--output', str(out)],
                                    factories={**FACTORIES, 'stand_in': True}), 0)
            r = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual((r['engine'], r['engine_name']), (tb_v3.ENGINE_HASH, 'TB-V3 Resolve'))
            self.assertEqual(r['runtime_hash'], tb_v3_runtime.runtime_hash('cpu'))
            self.assertEqual(set(r['ser5_check']), {'over_deadline', 'max', 'pass'})
            self.assertTrue(all(row['engine'] == 'tb-v3' and 'e4_attack' in row for row in r['rows']))
            with self.assertRaises(ValueError):  # a TB-V3 jsonl is not resumed as TB-V2-Value
                t.main(['measure', '--positions', str(pos), '--limit', '2', '--jsonl', str(jsonl),
                        '--output', str(tmp / 'y.json')], factories=FACTORIES)
        from hybrid.h6_fast import installed_mode
        self.assertEqual(installed_mode(), 'off')

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
