"""E1-holdout tooling (§12.34): holdout screen sources and the paired verdict."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.e1_build_suite import CLEAR, PROVEN_LOSS, UNKNOWN, canonical_hash, screen_items, select
from scripts.e1_holdout_judge import judge


def run_file(tmp, arm, games):
    path = Path(tmp, f'{arm}.json')
    path.write_text(json.dumps({'arm': arm, 'seed': 8417, 'games': games}), encoding='utf-8')
    return path


def game(pair, moves, plies):
    return {'pair': pair, 'v8_color': 'white', 'result': 'loss', 'moves': moves,
            'v8_moves': [{'route': 'tree', 'ply': p, 'played': moves[p], 'root': {'checked': []}} for p in plies]}


MOVES = [[7, 7], [6, 8], [8, 9], [5, 5], [9, 9], [4, 4]]


class HoldoutScreenTest(unittest.TestCase):
    def test_keys_dedupe_and_exclusion(self):
        with TemporaryDirectory() as tmp:
            other = MOVES[:5] + [[3, 3]]                      # E3-A played another move at ply 5
            s3 = run_file(tmp, 'puct_policy_vct2', [game(0, MOVES, [3, 5])])
            e3a = run_file(tmp, 'puct_policy_vct2_50k', [game(0, other, [3, 5])])
            items = screen_items([s3, e3a])
            self.assertEqual([i['key'] for i in items],
                             ['s3:8417-p0-white/3', 's3:8417-p0-white/5', 'e3a:8417-p0-white/5'])
            self.assertEqual(items[0]['also_in'], ['e3a:8417-p0-white/3'])  # same position, same move
            self.assertEqual(screen_items.dropped, {'excluded_d4': 0, 'same_move_other_arm': 1})
            excluded = screen_items([s3, e3a], {canonical_hash(MOVES[:3])})
            self.assertEqual([i['key'] for i in excluded], ['s3:8417-p0-white/5', 'e3a:8417-p0-white/5'])
            self.assertEqual(screen_items.dropped['excluded_d4'], 2)  # both arms' copies
            with self.assertRaises(ValueError):
                screen_items([run_file(tmp, 'puct_policy', [])])  # dev baseline logs are not holdout sources

    def test_select_orders_holdout_sources(self):
        rows = [{'source': 'b.json', 'pair': 0, 'v8_color': 'white', 'ply': 3, 'history': MOVES[:3],
                 'screen': {'status': PROVEN_LOSS}},
                {'source': 'a.json', 'pair': 0, 'v8_color': 'white', 'ply': 3, 'history': MOVES[:3],
                 'screen': {'status': PROVEN_LOSS}}]
        chosen = select(rows, ['a.json', 'b.json'])
        self.assertEqual([r['source'] for r in chosen['losses']], ['a.json'])


def evaluation(arm, runs):
    rows = {}
    for key, cls, seed, outcome, pre, played, checked, switched in runs:
        rows.setdefault(key, {'key': key, 'class': cls, 'arm': arm, 'truths_added': {}, 'runs': []})['runs'].append(
            {'seed': seed, 'outcome': outcome, 'pre': pre, 'played': played, 'switched': switched,
             'checked': [[m, s] for m, s in checked]})
    return {'arm': arm, 'rows': list(rows.values()), 'summary': {'cost': {}}}


TRUTHS = {'1,1': {'status': PROVEN_LOSS}, '2,2': {'status': CLEAR}, '3,3': {'status': PROVEN_LOSS},
          '4,4': {'status': UNKNOWN}}
MANIFEST = {'split': 'holdout', 'positions': [
    {'key': 'p', 'class': 'E1-P', 'ply': 11, 'source_arm': 'puct_policy_vct2', 'history': [], 'truths': TRUTHS},
    {'key': 'c', 'class': 'E1-C', 'ply': 30, 'source_arm': 'puct_policy_vct2_50k', 'history': [],
     'truths': {'2,2': {'status': CLEAR}, '3,3': {'status': PROVEN_LOSS}}}]}


class JudgeTest(unittest.TestCase):
    def s3(self):
        return evaluation('s3', [
            ('p', 'E1-P', 1, 'DETECT_MISS', [1, 1], [1, 1], [([1, 1], 'UNKNOWN')], False),
            ('p', 'E1-P', 2, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('c', 'E1-C', 1, 'NO_VETO', [2, 2], [2, 2], [([2, 2], 'SAFE')], False)])

    def test_pass_small_sample(self):
        e3a = evaluation('e3a', [
            ('p', 'E1-P', 1, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('p', 'E1-P', 2, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('c', 'E1-C', 1, 'NO_VETO', [2, 2], [2, 2], [([2, 2], 'SAFE')], False)])
        out = judge(MANIFEST, self.s3(), e3a, witness=lambda h, m: True)
        self.assertEqual((out['verdict'], out['tb_engine_v1'], out['gates']['G5']), ('PASS', 'E3-A', 'N/A'))
        self.assertEqual(out['veto_recall'], {'s3': [1, 2], 'e3a': [2, 2]})

    def test_regressions_fail(self):
        lost_rescue = evaluation('e3a', [                       # G2/G3: the S3 veto and rescue are lost
            ('p', 'E1-P', 1, 'DETECT_MISS', [1, 1], [1, 1], [([1, 1], 'UNKNOWN')], False),
            ('p', 'E1-P', 2, 'DETECT_MISS', [1, 1], [1, 1], [([1, 1], 'UNKNOWN')], False),
            ('c', 'E1-C', 1, 'NO_VETO', [2, 2], [2, 2], [([2, 2], 'SAFE')], False)])
        out = judge(MANIFEST, self.s3(), lost_rescue)
        self.assertEqual((out['verdict'], out['bug_suspected']), ('FAIL', True))
        self.assertEqual(out['g2_violations'], [['p', 2]])
        false_veto = evaluation('e3a', [                       # G1: a vetoed move that is VCT2_CLEAR
            ('p', 'E1-P', 1, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('p', 'E1-P', 2, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('c', 'E1-C', 1, 'FALSE_VETO', [2, 2], [3, 3], [([2, 2], 'UNSAFE'), ([3, 3], 'UNKNOWN')], True)])
        out = judge(MANIFEST, self.s3(), false_veto)
        self.assertFalse(out['gates']['G1'])
        self.assertEqual(out['loss_switches'], [['c', 1]])
        unsound = judge(MANIFEST, self.s3(), evaluation('e3a', [
            ('p', 'E1-P', 1, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('p', 'E1-P', 2, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE'), ([2, 2], 'SAFE')], True),
            ('c', 'E1-C', 1, 'NO_VETO', [2, 2], [2, 2], [([2, 2], 'SAFE')], False)]), witness=lambda h, m: False)
        self.assertEqual((unsound['verdict'], len(unsound['unsound_witness'])), ('FAIL', 2))

    def test_runs_must_match(self):
        with self.assertRaises(ValueError):
            judge(MANIFEST, self.s3(), evaluation('e3a', [
                ('p', 'E1-P', 1, 'RESCUED', [1, 1], [2, 2], [([1, 1], 'UNSAFE')], True)]))


class WitnessTest(unittest.TestCase):
    def test_e1_dev_proof_is_reproduced(self):
        # E1-dev 8411-p8 (7,8): a 50k selective proof (11,320 nodes, §12.30) verified by the full solver.
        from scripts.e1_holdout_judge import witness_ok
        manifest = json.loads(Path('docs/mcts-v8-results/e1/e1_manifest.json').read_text(encoding='utf-8'))
        history = next(p for p in manifest['positions'] if p['key'] == '8411-p8-white/11')['history']
        budget = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 50_000}
        self.assertTrue(witness_ok(history, [7, 8], budget))
        self.assertFalse(witness_ok(history, [7, 8], {**budget, 'node_budget': 10_000}))  # not reproduced at 10k


if __name__ == '__main__':
    unittest.main()
