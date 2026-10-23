"""scripts/h7_e0_vcf_transpositions.py: the instrumented VCF equals the frozen one; duplicate-work summary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import h6_5d
from scripts import h7_e0_vcf_transpositions as e0

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))


class InstrumentTest(unittest.TestCase):
    def test_same_results_and_counts(self):
        for ply in (40, 92, 94):
            r = e0.measure(WEB['moves'][:ply], WEB['moves'][ply], h6_5d.STANDARD)
            self.assertTrue(r['identical'], ply)
            self.assertEqual(r['duplicates'], sum(r['duplicates_by_first'].values()))
            self.assertLessEqual(r['reusable_nodes'], r['nodes'])
            self.assertEqual(r['entries'], r['unique_states'] + r['duplicates'])

    def test_direct_vcf_equals_frozen(self):
        from search.mcts_v7 import _find_vcf_with_stats
        game = h6_5d.replay(WEB['moves'][:95])
        for attacker in (1, -1):
            for limit in (50, 5000):
                want = _find_vcf_with_stats(game, attacker, max_fours=10, node_limit=limit)
                got = e0.instrumented_find_vcf_with_stats(e0.Recorder())(game, attacker, max_fours=10, node_limit=limit)
                self.assertEqual((got[1], got[2]), (want[1], want[2]))
                self.assertEqual(got[0], want[0])


def row(i, group, game, share, control_for=None, identical=True):
    m = {'status': 'UNKNOWN', 'identical': identical, 'vcf_calls': 1, 'nodes': 100, 'entries': 10, 'unique_states': 8,
         'unique_states_check': 8, 'duplicates': 2, 'duplicates_check': 2, 'duplicate_rate': 0.2,
         'reusable_nodes': int(share * 100), 'reusable_nodes_check': int(share * 100), 'reusable_share': share,
         'reusable_share_check': share, 'duplicates_by_first': {'win': 0, 'no_win': 2, 'cut': 0},
         'duplicate_remaining': None, 'seconds': 0.1}
    return {'id': f'{group}{i}', 'arm': h6_5d.CANDIDATE_ARM, 'game': game, 'group': group,
            'control_for': control_for, 'standard': m}


class DecideTest(unittest.TestCase):
    def build(self, target_share, control_share, n=30, identical=True):
        rows = []
        for i in range(n):
            rows.append(row(i, 'unknown4m', f'g{i}', target_share, identical=identical))
            rows.append(row(i, 'safe_control', f'g{i}', control_share, control_for=f'unknown4m{i}'))
        return e0.summarize(rows)['decision']['verdict']

    def test_verdicts(self):
        self.assertEqual(self.build(0.5, 0.0), 'strong_go')
        self.assertEqual(self.build(0.15, 0.0), 'weak_go')
        self.assertEqual(self.build(0.05, 0.0), 'no_go')
        self.assertEqual(self.build(0.5, 0.5), 'no_go')  # not above the controls
        self.assertEqual(self.build(0.5, 0.0, identical=False), 'tool_error')


class MainTest(unittest.TestCase):
    def test_end_to_end(self):
        def tree_move(ply, status):
            move = list(WEB['moves'][ply])
            return {'route': 'tree', 'ply': ply, 'v7_move': move, 'played': move, 'root': {'checked': [[move, status]]}}

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths, d2 = [], {'rows': []}
            for arm in (h6_5d.CANDIDATE_ARM, h6_5d.BASELINE_ARM):
                key = f'{arm}[c]@v8:full/8404/0/black'
                r = {'arm': arm, 'seed': 8404, 'opponent': 'v8:full', 'games': [
                    {'key': key, 'result': 'loss', 'length': 100, 'moves': WEB['moves'][:100],
                     'v8_moves': [tree_move(88, 'SAFE'), tree_move(94, 'UNKNOWN')]}]}
                path = tmp / f'{arm}.json'
                path.write_text(json.dumps(r), encoding='utf-8')
                paths.append(path)
                d2['rows'].append({'id': f'{arm}|{key}|94', 'large': {'status': 'UNKNOWN'}})
            (tmp / 'd2.json').write_text(json.dumps(d2), encoding='utf-8')
            out = tmp / 'e0.json'
            self.assertEqual(e0.main(['--runs', *map(str, paths), '--d2', str(tmp / 'd2.json'), '--output', str(out)]), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            self.assertTrue(result['summary']['identical'])
            self.assertEqual(len(result['rows']), 4)  # two unknown4m + two matched controls
            self.assertEqual(result['summary']['groups']['pooled']['unknown4m']['standard']['positions'], 2)


if __name__ == '__main__':
    unittest.main()
