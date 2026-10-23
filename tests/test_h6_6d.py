"""scripts/h6_6d.py: punishment of 4M-UNKNOWN tree moves (loss within 12 plies, V8 and VCT2 proven wins)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import h6_6d as d6

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))


def tree_move(ply, status, played=None, moves=WEB['moves']):
    move = list(moves[ply])
    return {'route': 'tree', 'ply': ply, 'v7_move': move, 'played': played or move,
            'root': {'checked': [[move, status]]}}


def run(arm, seed, games):
    return {'arm': arm, 'seed': seed, 'opponent': 'v8:full', 'games': [
        {'key': f'{arm}[c]@v8:full/{seed}/{i}/black', 'result': result, 'length': length, 'moves': WEB['moves'][:length],
         'v8_moves': moves} for i, (result, length, moves) in enumerate(games)]}


class AttackTest(unittest.TestCase):
    def test_vct2_finds_what_v8b_misses(self):
        """The web loss: after black's (3,4) at ply 92 white wins at depth 2, which V8-B does not see."""
        lost = d6.attack_task({'id': 'x', 'moves': WEB['moves'][:92], 'tree_move': WEB['moves'][92]})
        self.assertEqual((lost['v8b'], lost['vct2'], lost['v8_win'], lost['vct2_win']), (False, True, False, True))
        quiet = d6.attack_task({'id': 'y', 'moves': WEB['moves'][:20], 'tree_move': WEB['moves'][20]})
        self.assertFalse(quiet['vct2_win'])


class TreeMoveTest(unittest.TestCase):
    def test_groups_loss12_and_switched(self):
        cand = run(d6.CANDIDATE, 8404, [('loss', 100, [tree_move(80, 'SAFE'), tree_move(90, 'UNKNOWN'),
                                                       tree_move(92, 'UNKNOWN', played=[0, 0])])])
        d2 = {'rows': [{'id': f"{d6.CANDIDATE}|{cand['games'][0]['key']}|{p}", 'large': {'status': s}}
                       for p, s in ((90, 'UNKNOWN'), (92, 'SAFE'))]}
        rows = {r['ply']: r for r in d6.tree_moves([cand], [d2])}
        self.assertEqual((rows[80]['group'], rows[90]['group'], rows[92]['group']),
                         ('safe', 'unknown4m', 'resolved_safe'))
        self.assertEqual((rows[80]['loss12'], rows[90]['loss12']), (False, True))  # last move index 99
        self.assertTrue(rows[92]['switched'])
        self.assertIsNone(rows[92]['loss12'])
        with self.assertRaises(ValueError):
            d6.tree_moves([cand], [{'rows': []}])


def row(i, group, game, loss, v8, vct2, arm=d6.CANDIDATE):
    return {'id': f'{group}{i}', 'arm': arm, 'game': game, 'group': group, 'switched': False, 'loss12': loss,
            '_v8': v8, '_vct2': vct2}


def attacks_for(rows):
    return {r['id']: {'done': False, 'v8_win': r['_v8'], 'vct2_win': r['_vct2']} for r in rows}


class ReadingTest(unittest.TestCase):
    def build(self, loss, vct2, n=30):
        targets = [row(i, 'unknown4m', f'g{i}', loss(i), False, vct2(i)) for i in range(n)]
        controls = [row(i, 'safe', f'g{i}', False, False, False) for i in range(n)]
        rows = targets + controls
        return d6.summarize(rows, attacks_for(rows), {t['id']: c['id'] for t, c in zip(targets, controls)})

    def test_danger_proof_hard_mixed(self):
        self.assertEqual(self.build(lambda i: True, lambda i: True)['reading'], 'danger')
        self.assertEqual(self.build(lambda i: False, lambda i: False)['reading'], 'proof_hard')
        self.assertEqual(self.build(lambda i: True, lambda i: False)['reading'], 'mixed')
        self.assertEqual(self.build(lambda i: True, lambda i: True, n=10)['reading'], 'mixed')  # too few pairs

    def test_rates_by_group(self):
        out = self.build(lambda i: i < 15, lambda i: False)
        groups = out['arms'][d6.CANDIDATE]['groups']
        self.assertEqual(groups['unknown4m']['loss12']['mean'], 0.5)
        self.assertEqual(groups['safe']['loss12']['mean'], 0.0)
        self.assertEqual(out['arms'][d6.CANDIDATE]['unknown4m_vs_control']['pairs'], 30)


class MainTest(unittest.TestCase):
    def test_end_to_end(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = []
            for arm in (d6.CANDIDATE, d6.BASELINE):
                r = run(arm, 8404, [('loss', 100, [tree_move(20, 'SAFE'), tree_move(88, 'SAFE'), tree_move(92, 'UNKNOWN')])])
                path = tmp / f'{arm}.json'
                path.write_text(json.dumps(r), encoding='utf-8')
                files.append((path, r))
            d2 = {'rows': [{'id': f"{r['arm']}|{r['games'][0]['key']}|92", 'recorded_status': 'UNKNOWN',
                            'standard': {'status': 'UNKNOWN'}, 'large': {'status': 'UNKNOWN'}} for _, r in files]}
            (tmp / 'd2.json').write_text(json.dumps(d2), encoding='utf-8')
            out = tmp / 'h6_6d.json'
            self.assertEqual(d6.main(['--runs', *[str(p) for p, _ in files], '--d2', str(tmp / 'd2.json'),
                                      '--output', str(out)]), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            self.assertTrue(result['report_only'])
            self.assertEqual(len(result['attacks']), 4)  # two targets + their controls (ply 88)
            cand = result['summary']['arms'][d6.CANDIDATE]
            self.assertEqual(cand['groups']['unknown4m']['vct2_win']['mean'], 1.0)
            self.assertEqual(cand['unknown4m_vs_control']['pairs'], 1)
            self.assertEqual(result['summary']['reading'], 'mixed')  # one pair is too few


if __name__ == '__main__':
    unittest.main()
