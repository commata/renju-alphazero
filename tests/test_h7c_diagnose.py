"""H7-c diagnosis (scripts/h7c_diagnose.py): measures, input checks, and one end-to-end run on stand-in models."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from renju import Game
from rt30 import fast5
from scripts import h7c_arena as ar
from scripts import h7c_diagnose as dg


class MeasureTest(unittest.TestCase):
    def test_distributions(self):
        a, b = (7, 7), (7, 8)
        self.assertEqual(dg.restricted({a: 0.0, b: 0.0}, [a, b]), {a: 0.5, b: 0.5})  # the PUCT fallback
        r = dg.restricted({a: 0.3, b: 0.1, (0, 0): 0.6}, [a, b])
        self.assertAlmostEqual(r[a], 0.75)
        self.assertAlmostEqual(r[b], 0.25)
        self.assertAlmostEqual(dg.entropy({a: 0.5, b: 0.5}), math.log(2))
        p, q = {a: 0.9, b: 0.1}, {a: 0.5, b: 0.5}
        self.assertGreater(dg.kl(p, q), 0)
        self.assertAlmostEqual(dg.js(p, q), dg.js(q, p))
        self.assertEqual(dg.kl(p, p), 0)

    def test_threat_classes(self):
        game = Game()
        for move in [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9), (0, 4)]:  # black three on row 7, black to move
            game.play(*move)
        classes = dg.threat_classes(game, game.legal_moves())
        self.assertIn((7, 10), classes['four'])
        self.assertIn((7, 10), classes['open_four'])
        self.assertFalse(classes['five'])
        self.assertTrue(classes['four'] <= classes['window3'])

    def test_d5_visits_only(self):
        rows = [{'mask': False, 'moves': [[7, 7]], 'kind': 'visits', 'target': [[1, 1, 0.5], [2, 2, 0.5]]},
                {'mask': False, 'moves': [], 'kind': 'proof_one_hot', 'target': [[1, 1, 1.0]]},
                {'mask': True, 'moves': [], 'kind': 'visits', 'target': [[1, 1, 1.0]]}]
        out = dg.d5(rows)
        self.assertEqual((out['white']['visits_rows'], out['white']['children'], out['white']['top_share']), (1, 2, 0.5))
        self.assertAlmostEqual(out['white']['support'], 2.0, places=4)
        self.assertEqual((out['black']['visits_rows'], out['black']['proof_one_hot_rows']), (0, 1))

    def test_d2_strata(self):
        a, b = (7, 8), (8, 8)
        rows = [{'mask': False, 'split': 'val', 'kind': 'visits', 'moves': [[7, 7]], 'target': [[7, 8, 0.75], [8, 8, 0.25]]},
                {'mask': False, 'split': 'train', 'kind': 'visits', 'moves': [[7, 7]], 'target': [[7, 8, 1.0]]}]
        parent = lambda game: {a: 0.4, b: 0.6}  # noqa: E731
        candidate = lambda game: {a: 0.9, b: 0.1}  # noqa: E731
        out = dg.d2(rows, parent, candidate)
        cell = out['visits/white']
        self.assertEqual(cell['n'], 1)  # train rows are not validation
        self.assertAlmostEqual(cell['parent']['ce'], -(0.75 * math.log(0.4) + 0.25 * math.log(0.6)), places=4)
        self.assertEqual((cell['parent']['top1'], cell['candidate']['top1']), (0, 1))  # the mode is (7, 8)
        self.assertLess(cell['delta_ce'], 0)


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class EndToEndTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_run_and_refusals(self):
        from analysis import tb_v3_net
        from scripts.h7c_labels import value_row_dict, write_atomic
        from tests.test_h7c_arena import play_args, write_candidate
        from tests.test_h7c_labels import build, game
        from tests.test_tb_v3_net import write_models

        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parent_policy, parent_value = write_models(tmp / 'p', seed=2)
            policy, value = write_candidate(tmp / 'c', parent_policy, parent_value)
            out = ar.play(play_args(tmp, policy, value, parent_policy, parent_value, arm='p_only', max_plies=7),
                          log=quiet)
            out['games'][0]['result'] = 'loss'  # a cut game stands in for a loss: D3 replays its moves
            p_only = tmp / 'p_only.json'
            p_only.write_text(json.dumps(out), encoding='utf-8')
            parent = tb_v3_net.model_identity(parent_policy, parent_value)
            rows, policy_rows, _ = build([game(i, models=parent, model_hash=tb_v3_net.model_hash(parent))
                                          for i in range(40)])
            write_atomic({tmp / 'value_rows.jsonl.gz': [value_row_dict(r) for r in rows],
                          tmp / 'policy_rows.jsonl.gz': policy_rows})
            sha = {name: ar._file_sha(tmp / name) for name in ('value_rows.jsonl.gz', 'policy_rows.jsonl.gz')}
            verdict, evidence = tmp / 'verdict.json', tmp / 'evidence.json'
            verdict.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1,
                                           'candidate': tb_v3_net.model_identity(policy, value), 'parent': parent,
                                           'arm_sha256': {'p_only': ar._file_sha(p_only)}}))
            evidence.write_text(json.dumps({'files_sha256': sha, 'arena': {'arm_sha256': {'p_only': ar._file_sha(p_only)}}}))
            args = argparse.Namespace(cycle=1, verdict=verdict, evidence=evidence, p_only=p_only,
                                      candidate_policy=policy, candidate_value=value, parent_policy=parent_policy,
                                      parent_value=parent_value, policy_rows=tmp / 'policy_rows.jsonl.gz',
                                      value_rows=tmp / 'value_rows.jsonl.gz', parts=None, device='cpu', workers=1,
                                      output=tmp / 'diagnosis.json', allow_dirty=True)
            result = dg.run(args, log=quiet)
            parts = result['parts']
            tree_moves = sum(m['route'] == 'tree' for g in out['games'] for m in g['tested_moves'])
            self.assertEqual(parts['d1']['positions'], tree_moves)
            self.assertEqual(parts['d3']['losses'], 1)
            self.assertEqual(parts['d3']['status']['NOT_REPRODUCED'], 0)  # the record replays move for move
            self.assertEqual(sum(parts['d3']['status'].values()), 1)
            exact = parts['d4']['exact_leaf']
            self.assertTrue(0 < exact['exact_leaf_rows_share'] < 1)  # the five-in-one positions are settled exactly
            self.assertTrue(all(k.split('/')[1] in ('black', 'white') for k in parts['d4']['val_strata']))
            self.assertEqual(set(parts['d4']['probes']), {'parent', 'candidate'})
            self.assertEqual(parts['d4']['probes']['parent']['device'], 'cpu')
            self.assertIn('black', parts['d5'])
            self.assertEqual(result['inputs']['p_only_signature'], out['signature'])
            # fail closed on any input the records do not name
            other = tmp / 'other.json'
            other.write_text(json.dumps({**out, 'games': out['games'][:1]}))
            for change in ({'p_only': other}, {'candidate_value': parent_value},
                           {'policy_rows': tmp / 'value_rows.jsonl.gz'}, {'cycle': 2}):
                with self.assertRaises(ValueError, msg=str(change)):
                    dg.check_inputs(argparse.Namespace(**{**vars(args), **change}))


if __name__ == '__main__':
    unittest.main()
