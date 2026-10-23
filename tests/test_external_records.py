"""External-anchor records: the TB1 / TB-V3 bootstrap and the WSL2 gate manifest."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import unittest.mock

from scripts import external_pair_bootstrap as eb
from scripts import external_wsl_gate as wg

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTest(unittest.TestCase):
    def test_committed_record_reproduces(self):
        record = json.loads((ROOT / 'docs/external/tb1_tbv3_bootstrap.json').read_text(encoding='utf-8'))
        a, _ = eb.load_run('docs/external/x1_calibration.json:confirmation')
        b, _ = eb.load_run('docs/external/x2_c0_prerl.json')
        self.assertEqual({str(k): v for k, v in a.items()}, record['a']['pair_scores'])
        self.assertEqual((sum(a.values()) / 25, sum(b.values()) / 25), (0.26, 0.46))
        again = eb.bootstrap_difference(a, b, record['samples'], record['rng_seed'])
        self.assertEqual((again['difference'], again['ci95']), (record['difference'], record['ci95']))
        self.assertNotIn('p_value', record)

    def test_x1_recovery_guards(self):
        summary = json.loads((ROOT / 'docs/external/x1_calibration.json').read_text(encoding='utf-8'))
        summary = summary['confirmation']['summary']
        with self.assertRaises(ValueError):
            eb.pair_scores_from_x1({**summary, 'loss_candidates': summary['loss_candidates'][:5]})
        with self.assertRaises(ValueError):
            eb.pair_scores_from_x1({**summary, 'draw': 1})


class WslGateTest(unittest.TestCase):
    def test_pass_and_fail(self):
        want = wg.expected()
        with TemporaryDirectory() as tmp:
            engine = Path(tmp)
            with unittest.mock.patch.object(wg, '_sha', lambda p: (
                    want['executable_sha256'] if p.name == wg.EXECUTABLE else want['extra_files'].get(p.name))):
                crosscheck = {'seed': 1, 'radius': 1, 'summary': want['crosscheck']['summary'],
                              'engine': {'executable_sha256': want['executable_sha256'], 'about': 'Rapfi'},
                              'git_commit': 'abc', 'git_dirty': False}
                repro = {'seed': 8703, 'pairs': 2, 'run_signature': want['repro']['run_signature'],
                         'git_commit': 'abc', 'git_dirty': False,
                         'summary': {'fingerprint': want['repro']['fingerprint'], 'score': 0.75},
                         'engine': {'about': 'Rapfi'}}
                self.assertEqual(wg.evaluate(engine, crosscheck, repro)['result'], 'PASS')
                bad = {**repro, 'summary': {'fingerprint': 'f' * 64}}
                self.assertEqual(wg.evaluate(engine, crosscheck, bad)['result'], 'FAIL')
                bad = {**crosscheck, 'summary': {**crosscheck['summary'], 'mismatch_positions': 1}}
                self.assertEqual(wg.evaluate(engine, bad, repro)['result'], 'FAIL')
                for dirty in (True, None):  # a dirty tree, or git unavailable: never a clean record
                    out = wg.evaluate(engine, {**crosscheck, 'git_dirty': dirty}, repro)
                    self.assertEqual((out['result'], out['checks']['clean_tree']), ('FAIL', False))
                    self.assertEqual(wg.evaluate(engine, crosscheck, {**repro, 'git_dirty': dirty})['result'], 'FAIL')
            self.assertEqual(wg.evaluate(engine, crosscheck, repro)['checks']['executable'], False)  # no files


if __name__ == '__main__':
    unittest.main()
