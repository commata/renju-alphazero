"""scripts/summarize_h7_cpu_cert.py: CPU backend vs CUDA backend of TB-V2-Value + fast4, paired."""
import unittest

from analysis.tb_v2_runtime import ENGINE_HASH, runtime_hash
from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE
from scripts import summarize_h7_cpu_cert as cert
from tests.test_h7_e2 import run as e2_run


def run(device, seed, loss_every=0, pairs=50):
    games = {}
    for p in range(pairs):
        for colour in ('black', 'white'):
            lost = loss_every and device == 'cpu' and p % loss_every == 0 and colour == 'black'
            games[(p, colour)] = ('loss' if lost else 'win', ['SAFE'] * 3, 1.0)
    r = e2_run(cert.ARM, seed, games, table=True)
    r.update({'arm': cert.ARM, 'v8_config': {**TB_V2_VALUE, **ENGINE_OPTIONS}, 'fast': {'requested': 'fast4'},
              'value_device': device, 'pairs': pairs,
              'runtime': {'engine_hash': ENGINE_HASH, 'runtime_hash': runtime_hash(device)}})
    return r


class CertTest(unittest.TestCase):
    def go(self, **kw):
        return cert.evaluate([run('cpu', s, **kw) for s in cert.SEEDS], [run('cuda', s) for s in cert.SEEDS])

    def test_verdicts(self):
        out = self.go()
        self.assertEqual((out['verdict'], out['strength']['diff']), ('PASS', 0.0))
        self.assertEqual(out['report']['identical_games'], 200)
        self.assertEqual(self.go(loss_every=4)['verdict'], 'FAIL')  # -0.125 per opening on average

    def test_refusals(self):
        good = [run('cpu', s) for s in cert.SEEDS], [run('cuda', s) for s in cert.SEEDS]
        with self.assertRaises(ValueError):  # the CPU runs say cuda
            cert.evaluate(good[1], good[1])
        with self.assertRaises(ValueError):  # one seed missing
            cert.evaluate(good[0][:1], good[1][:1])
        bad = run('cpu', 8431)
        bad['fast'] = {'requested': 'fast3'}
        with self.assertRaises(ValueError):
            cert.evaluate([bad, good[0][1]], good[1])
        bad = run('cpu', 8431, pairs=25)
        with self.assertRaises(ValueError):
            cert.evaluate([bad, good[0][1]], good[1])


if __name__ == '__main__':
    unittest.main()
