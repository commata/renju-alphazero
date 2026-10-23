"""scripts/summarize_tb_v3_cpu_cert.py: TB-V3 Resolve / fast5 with its value on CPU vs on CUDA, paired."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from analysis import tb_v3, tb_v3_fast5
from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE
from scripts import summarize_tb_v3_cpu_cert as cert
from tests.test_h7_e2 import run as e2_run


def run(device, seed, loss_every=0, pairs=50):
    games = {}
    for p in range(pairs):
        for colour in ('black', 'white'):
            lost = loss_every and device == 'cpu' and p % loss_every == 0 and colour == 'black'
            games[(p, colour)] = ('loss' if lost else 'win', ['SAFE'] * 3, 1.0)
    r = e2_run(cert.ARM, seed, games, table=True)
    for g in r['games']:
        g['runtime'] = 'fast5'
    r.update({'arm': cert.ARM, 'v8_config': {**TB_V2_VALUE, **ENGINE_OPTIONS, 'tb_v3': True},
              'fast': {'requested': 'fast4'}, 'runtime_mode': 'fast5', 'value_device': device, 'pairs': pairs,
              'canonical_factory': True, 'provenance': {'git_dirty': False}, 'openings': [{'pair': 0}],
              'runtime': {'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': tb_v3_fast5.runtime_hash(device)}})
    return r


class CertTest(unittest.TestCase):
    def go(self, **kw):
        return cert.evaluate([run('cpu', s, **kw) for s in cert.SEEDS], [run('cuda', s) for s in cert.SEEDS])

    def test_verdicts(self):
        out = self.go()
        self.assertEqual((out['verdict'], out['strength']['diff']), ('PASS', 0.0))
        self.assertEqual(self.go(loss_every=4)['verdict'], 'FAIL')  # -0.125 per opening on average

    def test_refusals(self):
        cuda = [run('cuda', s) for s in cert.SEEDS]
        with self.assertRaises(ValueError):  # the CPU runs say cuda
            cert.evaluate(cuda, cuda)
        for change in ({'runtime_mode': 'fast4'}, {'pairs': 25}, {'provenance': {'git_dirty': True}},
                       {'runtime': {'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': 'f' * 64}},
                       {'seed': 8431}, {'canonical_factory': None}):
            bad = run('cpu', 8451)
            bad.update(change)
            with self.assertRaises(ValueError, msg=str(change)):
                cert.evaluate([bad, run('cpu', 8452)], cuda)
        bad = run('cpu', 8451)
        bad['games'][3]['runtime'] = 'fast4'  # one game resumed from another runtime
        with self.assertRaises(ValueError):
            cert.evaluate([bad, run('cpu', 8452)], cuda)

    def test_command_line(self):
        """As run on the desktop: ``python scripts/summarize_tb_v3_cpu_cert.py`` (only ``src`` on PYTHONPATH)."""
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            args = []
            for device in ('cpu', 'cuda'):
                args.append(f'--{device}')
                for seed in cert.SEEDS:
                    path = Path(tmp) / f'{device}_{seed}.json'
                    path.write_text(json.dumps(run(device, seed)), encoding='utf-8')
                    args.append(str(path))
            out = Path(tmp) / 'cert.json'
            env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
            done = subprocess.run([sys.executable, str(root / 'scripts/summarize_tb_v3_cpu_cert.py'), *args,
                                   '--output', str(out)], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            record = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual((record['format'], record['verdict']), (cert.FORMAT, 'PASS'))
            self.assertEqual(len(record['runs']), 4)
            self.assertTrue(all(len(v) == 64 for v in record['runs'].values()))


if __name__ == '__main__':
    unittest.main()
