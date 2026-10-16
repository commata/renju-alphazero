"""TB-RT30-v3 is frozen (docs/web-ser5.md §9.10)."""
import hashlib
import json
from pathlib import Path
import unittest

from rt30 import tb_rt30_v1, tb_rt30_v3
from rt30.tb_rt30_v3 import FREEZE_EVIDENCE, NAME, STATUS, TB_RT30_V3, config_hash

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v3.sha256'
V1_LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v1.sha256'
CONFIG_HASH = 'f80d2348a85b34add30e30c6157e197a365517aca64ca1668279a14298c29eda'


def evidence(name):
    path = FREEZE_EVIDENCE[name]
    text = (ROOT / path).read_text(encoding='utf-8')
    return text if path.endswith('.md') else json.loads(text)


def lock_entries(path):
    return [line.split(maxsplit=1) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]


class FrozenTest(unittest.TestCase):
    def test_status_and_config_hash(self):
        self.assertEqual(STATUS, 'frozen')
        self.assertEqual(config_hash(), CONFIG_HASH)

    def test_v3_is_v1_plus_x1_x2(self):
        v1 = tb_rt30_v1.TB_RT30_V1
        self.assertEqual({k for k in TB_RT30_V3 if TB_RT30_V3[k] != v1.get(k)}, {'parallel_attack', 'exact_speedups'})

    def test_evidence_ran_this_configuration(self):
        a3 = evidence('a3')
        self.assertEqual((a3['engine'], a3['engine_config_hash'], a3['native_scan']), (NAME, CONFIG_HASH, True))
        self.assertEqual(a3['deadline'], TB_RT30_V3['deadline_seconds'])

    def test_gates_held(self):
        self.assertIn('A3-v3 gate: PASS', evidence('gate'))
        self.assertEqual(evidence('equivalence')['summary']['identical'], 8)
        for name in ('divergence_w4', 'divergence_w2', 'divergence_w1'):
            summary = evidence(name)['summary']
            self.assertEqual((summary['recovered'], summary['cases']), (34, 34), name)
            self.assertEqual(summary['seconds']['over_27.5s'], 0, name)
        control = evidence('control_w4')['summary']
        self.assertEqual((control['recovered'], control['cases']), (26, 26))
        for name in ('a1_s60_recheck', 'a1_crit_recheck'):
            self.assertEqual(evidence(name)['summary']['changed_move_proven_unsafe'], 0, name)

    def test_a3_is_not_harm(self):
        summary = evidence('a3')['summary']
        ref = summary['vs_reference']
        self.assertEqual(summary['finished'], 100)
        self.assertGreaterEqual(ref['diff'], -0.05)          # HARM: diff < -0.05 ...
        self.assertGreaterEqual(ref['diff_ci95_pairs'][1], 0.0)  # ... or the pair CI95 upper bound < 0
        self.assertEqual(summary['rt30_seconds']['over_27.5s'], 0)
        self.assertEqual(summary['attack_helper_failed'], 0)

    def test_no_deadline_move_is_tb1s_own_on_ser5(self):
        rows = [json.loads((ROOT / path).read_text(encoding='utf-8'))['rows'][0]
                for path in FREEZE_EVIDENCE['no_deadline_replay']]
        self.assertEqual({tuple(r['move']) for r in rows}, {(4, 6)})
        traces = [[(t['ply'], tuple(t['move']), t['rng_before']) for t in r['trace']] for r in rows]
        self.assertTrue(all(t == traces[0] for t in traces))

    def test_locked_files(self):
        entries = lock_entries(LOCK)
        self.assertGreaterEqual(len(entries), 16)
        for digest, relative in entries:
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)
        locked = {relative for _, relative in entries}
        self.assertTrue({relative for _, relative in lock_entries(V1_LOCK)} <= locked)  # v1's files too

    def test_native_source_is_the_one_built_on_ser5(self):
        build = evidence('native_build')
        self.assertEqual(build['python'], '3.14.4')
        source = ROOT / 'src' / 'rt30' / 'native' / 'threat_windows.c'
        self.assertEqual(build['sources']['threat_windows.c'], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_name(self):
        self.assertEqual(tb_rt30_v3.NAME, 'TB-RT30-v3')


if __name__ == '__main__':
    unittest.main()
