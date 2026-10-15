"""TB-RT30-v1 is frozen (docs/web-ser5.md §9)."""
import hashlib
import json
from pathlib import Path
import unittest

from analysis.tb_engine_v1 import TB_ENGINE_V1
from rt30 import tb_rt30_v1
from rt30.engine import RT30_TIMING
from rt30.tb_rt30_v1 import FREEZE_EVIDENCE, NAME, TB_RT30_V1, config_hash, make_agent

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v1.sha256'
CONFIG_HASH = '4bea6459d55eef979894de924f79fd08c6a4bdbb317dbcd2f84cc9aac7a266cc'


def evidence(name):
    return json.loads((ROOT / FREEZE_EVIDENCE[name]).read_text(encoding='utf-8'))


class FrozenTest(unittest.TestCase):
    def test_base_is_tb_engine_v1(self):
        self.assertEqual(TB_RT30_V1['base_config'], TB_ENGINE_V1)

    def test_config_hash_is_pinned(self):
        self.assertEqual(config_hash(), CONFIG_HASH)

    def test_real_time_layer_is_the_one_the_a1_runs_used(self):
        for name in ('cap8_s60', 'cap8_critical'):
            run = evidence(name)
            self.assertEqual(run['engine'], 'TB-Engine-v1', name)  # real H3 policy
            self.assertEqual(run['timing'], TB_RT30_V1['timing'], name)
            self.assertEqual(run['deadline'], TB_RT30_V1['deadline_seconds'], name)
            self.assertIs(run['root_rescue'], TB_RT30_V1['root_rescue'], name)
        self.assertEqual(RT30_TIMING, TB_RT30_V1['timing'])  # the code default is the adopted one

    def test_a1_gates_held(self):
        for name in ('cap8_s60', 'cap8_critical'):
            summary = evidence(name)['summary']
            self.assertEqual(summary['rt']['seconds']['over_27.5s'], 0, name)
            self.assertEqual(summary['vs_baseline']['played_baseline_unsafe'], [], name)
        for name in ('cap8_s60_recheck', 'cap8_critical_recheck'):
            self.assertTrue(evidence(name)['summary']['gate_pass'], name)
        self.assertEqual(evidence('equivalence')['summary']['identical'], 8)

    def test_locked_files(self):
        entries = [line.split(maxsplit=1) for line in LOCK.read_text(encoding='utf-8').splitlines()
                   if line.strip() and not line.startswith('#')]
        self.assertGreaterEqual(len(entries), 8)
        for digest, relative in entries:
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_agent(self):
        agent = make_agent(None, seed=3, uniform_prior=True)
        self.assertEqual(agent.name, f'{NAME} (uniform prior)')
        self.assertEqual(agent.timing, TB_RT30_V1['timing'])
        self.assertEqual((agent.attack_order, agent.attack_scheduler, agent.root_rescue),
                         (TB_RT30_V1['attack_order'], TB_RT30_V1['attack_scheduler'], TB_RT30_V1['root_rescue']))
        for key, value in TB_RT30_V1['base_config'].items():
            if key != 'puct_prior':
                self.assertEqual(getattr(agent, key), value, key)
        self.assertEqual(agent.puct_prior, 'uniform')
        with self.assertRaises(ValueError):
            make_agent(None)  # the real engine needs the H3 policy

    def test_exact_speedups_are_installed(self):
        from rt30 import fast

        make_agent(None, uniform_prior=True)
        try:
            self.assertEqual(fast.installed(), {'windows': True, 'extensions': True})
        finally:
            fast.uninstall_all()

    def test_supervisor_factory(self):
        from rt30.supervisor import make_tb_rt30_v1_agent
        from rt30 import fast

        try:
            agent = make_tb_rt30_v1_agent(seed=1, uniform_prior=True)
            self.assertTrue(agent.supports_deadline)
            self.assertEqual(agent.timing, TB_RT30_V1['timing'])
        finally:
            fast.uninstall_all()
        with self.assertRaises(ValueError):
            make_tb_rt30_v1_agent()


if __name__ == '__main__':
    unittest.main()
