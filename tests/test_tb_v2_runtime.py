"""TB-V2-Value + fast4 runtime profile is frozen (docs/h6-value.md, P3)."""
import hashlib
import json
from pathlib import Path
import unittest

from analysis import tb_v2_runtime as rt
from rt30 import fast4_forbidden
from tests.test_tb_v2_value import Leaf
from tests.tb_engine_v1_regression import hashed_policy

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenTest(unittest.TestCase):
    def test_identities(self):
        from analysis.tb_v2_value import VALUE_CHECKPOINT_SHA256, config_hash
        self.assertEqual(rt.ENGINE_HASH, config_hash())
        for name, digest in rt.NATIVE_SOURCES.items():
            self.assertEqual(sha(ROOT / 'src' / 'rt30' / 'native' / name), digest, name)
        for name, digest in rt.BINDING_SOURCES.items():
            self.assertEqual(sha(ROOT / name), digest, name)
        self.assertNotEqual(rt.runtime_hash('cuda'), rt.runtime_hash('cpu'))
        self.assertEqual(rt.runtime_hash('cuda:0'), rt.runtime_hash('cuda'))
        self.assertEqual(rt.CERTIFICATION['cpu'], None)  # until the CPU-backend certification passes
        self.assertEqual(len(VALUE_CHECKPOINT_SHA256), 64)

    def test_evidence(self):
        p2 = json.loads((ROOT / rt.EVIDENCE['desktop_p2_cpu']).read_text(encoding='utf-8'))
        self.assertEqual(p2['verdict'], 'PASS')
        self.assertEqual(json.loads((ROOT / rt.EVIDENCE['function_check']).read_text(encoding='utf-8'))['summary']['verdict'],
                         'PASS')
        for key in ('ser5_r1', 'ser5_r2'):
            s = json.loads((ROOT / rt.EVIDENCE[key]).read_text(encoding='utf-8'))
            self.assertEqual(s['engine'], rt.ENGINE_HASH)
            self.assertEqual(s['summary']['all']['over_deadline'], 0)
            self.assertLessEqual(s['summary']['all']['max'], 25.0)


@unittest.skipUnless(fast4_forbidden.available(), 'python -m rt30.native.build first')
class BindingTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        install('off')

    def test_agent_only_on_fast4(self):
        from hybrid.h6_fast import install
        from analysis.tb_v2_value import VALUE_CHECKPOINT_SHA256

        leaf = Leaf(VALUE_CHECKPOINT_SHA256)
        for mode in ('off', 'fast3'):
            install(mode)
            with self.assertRaises(RuntimeError):
                rt.make_agent(hashed_policy, leaf)
        rt.install()
        agent = rt.make_agent(hashed_policy, leaf, seed=3)
        self.assertEqual((agent.name, agent.root_vcf_table), (rt.NAME, True))
        p = rt.provenance('cpu')
        self.assertEqual((p['engine_hash'], p['runtime_hash'], p['certified']),
                         (rt.ENGINE_HASH, rt.runtime_hash('cpu'), None))
        self.assertTrue(p['native_binaries'])

    def test_a_rebound_name_is_caught(self):
        from search import mcts

        rt.install()
        saved = mcts.forbidden_reason
        mcts.forbidden_reason = fast4_forbidden.FROZEN  # partly installed
        try:
            with self.assertRaises(RuntimeError):
                rt.check_bound()
        finally:
            mcts.forbidden_reason = saved


if __name__ == '__main__':
    unittest.main()
