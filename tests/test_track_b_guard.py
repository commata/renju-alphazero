"""analysis.track_b_guard and scripts/check_track_b_frozen.py: the checks the frozen canonical leaves to callers."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis import track_b_guard as guard
from analysis.tb_v2_value import VALUE_CHECKPOINT_SHA256


class Leaf:
    def __init__(self, sha):
        if sha is not None:
            self.sha256 = sha


class GuardTest(unittest.TestCase):
    def test_leaf_fails_closed(self):
        self.assertEqual(guard.check_leaf(Leaf(VALUE_CHECKPOINT_SHA256)), VALUE_CHECKPOINT_SHA256)
        for bad in (None, 'f' * 64):  # no sha (the frozen make_agent lets this through) / another checkpoint
            with self.assertRaises(ValueError):
                guard.check_leaf(Leaf(bad))

    def test_policy_checkpoint(self):
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'best.pt'
            ckpt.write_bytes(b'another policy')
            ckpt.with_suffix('.json').write_text(json.dumps({'policy_trained': True}))
            with self.assertRaises(ValueError):
                guard.check_policy_checkpoint(ckpt)

    def test_value_checkpoint_and_objects(self):
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / 'best.pt'
            ckpt.write_bytes(b'another value')
            (Path(tmp) / 'best.json').write_text('{}')
            with self.assertRaises(ValueError):
                guard.check_value_checkpoint(ckpt)
            with self.assertRaises(TypeError):  # stand-ins are not the real RootPolicy / ValueLeaf
                guard.check_models(object(), Leaf(VALUE_CHECKPOINT_SHA256), ckpt, ckpt)
        self.assertEqual(len(guard.VALUE_METADATA_SHA256), 64)
        e3 = json.loads(Path(guard.ROOT / 'docs/h7/h7_e3_p92.json').read_text(encoding='utf-8'))['provenance']
        self.assertEqual((e3['value_checkpoint_sha256'], e3['value_metadata_sha256']),
                         (VALUE_CHECKPOINT_SHA256, guard.VALUE_METADATA_SHA256))  # recorded with the frozen value

    def test_sources_and_code_manifest(self):
        guard.check_runtime_sources()
        self.assertEqual(guard.check_code(), [])
        names = guard.read_manifest()
        for core in ('src/analysis/mcts_v8.py', 'src/analysis/threats.py', 'src/hybrid/h6_value_leaf.py',
                     'src/hybrid/h4_policy.py', 'src/model/network.py', 'src/renju/rules.py',
                     'src/rt30/fast4_forbidden.py', 'src/analysis/tb_v2_runtime.py'):
            self.assertIn(core, names)
        self.assertNotIn('src/hybrid/h6_labels.py', names)  # tooling (seed lists), not the engine
        with TemporaryDirectory() as tmp:
            changed = Path(tmp) / 'm.sha256'
            changed.write_text('0' * 64 + '  src/analysis/mcts_v8.py\n' +
                               hashlib.sha256(Path(guard.ROOT / 'src/renju/rules.py').read_bytes()).hexdigest()
                               + '  src/renju/rules.py\n')
            self.assertEqual(guard.check_code(changed), ['src/analysis/mcts_v8.py'])

    def test_checker_without_models(self):
        from scripts import check_track_b_frozen as checker
        self.assertEqual(checker.main(['--no-models']), 0)
        self.assertEqual(sorted(guard.read_manifest()), checker.closure())  # nothing imported is unlisted


class BenchmarkTest(unittest.TestCase):
    def test_canonical_arms_use_the_factory_only_unchanged(self):
        from scripts.run_mcts_v8_benchmark import is_canonical, v8_config
        for arm in ('puct_value_vct2_50k_tt', 'puct_value_vct2_50k_tt_e4'):
            self.assertTrue(is_canonical(arm, 'fast4', v8_config(arm, {'puct_c': 1.5})))
            self.assertFalse(is_canonical(arm, 'off', v8_config(arm, {'puct_c': 1.5})))  # not the runtime profile
            self.assertFalse(is_canonical(arm, 'fast4', v8_config(arm, {'puct_c': 1.0})))
            self.assertFalse(is_canonical(arm, 'fast4', v8_config(arm, {'puct_c': 1.5, 'simulations': 10})))
        self.assertFalse(is_canonical('puct_policy_vct2_50k_tt', 'fast4', v8_config('puct_policy_vct2_50k_tt')))


if __name__ == '__main__':
    unittest.main()
