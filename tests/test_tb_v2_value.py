"""TB-V2-Value is frozen (docs/h6-value.md, H7-E3): configuration, table and value identity, evidence."""
import hashlib
import json
from pathlib import Path
import unittest

from analysis.mcts_v8 import V8_DEFAULTS
from analysis.tb_engine_v2 import ENGINE_OPTIONS as V2_OPTIONS
from analysis.tb_engine_v2 import TB_ENGINE_V2
from analysis.tb_engine_v2 import VCF_TABLE_SHA256 as V2_TABLE_SHA256
from analysis.tb_v1_value import VALUE_CHECKPOINT_SHA256 as V1_VALUE_SHA256
from analysis.tb_v1_value import VALUE_INFERENCE as V1_INFERENCE
from analysis.tb_v1_value import VALUE_LAMBDA as V1_LAMBDA
from analysis.tb_v2_value import (ENGINE_OPTIONS, FREEZE_EVIDENCE, NAME, TB_V2_VALUE, VALUE_CHECKPOINT_SHA256,
                                  VALUE_INFERENCE, VALUE_LAMBDA, VCF_TABLE, VCF_TABLE_SHA256, config_hash, make_agent)
from hybrid.h6_value_leaf import V3_CHECKPOINT_SHA256
from tests.tb_engine_v1_regression import hashed_policy

ROOT = Path(__file__).resolve().parents[1]
# The value route's code (tree and D4 leaf): not in config_hash, locked here (CLAUDE.md lists them as frozen).
VALUE_CODE_SHA256 = {
    'src/analysis/puct_value_v1.py': '43f569dcce8c706ddc53121d0557d889b00bdf00f8f6cf14c099ede9def0726b',
    'src/hybrid/h6_value_leaf.py': '31c3f8c19b3e06df024963e65d025cc96a2a253c1ed6d12c65d3a142f0cfcfbb',
}


class Leaf:
    def __init__(self, sha256):
        self.sha256 = sha256

    def __call__(self, game):
        return 0.0


class FrozenTest(unittest.TestCase):
    def test_configuration_is_the_adopted_candidate(self):
        from scripts.run_mcts_v8_benchmark import v8_config

        self.assertEqual({k for k in TB_V2_VALUE if TB_V2_VALUE[k] != TB_ENGINE_V2[k]}, {'tree_mode'})
        self.assertEqual(TB_V2_VALUE['tree_mode'], 'puct_value')
        self.assertEqual(set(TB_V2_VALUE), set(V8_DEFAULTS))
        self.assertEqual(ENGINE_OPTIONS, V2_OPTIONS)
        self.assertEqual(v8_config('puct_value_vct2_50k_tt', {'puct_c': 1.5}), {**TB_V2_VALUE, **ENGINE_OPTIONS})

    def test_identity(self):
        path = ROOT / 'src' / (VCF_TABLE.replace('.', '/') + '.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), VCF_TABLE_SHA256)
        self.assertEqual(VCF_TABLE_SHA256, V2_TABLE_SHA256)
        self.assertEqual(VALUE_CHECKPOINT_SHA256, V3_CHECKPOINT_SHA256)
        self.assertEqual((VALUE_CHECKPOINT_SHA256, VALUE_INFERENCE, VALUE_LAMBDA), (V1_VALUE_SHA256, V1_INFERENCE, V1_LAMBDA))
        from analysis.tb_engine_v2 import config_hash as v2_hash
        from analysis.tb_v1_value import config_hash as v1_value_hash
        self.assertEqual(len({config_hash(), v2_hash(), v1_value_hash()}), 3)
        self.assertEqual(config_hash(), config_hash())

    def test_value_code_is_locked(self):
        for name, sha in VALUE_CODE_SHA256.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha, name)

    def test_freeze_evidence(self):
        for key in ('e2', 'e3'):
            self.assertEqual(json.loads((ROOT / FREEZE_EVIDENCE[key]).read_text(encoding='utf-8'))['verdict']['decision'],
                             'ADOPT', key)

    def test_agent(self):
        agent = make_agent(hashed_policy, Leaf(VALUE_CHECKPOINT_SHA256), seed=3)
        self.assertEqual((agent.name, agent.root_vcf_table), (NAME, True))
        for key, value in TB_V2_VALUE.items():
            self.assertEqual(getattr(agent, key), value, key)
        with self.assertRaises(ValueError):
            make_agent(hashed_policy, Leaf('0' * 64))


if __name__ == '__main__':
    unittest.main()
