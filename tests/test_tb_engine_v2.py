"""TB-Engine-v2 is frozen (docs/h6-value.md, H7-E2): configuration, table implementation, evidence."""
import hashlib
import json
from pathlib import Path
import unittest

from analysis.mcts_v8 import V8_DEFAULTS
from analysis.tb_engine_v1 import TB_ENGINE_V1
from analysis.tb_engine_v1_t import TB_ENGINE_V1_T
from analysis.tb_engine_v1_t import ENGINE_OPTIONS as T_OPTIONS
from analysis.tb_engine_v2 import (ENGINE_OPTIONS, FREEZE_EVIDENCE, NAME, TB_ENGINE_V2, VCF_TABLE, VCF_TABLE_SHA256,
                                   config_hash, make_agent)
from renju import Game
from tests.tb_engine_v1_regression import FIXTURE, SEED, WEB, hashed_policy

ROOT = Path(__file__).resolve().parents[1]


class FrozenTest(unittest.TestCase):
    def test_configuration_is_the_adopted_candidate(self):
        self.assertEqual(TB_ENGINE_V2, TB_ENGINE_V1)  # written out in full; equal to v1 by construction
        self.assertEqual((TB_ENGINE_V2, ENGINE_OPTIONS), (TB_ENGINE_V1_T, T_OPTIONS))
        self.assertEqual(set(TB_ENGINE_V2), set(V8_DEFAULTS))
        self.assertEqual(ENGINE_OPTIONS, {'root_vcf_table': True})
        self.assertNotIn('root_vcf_table', V8_DEFAULTS)

    def test_table_implementation_is_locked(self):
        path = ROOT / 'src' / (VCF_TABLE.replace('.', '/') + '.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), VCF_TABLE_SHA256)

    def test_freeze_evidence(self):
        e2 = json.loads((ROOT / FREEZE_EVIDENCE['e2']).read_text(encoding='utf-8'))
        self.assertEqual((e2['seeds'], e2['verdict']['decision']), ([8421, 8422], 'ADOPT'))
        self.assertEqual(json.loads((ROOT / FREEZE_EVIDENCE['e1']).read_text(encoding='utf-8'))['summary']['verdict'],
                         'go')

    def test_agent_and_hash(self):
        agent = make_agent(hashed_policy, seed=3)
        self.assertEqual((agent.name, agent.root_vcf_table), (NAME, True))
        for key, value in TB_ENGINE_V2.items():
            self.assertEqual(getattr(agent, key), value, key)
        self.assertEqual(len(config_hash()), 64)
        from analysis.tb_engine_v1 import config_hash as v1_hash
        self.assertNotEqual(config_hash(), v1_hash())

    def test_fixture_decisions(self):
        """The fixture's root checks finish inside the frozen budget: same moves and statuses as TB-Engine-v1."""
        moves = json.loads(WEB.read_text(encoding='utf-8'))['moves']
        for want in json.loads(FIXTURE.read_text(encoding='utf-8'))[:3]:
            game = Game()
            for m in moves[:want['ply']]:
                game.play(*m)
            agent = make_agent(hashed_policy, seed=SEED)
            self.assertEqual(list(agent.select_move(game)), want['move'])
            self.assertEqual([[list(m), s] for m, s in agent.diagnostics.v8_root_checked], want['root_checked'])
            self.assertTrue(agent.diagnostics.v8_root_vcf_table)


if __name__ == '__main__':
    unittest.main()
