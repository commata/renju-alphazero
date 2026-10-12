"""TB-Engine-v1 is frozen (docs/mcts-v8-teacher.md §12.36)."""
import json
from pathlib import Path
import unittest

from analysis.e3_arms import E3_A
from analysis.mcts_v8 import V8_DEFAULTS
from analysis.tb_engine_v1 import FREEZE_EVIDENCE, NAME, TB_ENGINE_V1, config_hash, make_agent

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs' / 'mcts-v8-results'


def uniform_policy(game):
    legal = game.legal_moves()
    return {m: 1.0 / len(legal) for m in legal}


class FrozenTest(unittest.TestCase):
    def test_config_is_the_one_e3_and_the_holdout_ran(self):
        for name in ('e3/puct_policy_vct2_50k_8415.json', 'e3/puct_policy_vct2_50k_8416.json',
                     'e1h/puct_policy_vct2_50k_8417.json', 'e1h/puct_policy_vct2_50k_8418.json'):
            run = json.loads((RESULTS / name).read_text(encoding='utf-8'))
            self.assertEqual(TB_ENGINE_V1, run['v8_config'], name)
        self.assertEqual(TB_ENGINE_V1, E3_A)

    def test_every_v8_option_is_pinned(self):
        self.assertEqual(set(TB_ENGINE_V1), set(V8_DEFAULTS))

    def test_freeze_evidence(self):
        self.assertEqual(json.loads((ROOT / FREEZE_EVIDENCE['e1_holdout']).read_text(encoding='utf-8'))['verdict'],
                         'PASS')
        self.assertEqual(json.loads((ROOT / FREEZE_EVIDENCE['e3']).read_text(encoding='utf-8'))['reading'],
                         'MECHANISM_ESTABLISHED_BUT_MATCH_UNPROVEN')

    def test_agent_and_hash(self):
        agent = make_agent(uniform_policy, seed=3)
        self.assertEqual(agent.name, NAME)
        for key, value in TB_ENGINE_V1.items():
            self.assertEqual(getattr(agent, key), value, key)
        self.assertEqual(len(config_hash()), 64)
        self.assertEqual(config_hash(), config_hash())


if __name__ == '__main__':
    unittest.main()


class WebPlayTB1Test(unittest.TestCase):
    def setUp(self):
        import scripts.run_web_play as web
        self.web = web
        self.saved = (web._TB1_FACTORY, dict(web._TB1_INFO), dict(web.VERSION_LABELS))
        web.register_tb1(lambda seed: make_agent(uniform_policy, seed=seed),
                         {'policy_checkpoint': 'test', 'policy_checkpoint_sha256': None})

    def tearDown(self):
        web = self.web
        web._TB1_FACTORY, web._TB1_INFO = self.saved[0], self.saved[1]
        web.VERSION_LABELS.clear()
        web.VERSION_LABELS.update(self.saved[2])

    def test_game_log_records_the_frozen_engine(self):
        from tempfile import TemporaryDirectory
        from renju import WHITE

        with TemporaryDirectory() as tmp:
            session = self.web.PlaySession(log_root=Path(tmp))
            session.reset(agent_key='tb1', human_color=WHITE, seed=5)
            session.play_human(7, 8)
            self.assertEqual(session.move_records[-1]['actor'], NAME)
            self.assertIn('v8_vct2_checked', session.move_records[-1]['diagnostics'])
            session.reset(agent_key='v6', human_color=WHITE, seed=5)  # leaving an unfinished tb1 game keeps its log
            logs = list(Path(tmp).glob('*_tb1_*/game.json'))
            self.assertEqual(len(logs), 1)
            info = json.loads(logs[0].read_text(encoding='utf-8'))['agent_info']
            self.assertEqual((info['engine'], info['engine_config_hash']), (NAME, config_hash()))
            self.assertEqual(info['config']['vct2_node_budget'], 50_000)
