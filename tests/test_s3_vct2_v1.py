"""S3-VCT2-v1 is frozen (docs/mcts-v8-teacher.md §12.25) and reachable from the web UI."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.mcts_v8 import V8_DEFAULTS
from analysis.s3_vct2_v1 import S3_VCT2_V1, check_checkpoint, make_agent
from renju import WHITE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs' / 'mcts-v8-results'


def uniform_policy(game):
    legal = game.legal_moves()
    return {m: 1.0 / len(legal) for m in legal}


class FrozenConfigTest(unittest.TestCase):
    def test_config_is_the_one_the_s3_runs_used(self):
        for name in ('s3_vct2_8411.json', 's3_vct2_8412.json'):
            run = json.loads((RESULTS / name).read_text(encoding='utf-8'))
            self.assertEqual(S3_VCT2_V1, run['v8_config'], name)

    def test_every_v8_option_is_pinned(self):
        self.assertEqual(set(S3_VCT2_V1), set(V8_DEFAULTS))

    def test_other_checkpoints_are_refused(self):
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp, 'best.pt')
            ckpt.write_bytes(b'not the H3 weights')
            ckpt.with_suffix('.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                check_checkpoint(ckpt)

    def test_agent(self):
        agent = make_agent(uniform_policy, seed=3)
        self.assertEqual(agent.name, 'S3-VCT2-v1')
        for key, value in S3_VCT2_V1.items():
            self.assertEqual(getattr(agent, key), value, key)


class WebPlayS3Test(unittest.TestCase):
    def setUp(self):
        import scripts.run_web_play as web
        self.web = web
        self.saved = (web._S3_FACTORY, dict(web._S3_INFO), dict(web.VERSION_LABELS))
        web.register_s3(lambda seed: make_agent(uniform_policy, seed=seed),
                        {'policy_checkpoint': 'test', 'policy_checkpoint_sha256': None})

    def tearDown(self):
        web = self.web
        web._S3_FACTORY, web._S3_INFO = self.saved[0], self.saved[1]
        web.VERSION_LABELS.clear()
        web.VERSION_LABELS.update(self.saved[2])

    def test_game_undo_and_unfinished_log(self):
        with TemporaryDirectory() as tmp:
            session = self.web.PlaySession(log_root=Path(tmp))
            state = session.reset(agent_key='s3', human_color=WHITE, seed=5)
            self.assertEqual(state['move_count'], 1)
            state = session.play_human(7, 8)
            self.assertEqual(state['move_count'], 3)
            ai = session.move_records[-1]
            self.assertEqual(ai['actor'], 'S3-VCT2-v1')
            self.assertIn('v8_vct2_checked', ai['diagnostics'])
            self.assertEqual(len(ai['board_hash']), 16)

            state = session.undo()  # back to the human's turn: the AI reply and the human move go
            self.assertEqual(state['move_count'], 1)
            self.assertEqual(state['undo_count'], 1)
            self.assertIsNone(state['last_ai_move'])
            with self.assertRaises(Exception):
                session.undo()  # no human move left

            session.play_human(7, 8)
            session.reset(agent_key='v6', human_color=WHITE, seed=5)  # leaving an S3 game keeps its log
            logs = list(Path(tmp).glob('*_s3_*/game.json'))
            self.assertEqual(len(logs), 1)
            payload = json.loads(logs[0].read_text(encoding='utf-8'))
            self.assertEqual(payload['result'], 'UNFINISHED')
            self.assertEqual(payload['undo_count'], 1)
            self.assertEqual(payload['agent_info']['config'], S3_VCT2_V1)
            self.assertEqual(payload['agent_info']['engine'], 'S3-VCT2-v1')
            header = (logs[0].parent / 'moves.csv').read_text(encoding='utf-8-sig').splitlines()[0]
            self.assertIn('vct2_checked', header)

    def test_veto_position_is_saved_at_once(self):
        with TemporaryDirectory() as tmp:
            session = self.web.PlaySession(log_root=Path(tmp))
            session.reset(agent_key='s3', human_color=WHITE, seed=5)
            diag = {'v8_v7_move': [6, 6], 'v8_vct2_checked': [[[6, 6], 'UNSAFE'], [[8, 8], 'SAFE']],
                    'v8_vct2_switched': True}
            session._save_veto_position_locked((8, 8), 1.5, diag)
            files = list(Path(tmp, 'veto_positions').glob('*.json'))
            self.assertEqual(len(files), 1)
            saved = json.loads(files[0].read_text(encoding='utf-8'))
            self.assertEqual(saved['history'], [[7, 7]])
            self.assertEqual(saved['played'], [8, 8])
            self.assertTrue(saved['switched'])
            self.assertEqual(session.state()['veto_positions'], [str(files[0])])


if __name__ == '__main__':
    unittest.main()
