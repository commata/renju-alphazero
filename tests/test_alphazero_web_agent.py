"""AlphaZero opponent for web play: root diagnostics and saved logs (torch-free)."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agents.alphazero_agent import AlphaZeroAgent
from renju import BLACK, WHITE, Game
from search.alphazero import SearchConfig
from search.evaluator import UniformEvaluator
import scripts.run_web_play as web

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


def make_agent(rules=True, simulations=16):
    config = SearchConfig(num_simulations=simulations, temperature_moves=0,
                          noise_enabled=False, tactical_rules=rules)
    return AlphaZeroAgent(UniformEvaluator(), config, name='AlphaZero-test',
                          info={'generation': 0})


class AlphaZeroAgentTest(unittest.TestCase):
    def test_rejects_noise_or_temperature(self):
        with self.assertRaises(ValueError):
            AlphaZeroAgent(UniformEvaluator(), SearchConfig(num_simulations=4))

    def test_root_diagnostics(self):
        game = Game()
        game.play(7, 7)
        agent = make_agent()
        move = agent.select_move(game)
        diag = agent.diagnostics
        self.assertIn(move, game.legal_moves())
        self.assertEqual(diag['chosen'], list(move))
        self.assertEqual(sum(diag['root_visits'].values()), 16)
        self.assertEqual(diag['root_value'], 0.0)
        self.assertLessEqual(len(diag['top_visits']), agent.top_k)
        self.assertEqual(diag['top_visits'][0]['move'], list(move))
        self.assertEqual(diag['tactical_allowed'], len(game.legal_moves()))
        self.assertIsNone(diag['tactical_proven'])
        json.dumps(diag)

    def test_tactical_rules_record_forced_block(self):
        game = Game()
        for move in [(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0), (7, 4)]:
            game.play(*move)  # black four 7,4..7,7 with (7,3) and (7,8) open
        agent = make_agent()
        agent.select_move(game)
        # Two winning points: the rules prove the root lost and allow every move.
        self.assertEqual(agent.diagnostics['tactical_proven'], -1.0)

    def test_single_legal_fast_path_has_no_tree(self):
        agent = make_agent()
        move = agent.select_move(Game())
        self.assertEqual(move, (7, 7))
        self.assertTrue(agent.diagnostics['fast_path'])
        self.assertNotIn('root_visits', agent.diagnostics)


@unittest.skipIf(torch is None, 'requires torch')
class AlphaZeroCheckpointTest(unittest.TestCase):
    def test_from_training_checkpoint_uses_its_evaluation_search(self):
        from training.config import load_config
        from training.training_checkpoint import build_checkpoint, save_atomic
        from training.training_state import init_training_state

        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['evaluation']['tactical_rules'] = True
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint_gen000.pt'
            save_atomic(path, build_checkpoint(init_training_state(config)))
            agent = AlphaZeroAgent.from_checkpoint(path)
            self.assertEqual(agent.config.num_simulations, config['evaluation']['puct_simulations'])
            self.assertTrue(agent.config.tactical_rules)
            self.assertEqual(agent.name, 'AlphaZero-gen0')
            self.assertEqual(len(agent.info['checkpoint_sha256']), 64)
            off = AlphaZeroAgent.from_checkpoint(path, simulations=5, tactical_rules='off')
            self.assertFalse(off.config.tactical_rules)
            game = Game()
            game.play(7, 7)
            move = off.select_move(game)
            self.assertIn(move, game.legal_moves())
            self.assertEqual(sum(off.diagnostics['root_visits'].values()), 5)


class WebPlayAlphaZeroTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(web.VERSION_LABELS)
        web.register_alphazero(lambda: make_agent(simulations=8), 'AlphaZero test')

    def tearDown(self):
        web.VERSION_LABELS.clear()
        web.VERSION_LABELS.update(self.saved)
        web._ALPHAZERO_FACTORY = None

    def test_session_logs_root_record(self):
        with TemporaryDirectory() as tmp:
            session = web.PlaySession(log_root=Path(tmp))
            state = session.reset(agent_key='az', human_color=BLACK, seed=1)
            self.assertIn('az', state['versions'])
            self.assertEqual(state['move_count'], 2)  # center + AI reply
            record = session.move_records[-1]
            self.assertEqual(record['actor'], 'AlphaZero-test')
            self.assertEqual(sum(record['diagnostics']['root_visits'].values()), 8)
            # Finish quickly: mark the game done and save.
            session.game.done = True
            session.game.winner = WHITE
            log_dir = session._save_completed_game_locked()
            payload = json.loads((log_dir / 'game.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['agent_info'], {'generation': 0})
            self.assertIn('top_visits', payload['moves'][-1]['diagnostics'])
            with open(log_dir / 'moves.csv', encoding='utf-8-sig') as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn('az_root_value', rows[0])
            self.assertEqual(rows[-1]['az_root_value'], '0.0')
            self.assertEqual(json.loads(rows[-1]['az_top_visits'])[0][:2],
                             [record['row'], record['col']])

    def test_classic_versions_keep_csv_columns(self):
        with TemporaryDirectory() as tmp:
            session = web.PlaySession(log_root=Path(tmp))
            session.game.done = True
            session.game.winner = BLACK
            log_dir = session._save_completed_game_locked()
            with open(log_dir / 'moves.csv', encoding='utf-8-sig') as handle:
                header = next(csv.reader(handle))
            self.assertNotIn('az_root_value', header)
            self.assertNotIn('agent_info',
                             json.loads((log_dir / 'game.json').read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
