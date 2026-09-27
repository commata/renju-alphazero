"""Stage 7-B: rule-level PUCT v2 teacher, arm configs, weights export and init."""
import json
import tempfile
import unittest
from pathlib import Path

from renju import BLACK, EMPTY, WHITE, Game
from search.alphazero import SearchConfig
from search.tactics import tactical_filter, winning_points
from training.config import (config_differences, critical_config, critical_config_hash,
                             load_config)

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'
# Stage 6 MVP / Stage 7-A critical hash recorded in the real Stage 7-A checkpoints.
STAGE6_MVP_CRITICAL_HASH = 'ef73e8ae2b26ca1b8830f72183b0320d8d1767e7feafdbcfaae0d3485df39dfc'

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig, coordinate_to_action
    from model.evaluator import PolicyValueEvaluator
    from model.network import PolicyValueNet
    from search.alphazero import run_search
    from training.loop import run_training
    from training.self_play import play_self_play_game, replay_record
    from scripts.export_stage7_weights import export_weights


def _board(stones):
    board = [[EMPTY] * 15 for _ in range(15)]
    for (r, c), colour in stones.items():
        board[r][c] = colour
    return board


def _replay(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


class WinningPointsTest(unittest.TestCase):
    def test_black_needs_exact_five_white_accepts_overline(self):
        # X X X X . X on row 7: filling col 4 makes six.
        stones = {(7, c): BLACK for c in (0, 1, 2, 3, 5)}
        self.assertEqual(winning_points(_board(stones), BLACK), [])
        self.assertEqual(winning_points(_board({(7, c): WHITE for c in (0, 1, 2, 3, 5)}),
                                        WHITE), [(7, 4)])
        # Plain four: both ends win.
        four = {(7, c): BLACK for c in (3, 4, 5, 6)}
        self.assertEqual(winning_points(_board(four), BLACK), [(7, 2), (7, 7)])

    def test_board_is_restored(self):
        board = _board({(7, c): WHITE for c in (3, 4, 5, 6)})
        before = [row[:] for row in board]
        winning_points(board, WHITE)
        self.assertEqual(board, before)


class TacticalFilterAgainstProbeLabelsTest(unittest.TestCase):
    """The probe labels were derived independently from the rules engine."""

    @classmethod
    def setUpClass(cls):
        cls.probes = json.loads(PROBES.read_text(encoding='utf-8'))['probes']

    def test_matches_every_labelled_probe(self):
        checked = 0
        for probe in self.probes:
            game = _replay([tuple(m) for m in probe['moves']])
            allowed, proven = tactical_filter(game, game.legal_moves())
            correct = [tuple(m) for m in probe['correct_moves']]
            with self.subTest(probe=probe['id']):
                if probe['kind'] == 'immediate_win':
                    self.assertEqual(sorted(allowed), correct)
                    self.assertEqual(proven, 1.0)
                elif probe['kind'] == 'must_block':
                    self.assertEqual(allowed, correct)
                    self.assertIsNone(proven)
                elif probe['kind'] == 'forced_loss':
                    self.assertEqual(proven, -1.0)
                else:
                    continue
                checked += 1
        self.assertEqual(checked, 120)


class ConfigCompatibilityTest(unittest.TestCase):
    def test_search_config_default_hash_fields_unchanged(self):
        self.assertNotIn('tactical_rules', SearchConfig().to_dict())
        self.assertTrue(SearchConfig(tactical_rules=True).to_dict()['tactical_rules'])
        with self.assertRaises(ValueError):
            SearchConfig(tactical_rules=1)

    def test_existing_configs_keep_their_critical_hash(self):
        for name in ('stage6_mvp.yaml', 'stage7a_continuation.yaml'):
            config = load_config(ROOT / 'configs' / name)
            self.assertEqual(critical_config_hash(config), STAGE6_MVP_CRITICAL_HASH, name)
        # A checkpoint config written before the key existed hashes the same.
        legacy = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        del legacy['self_play']['tactical_rules']
        del legacy['evaluation']['tactical_rules']
        self.assertEqual(critical_config_hash(legacy), STAGE6_MVP_CRITICAL_HASH)

    def test_arms_differ_only_in_the_teacher(self):
        a = load_config(ROOT / 'configs' / 'stage7b_a_scale.yaml')
        b = load_config(ROOT / 'configs' / 'stage7b_b_rules.yaml')
        self.assertEqual(config_differences(critical_config(a), critical_config(b)),
                         ['evaluation.tactical_rules', 'self_play.tactical_rules'])
        self.assertFalse(a['self_play']['tactical_rules'])
        self.assertTrue(b['self_play']['tactical_rules'])
        self.assertEqual(a['training']['init_checkpoint'], b['training']['init_checkpoint'])
        stage7a = load_config(ROOT / 'configs' / 'stage7a_continuation.yaml')
        self.assertEqual(sorted(config_differences(critical_config(stage7a), critical_config(a))),
                         ['self_play.simulations', 'training.games_per_generation'])


@unittest.skipIf(torch is None, 'requires torch')
class PuctV2SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(3)
        cls.evaluator = PolicyValueEvaluator(
            PolicyValueNet(ModelConfig(channels=8, blocks=1, value_hidden=8)).eval())
        cls.probes = json.loads(PROBES.read_text(encoding='utf-8'))['probes']

    def test_root_visits_go_to_forced_moves(self):
        config = SearchConfig(num_simulations=8, temperature_moves=0, noise_enabled=False,
                              tactical_rules=True)
        for kind in ('immediate_win', 'must_block'):
            probe = next(p for p in self.probes if p['kind'] == kind)
            game = _replay([tuple(m) for m in probe['moves']])
            result = run_search(game, self.evaluator, config, None)
            correct = {coordinate_to_action(*m) for m in probe['correct_moves']}
            self.assertEqual(sum(result.visit_counts), 8)
            self.assertEqual(sum(result.visit_counts[a] for a in correct), 8, kind)
            self.assertAlmostEqual(sum(result.priors), 1.0, places=6)

    def test_v2_self_play_record_replays(self):
        config = SearchConfig(num_simulations=6, tactical_rules=True)
        game = play_self_play_game(self.evaluator, config, 5)
        replay_record(game.record, game.final_game)
        self.assertTrue(game.record.search_config['tactical_rules'])


@unittest.skipIf(torch is None, 'requires torch')
class ExportAndInitTest(unittest.TestCase):
    def test_export_round_trip_and_new_v2_run_from_exported_weights(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
                config['training']['generations'] = 1
                run_training(config, run_dir=tmp / 'source', log=lambda _: None)
                source = tmp / 'source' / 'checkpoints' / 'checkpoint_gen001.pt'
                weights = tmp / 'init' / 'weights.pt'
                provenance = export_weights(source, weights)
                self.assertTrue(provenance['round_trip_state_dict_equal'])
                self.assertEqual(provenance['source_generation'], 1)
                self.assertTrue(weights.with_suffix('.pt.json').is_file())

                arm = json.loads(json.dumps(config))
                arm['training']['init_checkpoint'] = str(weights)
                arm['self_play']['tactical_rules'] = True
                arm['evaluation']['tactical_rules'] = True
                state = run_training(arm, run_dir=tmp / 'arm_b', log=lambda _: None)
                self.assertEqual(state.generation, 1)
                initial = torch.load(tmp / 'arm_b' / 'checkpoints' / 'checkpoint_init.pt',
                                     weights_only=True)['model_state_dict']
                exported = load_checkpoint(weights, ModelConfig(**provenance['model_config']))
                for key, value in exported.state_dict().items():
                    self.assertTrue(torch.equal(initial[key], value), key)
                record = json.loads((tmp / 'arm_b' / 'self_play' / 'gen000.json')
                                    .read_text(encoding='utf-8'))
                self.assertIn('tactical_rules', json.dumps(record))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
