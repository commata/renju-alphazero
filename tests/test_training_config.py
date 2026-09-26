import subprocess
import sys
import unittest
from pathlib import Path

from training.config import (ConfigError, critical_config_hash, config_differences,
                             evaluation_search_config, load_config, resolve_config,
                             self_play_search_config)

ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTest(unittest.TestCase):
    def test_repository_configs_load(self):
        for name in ('stage6_mvp.yaml', 'stage6_test.yaml'):
            config = load_config(ROOT / 'configs' / name)
            self.assertEqual(config['self_play']['max_moves'], 225)

    def test_loading_config_does_not_import_torch(self):
        # Config handling is torch-free (engine-only installs can still validate configs).
        code = ('import sys; from training.config import load_config; '
                f'load_config({str(ROOT / "configs" / "stage6_test.yaml")!r}); '
                "assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)

    def test_mvp_self_play_uses_stage5_defaults(self):
        config = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        search = self_play_search_config(config)
        self.assertEqual(search.num_simulations, 25)
        self.assertTrue(search.noise_enabled)
        self.assertEqual((search.temperature_moves, search.dirichlet_alpha,
                          search.dirichlet_epsilon), (10, 0.05, 0.25))

    def test_evaluation_search_is_noise_free_argmax(self):
        search = evaluation_search_config(resolve_config({}))
        self.assertFalse(search.noise_enabled)
        self.assertEqual(search.temperature_moves, 0)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_config({'training': {'batchsize': 4}})

    def test_weight_decay_and_l2_together_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_config({'optimizer': {'weight_decay': 1e-4}, 'loss': {'l2_coeff': 1e-4}})
        resolve_config({'optimizer': {'weight_decay': 0.0}, 'loss': {'l2_coeff': 1e-4}})

    def test_truncation_not_supported(self):
        with self.assertRaises(ConfigError):
            resolve_config({'self_play': {'max_moves': 30}})

    def test_unbalanced_colours_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_config({'evaluation': {'random': {'black_games': 2, 'white_games': 1}}})

    def test_v6_must_stay_frozen(self):
        with self.assertRaises(ConfigError):
            resolve_config({'evaluation': {'mcts_v6': {'use_frozen_config': False}}})

    def test_critical_hash_ignores_execution_control_only(self):
        base = resolve_config({})
        control = resolve_config({'training': {'generations': 9, 'keep_checkpoints': 1},
                                  'output': {'run_name': 'x'}, 'torch_threads': 2})
        self.assertEqual(critical_config_hash(base), critical_config_hash(control))
        for change in ({'seed': 1}, {'optimizer': {'lr': 0.01}},
                       {'model': {'channels': 32}}, {'training': {'replay_capacity': 5}},
                       {'self_play': {'simulations': 5}},
                       {'evaluation': {'puct_simulations': 5}}):
            self.assertNotEqual(critical_config_hash(base),
                                critical_config_hash(resolve_config(change)), change)

    def test_config_differences(self):
        a = resolve_config({})
        b = resolve_config({'optimizer': {'lr': 0.5}})
        self.assertEqual(config_differences(a, b), ['optimizer.lr'])


if __name__ == '__main__':
    unittest.main()
