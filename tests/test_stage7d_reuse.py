"""Stage 7-D: D16 vs D32 differ only in games per generation."""
import unittest
from pathlib import Path

from training.config import config_differences, critical_config, load_config

ROOT = Path(__file__).resolve().parents[1]


class ReuseArmsTest(unittest.TestCase):
    def test_single_critical_difference_and_equal_total_games(self):
        d16 = load_config(ROOT / 'configs' / 'stage7d_b16.yaml')
        d32 = load_config(ROOT / 'configs' / 'stage7d_b32.yaml')
        self.assertEqual(config_differences(critical_config(d16), critical_config(d32)),
                         ['training.games_per_generation'])
        for config in (d16, d32):
            t = config['training']
            self.assertEqual(t['games_per_generation'] * t['generations'], 2560)
            # checkpoints kept at the same cumulative game counts (every 320 games)
            self.assertEqual(t['games_per_generation'] * t['keep_every'], 320)
            self.assertTrue(config['self_play']['tactical_rules'])
            self.assertEqual(t['init_checkpoint'], 'runs/stage7d_init/stage7c_b_gen110_weights.pt')

    def test_same_teacher_and_model_as_stage7b_arm_b(self):
        b = load_config(ROOT / 'configs' / 'stage7b_b_rules.yaml')
        d16 = load_config(ROOT / 'configs' / 'stage7d_b16.yaml')
        self.assertEqual(config_differences(critical_config(b), critical_config(d16)), [])


if __name__ == '__main__':
    unittest.main()
