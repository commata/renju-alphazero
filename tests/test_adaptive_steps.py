"""training.adaptive_steps: SGD steps follow the fresh samples of each generation."""
from copy import deepcopy
from pathlib import Path
import unittest

from training.config import (ConfigError, critical_config_hash, load_config, training_steps,
                             validate_config)

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveStepsTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs' / 'stage8_s640_temp2.yaml')

    def test_default_is_fixed_and_keeps_the_hash(self):
        self.assertIsNone(self.config['training']['adaptive_steps'])
        self.assertEqual(training_steps(self.config, 10), 50)
        explicit = deepcopy(self.config)
        explicit['training']['adaptive_steps'] = None
        self.assertEqual(critical_config_hash(explicit), critical_config_hash(self.config))

    def test_steps_follow_fresh_samples_and_clamp(self):
        config = deepcopy(self.config)
        config['training']['adaptive_steps'] = {'target_reuse': 6.4, 'min_steps': 20,
                                                'max_steps': 100}
        validate_config(config)
        self.assertEqual(training_steps(config, 250), 50)    # 6.4 * 250 / 32
        self.assertEqual(training_steps(config, 60), 20)     # clamped up
        self.assertEqual(training_steps(config, 1000), 100)  # clamped down
        self.assertNotEqual(critical_config_hash(config), critical_config_hash(self.config))

    def test_invalid_settings_are_rejected(self):
        for bad in ({'target_reuse': 0, 'min_steps': 1, 'max_steps': 2},
                    {'target_reuse': 6.4, 'min_steps': 5, 'max_steps': 2},
                    {'target_reuse': 6.4}):
            config = deepcopy(self.config)
            config['training']['adaptive_steps'] = bad
            with self.assertRaises(ConfigError):
                validate_config(config)

    def test_adaptive_config_differs_from_s2_only_in_adaptive_steps(self):
        from training.config import config_differences, critical_config

        adaptive = load_config(ROOT / 'configs' / 'stage8_s2_adaptive.yaml')
        self.assertEqual(config_differences(critical_config(self.config), critical_config(adaptive)),
                         ['training.adaptive_steps'])


if __name__ == '__main__':
    unittest.main()
