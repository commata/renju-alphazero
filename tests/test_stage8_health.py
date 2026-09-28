"""Stage 8-B: execution-only colour health and the Stage 8 continuation config."""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from training.config import (config_differences, critical_config, critical_config_hash,
                             load_config, validate_config)
from training.health import (color_window, detect_color_imbalance, fisher_less_p,
                             update_color_regression)

ROOT = Path(__file__).resolve().parents[1]
D16_HASH = '25ab9c5a870900e4827b71aae63d1509a5c664640f342c616e51c21cbaa7fd4a'

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.loop import run_training
    from training.metrics import read_metrics


def _generations(black_wins, games=16, length=12.0):
    return [{'type': 'generation', 'generation': g, 'self_play_games': games,
             'black_wins': b, 'white_wins': games - b, 'draws': 0,
             'average_game_length': length} for g, b in enumerate(black_wins)]


class Stage8ConfigTest(unittest.TestCase):
    def test_continuation_keeps_the_stage7d_critical_hash(self):
        stage7 = load_config(ROOT / 'configs' / 'stage7d_b16.yaml')
        stage8 = load_config(ROOT / 'configs' / 'stage8_d16.yaml')
        self.assertEqual(critical_config_hash(stage7), D16_HASH)
        self.assertEqual(critical_config_hash(stage8), D16_HASH)
        self.assertTrue(stage8['health']['color_imbalance']['enabled'])
        self.assertEqual(stage8['training']['generations'], 200)
        self.assertEqual(stage8['training']['keep_every'], 20)  # 16 x 20 = 320 games

    def test_health_is_execution_only(self):
        config = load_config(ROOT / 'configs' / 'stage8_d16.yaml')
        changed = deepcopy(config)
        changed['health']['color_imbalance'].update(enabled=False, window=3, upper=0.8)
        self.assertEqual(critical_config_hash(changed), critical_config_hash(config))

    def test_stage7_checkpoint_config_without_health_still_validates(self):
        # A Stage 7 checkpoint stores its resolved config, which has no 'health' key.
        stored = load_config(ROOT / 'configs' / 'stage7d_b16.yaml')
        stored.pop('health')
        validate_config(stored)
        requested = load_config(ROOT / 'configs' / 'stage8_d16.yaml')
        self.assertEqual(config_differences(critical_config(stored),
                                            critical_config(requested)), [])

    def test_invalid_thresholds_rejected(self):
        from training.config import ConfigError
        config = load_config(ROOT / 'configs' / 'stage8_d16.yaml')
        config['health']['color_imbalance'].update(lower=0.9, upper=0.1)
        with self.assertRaises(ConfigError):
            validate_config(config)


class ColorImbalanceTest(unittest.TestCase):
    SETTINGS = {'enabled': True, 'window': 3, 'lower': 0.10, 'upper': 0.90}

    def test_window_statistics(self):
        events = _generations([8, 16, 15, 1])
        self.assertIsNone(color_window(events, 1, 3))
        stats = color_window(events, 3, 3)
        self.assertEqual(stats['generations'], [1, 3])
        self.assertEqual(stats['games'], 48)
        self.assertAlmostEqual(stats['black_win_rate'], 32 / 48)
        self.assertAlmostEqual(stats['white_win_rate'] + stats['black_win_rate'], 1)

    def test_warning_once_per_window_and_streak(self):
        events = _generations([16, 15, 15, 16, 15, 16, 15])  # >= 90% black throughout
        warnings = []
        fired = []
        for gen in range(len(events)):
            health, warning = detect_color_imbalance(events, warnings, gen, self.SETTINGS)
            if gen < 2:
                self.assertIsNone(health)
                continue
            self.assertEqual(health['extreme'], 'black')
            self.assertEqual(health['extreme_generations'], gen - 1)
            if warning:
                warnings.append({'type': 'health_warning', 'generation': gen, **warning})
                fired.append(gen)
        self.assertEqual(fired, [2, 5])  # suppressed for `window` generations

    def test_white_direction_and_normal_window(self):
        events = _generations([1, 0, 1, 8, 8, 8])
        _, warning = detect_color_imbalance(events, [], 2, self.SETTINGS)
        self.assertEqual(warning['dominant'], 'white')
        health, warning = detect_color_imbalance(events, [], 5, self.SETTINGS)
        self.assertIsNone(health['extreme'])
        self.assertIsNone(warning)


class ColorRegressionTest(unittest.TestCase):
    def test_fisher_matches_stage7_value(self):
        # Stage 7-D: D32 gen 80 white 1/25 vs 7-C gen 110 white 10/25 -> p = 0.0023
        self.assertAlmostEqual(fisher_less_p(1, 25, 10, 25), 0.0023, places=4)
        self.assertEqual(fisher_less_p(25, 25, 0, 25), 1.0)

    def _run(self, wins):
        state, statuses = None, []
        for g, w in enumerate(wins):
            event, state = update_color_regression(state, {'generation': g, 'wins': w,
                                                           'games': 25})
            statuses.append(event['status'])
        return statuses, state

    def test_sustained_collapse_is_confirmed_against_the_same_reference(self):
        # D32 white vs MCTS-v2: 17 -> 3 -> 1 (the second step alone, 3 -> 1, is not significant)
        self.assertGreater(fisher_less_p(1, 25, 3, 25), 0.05)
        statuses, state = self._run([17, 3, 1])
        self.assertEqual(statuses, ['healthy', 'candidate', 'confirmed'])
        self.assertEqual(state['reference']['wins'], 17)

    def test_d16_is_healthy_and_one_dip_recovers(self):
        self.assertEqual(self._run([21, 17, 23])[0], ['healthy'] * 3)   # D16 white
        self.assertEqual(self._run([18, 22, 20])[0], ['healthy'] * 3)   # D16 black
        self.assertEqual(self._run([20, 9, 19, 18])[0],
                         ['healthy', 'candidate', 'healthy', 'healthy'])

    def test_gradual_slide_cannot_lower_the_reference(self):
        statuses, state = self._run([17, 14, 11, 8, 6])
        self.assertEqual(state['reference']['wins'], 17)
        self.assertEqual(statuses[-1], 'confirmed')


@unittest.skipIf(torch is None, 'requires torch')
class HealthInLoopTest(unittest.TestCase):
    def test_loop_logs_health_events_without_touching_training(self):
        threads = torch.get_num_threads()
        base = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        monitored = deepcopy(base)
        # window 1, lower 0.99: every non-all-black generation is 'white' dominant
        monitored['health']['color_imbalance'].update(enabled=True, window=1, lower=0.99,
                                                      upper=1.0)
        self.assertEqual(critical_config_hash(monitored), critical_config_hash(base))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                runs = {}
                for name, config in (('base', base), ('monitored', monitored)):
                    runs[name] = run_training(config, run_dir=Path(tmp) / name,
                                              log=lambda message: None)
                events = read_metrics(Path(tmp) / 'monitored' / 'metrics.jsonl')
                health = [e for e in events if e['type'] == 'health']
                self.assertEqual([e['generation'] for e in health], [0, 1])
                warnings = [e for e in events if e['type'] == 'health_warning']
                self.assertTrue(warnings)
                self.assertFalse([e for e in read_metrics(Path(tmp) / 'base' / 'metrics.jsonl')
                                  if e['type'].startswith('health')])
                # monitoring never changes the trained model
                for a, b in zip(runs['base'].model.state_dict().values(),
                                runs['monitored'].model.state_dict().values()):
                    self.assertTrue(torch.equal(a, b))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
