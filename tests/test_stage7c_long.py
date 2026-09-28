"""Stage 7-C long continuation: configs resume the Stage 7-B arms; periodic keep."""
import json
import tempfile
import unittest
from pathlib import Path

from training.config import critical_config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFENSE = ROOT / 'tests' / 'fixtures' / 'stage7_probes_defense_v1.json'

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.training_checkpoint import (INIT_NAME, LATEST_NAME,
                                              generation_checkpoint_name, prune_checkpoints)


class LongContinuationConfigTest(unittest.TestCase):
    def test_same_critical_hash_as_stage7b_arms(self):
        for arm in ('b_rules', 'a_scale'):
            base = load_config(ROOT / 'configs' / f'stage7b_{arm}.yaml')
            long = load_config(ROOT / 'configs' / f'stage7c_{arm}_long.yaml')
            self.assertEqual(critical_config_hash(base), critical_config_hash(long), arm)
            self.assertEqual(long['training']['generations'], 110)
            self.assertEqual(long['training']['keep_every'], 10)

    def test_keep_every_does_not_change_existing_hashes(self):
        config = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        before = critical_config_hash(config)
        config['training']['keep_every'] = 10
        self.assertEqual(critical_config_hash(config), before)


def _evals(opponent, wins_by_generation, games=4):
    return [{'type': 'evaluation', 'generation': g, 'opponent': opponent, 'games': games,
             'wins': w, 'losses': games - w, 'draws': 0}
            for g, w in wins_by_generation.items()]


class MilestoneDetectionTest(unittest.TestCase):
    SETTINGS = {'window': 3, 'threshold': 0.25, 'opponents': ['tactical'],
                'first_win': ['tactical']}

    def test_jump_first_win_and_cooldown(self):
        from training.milestones import detect_milestones

        # gens 0-2: 0/12, gens 3-5: 1+3+4 = 8/12 -> +0.67 at gen 5
        evaluations = _evals('tactical', {0: 0, 1: 0, 2: 0, 3: 1, 4: 3, 5: 4, 6: 4})
        self.assertEqual([m['kind'] for m in detect_milestones(evaluations, [], 3,
                                                                self.SETTINGS)],
                         ['first_win'])
        jumps = detect_milestones(evaluations, [], 5, self.SETTINGS)
        self.assertEqual([m['kind'] for m in jumps], ['win_rate_jump'])
        self.assertAlmostEqual(jumps[0]['delta'], 8 / 12)
        self.assertEqual(jumps[0]['generations'], [3, 5])
        logged = [{'generation': 5, **jumps[0]}]
        # the sliding window would repeat the jump at gen 6: suppressed by the cooldown
        self.assertEqual(detect_milestones(evaluations, logged, 6, self.SETTINGS), [])

    def test_continuation_configs_enable_monitoring_only(self):
        for arm in ('b_rules', 'a_scale'):
            config = load_config(ROOT / 'configs' / f'stage7c_{arm}_long.yaml')
            self.assertTrue(config['milestones']['enabled'])
            config['milestones']['enabled'] = False
            self.assertEqual(critical_config_hash(config), critical_config_hash(
                load_config(ROOT / 'configs' / f'stage7b_{arm}.yaml')))


class DefenseProbeFixtureTest(unittest.TestCase):
    def test_labels_are_exact_rule_facts(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from build_stage7_defense_probes import defense_label
        from renju import Game

        data = json.loads(DEFENSE.read_text(encoding='utf-8'))
        self.assertEqual(data['format'], 'stage7-probes-v1')
        self.assertEqual(data['counts'], {'BLACK': 20, 'WHITE': 20})
        for probe in data['probes'][::5]:  # every 5th keeps the test fast
            game = Game()
            for move in probe['moves']:
                game.play(*move)
            with self.subTest(probe=probe['id']):
                self.assertEqual(defense_label(game),
                                 [tuple(m) for m in probe['correct_moves']])
                self.assertLess(len(probe['correct_moves']), len(game.legal_moves()))


@unittest.skipIf(torch is None, 'requires torch')
class MilestonePinTest(unittest.TestCase):
    def test_record_milestones_pins_checkpoint_and_logs(self):
        from types import SimpleNamespace

        from training.loop import record_milestones
        from training.metrics import MetricsLogger, read_metrics

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'checkpoints').mkdir()
            checkpoint = run / 'checkpoints' / generation_checkpoint_name(6)
            checkpoint.write_bytes(b'weights')
            metrics = MetricsLogger(run / 'metrics.jsonl')
            for event in _evals('tactical', {0: 0, 1: 0, 2: 0, 3: 1, 4: 3, 5: 4}):
                metrics.log(event)
            state = SimpleNamespace(generation=6, config={'milestones': {
                'enabled': True, 'window': 3, 'threshold': 0.25,
                'opponents': ['tactical'], 'first_win': []}})
            found = record_milestones(state, run, metrics, 5, checkpoint, lambda _: None)
            self.assertEqual([m['kind'] for m in found], ['win_rate_jump'])
            pinned = run / 'checkpoints' / 'milestone_gen006.pt'
            self.assertEqual(pinned.read_bytes(), b'weights')
            logged = [e for e in read_metrics(run / 'metrics.jsonl') if e['type'] == 'milestone']
            self.assertEqual(logged[0]['checkpoint'], 'checkpoints/milestone_gen006.pt')
            # pruning never removes the pin
            prune_checkpoints(run / 'checkpoints', 0)
            self.assertTrue(pinned.is_file())


@unittest.skipIf(torch is None, 'requires torch')
class PruneKeepEveryTest(unittest.TestCase):
    def test_periodic_checkpoints_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            names = [INIT_NAME, LATEST_NAME] + [generation_checkpoint_name(g)
                                                for g in range(1, 26)]
            for name in names:
                (directory / name).write_bytes(b'x')
            prune_checkpoints(directory, 3, keep_every=10)
            kept = sorted(p.name for p in directory.iterdir())
            expected = sorted([INIT_NAME, LATEST_NAME] + [generation_checkpoint_name(g)
                                                          for g in (10, 20, 23, 24, 25)])
            self.assertEqual(kept, expected)


if __name__ == '__main__':
    unittest.main()
