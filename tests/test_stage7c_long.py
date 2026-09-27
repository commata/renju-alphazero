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
