"""scripts/forensic_short_games.py on a synthetic 9-ply game (no defence by white)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.threats import ThreatSolver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from forensic_short_games import losing_decision, opening_cluster  # noqa: E402

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

# Black (7,7)..(7,11) wins at ply 9 while white plays on row 0.
MOVES = [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9), (0, 4), (7, 10), (0, 6), (7, 11)]
ACTIONS = [r * 15 + c for r, c in MOVES]


class ForensicTest(unittest.TestCase):
    def test_losing_decision_is_the_last_one_with_a_safe_move(self):
        ply, safe, played = losing_decision(ACTIONS, 1, ThreatSolver(node_limit=20_000))
        self.assertEqual(ply, 5)              # white's 3rd move: black has an open three
        self.assertEqual(played, (0, 4))
        self.assertIn((7, 6), safe)           # blocking an end of the three
        self.assertNotIn((0, 4), safe)

    def test_opening_cluster_is_d4_invariant(self):
        mirrored = [r * 15 + (14 - c) for r, c in MOVES]
        self.assertEqual(opening_cluster(ACTIONS), opening_cluster(mirrored))

    @unittest.skipIf(torch is None, 'requires torch')
    def test_end_to_end_with_a_checkpoint(self):
        from training.config import load_config
        from training.training_checkpoint import build_checkpoint, save_atomic
        from training.training_state import init_training_state

        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            (run / 'self_play').mkdir(parents=True)
            (run / 'self_play' / 'gen000.json').write_text(json.dumps(
                {'games': [{'record': {'moves': ACTIONS, 'winner': 1}}]}))
            checkpoint = Path(tmp) / 'c.pt'
            save_atomic(checkpoint, build_checkpoint(init_training_state(config)))
            output = Path(tmp) / 'out.json'
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'forensic_short_games.py'),
                            str(run), '--checkpoint', str(checkpoint), '--deep-simulations', '8',
                            '--noise-trials', '2', '--output', str(output)],
                           check=True, capture_output=True, cwd=ROOT)
            data = json.loads(output.read_text())
            self.assertEqual(data['short_games'], 1)
            row = data['rows'][0]
            self.assertEqual(row['ply'], 6)
            self.assertIn(row['category'], ('noise', 'base_safe', 'search_budget',
                                            'prior_blind', 'value_blind'))
            for key in ('prior_safe', 'base_visit_safe', 'deep_visit_safe', 'noisy_safe_rate'):
                self.assertGreaterEqual(row[key], 0.0)


if __name__ == '__main__':
    unittest.main()
