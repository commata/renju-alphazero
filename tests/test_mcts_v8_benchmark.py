import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.run_mcts_v8_benchmark import ARMS, build_tasks, main

SMOKE = ['--pairs', '1', '--seed', '3', '--simulations', '4', '--tactical-simulations', '8']


def _args(arm):
    return SimpleNamespace(arm=arm, pairs=2, seed=11, opening_random_plies=2, opening_radius=2,
                           counterfactual=False, search_overrides={})


class BenchmarkTaskTest(unittest.TestCase):
    def test_arms_share_openings_and_seeds(self):
        # Paired ablation: only the arm (V8 configuration) may differ between runs.
        base_tasks, base_openings = build_tasks(_args('full'))
        for arm in ARMS:
            tasks, openings = build_tasks(_args(arm))
            self.assertEqual(openings, base_openings)
            self.assertEqual([(t['pair'], t['seed'], t['v8_color'], t['opening']) for t in tasks],
                             [(t['pair'], t['seed'], t['v8_color'], t['opening']) for t in base_tasks])
        self.assertEqual(len(base_tasks), 4)
        self.assertEqual({t['v8_color'] for t in base_tasks}, {1, -1})


class BenchmarkRunTest(unittest.TestCase):
    def test_smoke_run_resume_and_summary(self):
        # The "off" arm is V7 inside V8's code path: cheap, and exercises the full runner.
        with TemporaryDirectory() as tmp:
            jsonl, first, second = Path(tmp, 'g.jsonl'), Path(tmp, 'a.json'), Path(tmp, 'b.json')
            main(['--arm', 'off', *SMOKE, '--games-jsonl', str(jsonl), '--output', str(first)])
            main(['--arm', 'off', *SMOKE, '--games-jsonl', str(jsonl), '--output', str(second)])
            self.assertEqual(len(jsonl.read_text(encoding='utf-8').splitlines()), 2)  # resumed, not replayed
            a = json.loads(first.read_text(encoding='utf-8'))
            b = json.loads(second.read_text(encoding='utf-8'))
            self.assertEqual(a['summary']['outcome_history_sha256'], b['summary']['outcome_history_sha256'])
            summary = a['summary']
            self.assertEqual(summary['games'], 2)
            self.assertEqual(summary['wins'] + summary['draws'] + summary['losses'], 2)
            self.assertEqual(summary['v8_b_attack']['ran'], 0)  # V8-B off
            self.assertNotIn('own_vct', summary['v8_routes'])
            self.assertEqual(a['v8_config']['own_vct_attack'], False)
            self.assertEqual(a['v8_config']['root_vct_safety'], False)
            self.assertEqual(summary['v8_c_root']['ran'], 0)
            self.assertEqual(a['v8_config']['simulations'], 4)
            for game in a['games']:
                for move in game['v8_moves']:
                    self.assertEqual(move['played'], game['moves'][move['ply']])


if __name__ == '__main__':
    unittest.main()
