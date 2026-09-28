"""Stage 8 Gate 3: colour-balanced sampling, arm configs, new-run orchestration, head-to-head."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random

from training.config import (config_differences, critical_config, critical_config_hash,
                             load_config)

ROOT = Path(__file__).resolve().parents[1]
D16_HASH = '25ab9c5a870900e4827b71aae63d1509a5c664640f342c616e51c21cbaa7fd4a'

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.replay_buffer import ReplayBuffer, TrainingSample


class ArmConfigTest(unittest.TestCase):
    def test_arms_differ_from_control_in_one_critical_key(self):
        arms = {arm: load_config(ROOT / 'configs' / f'stage8_g3_{arm}.yaml') for arm in 'cbw'}
        self.assertEqual(critical_config_hash(arms['c']), D16_HASH)
        self.assertEqual(config_differences(critical_config(arms['c']), critical_config(arms['b'])),
                         ['training.balanced_sampling'])
        self.assertEqual(config_differences(critical_config(arms['c']), critical_config(arms['w'])),
                         ['training.replay_capacity'])
        self.assertTrue(arms['b']['training']['balanced_sampling'])
        self.assertEqual(arms['w']['training']['replay_capacity'], 40000)
        for config in arms.values():
            self.assertEqual(config['training']['init_checkpoint'],
                             'runs/stage8_init/stage8_d16_gen200_weights.pt')
            self.assertEqual(config['training']['keep_every'], 20)
            self.assertTrue(config['flush_denormal'])

    def test_balanced_sampling_default_keeps_old_hashes(self):
        config = load_config(ROOT / 'configs' / 'stage7d_b16.yaml')
        self.assertFalse(config['training']['balanced_sampling'])
        self.assertEqual(critical_config_hash(config), D16_HASH)


def _sample(black_to_move: bool, value: float, tag: int):
    state = torch.zeros(6, 15, 15)
    state[3].fill_(1.0 if black_to_move else 0.0)
    state[5, 0, tag % 15] = 1.0
    mask = torch.zeros(225, dtype=torch.bool)
    mask[tag % 15] = True
    policy = torch.zeros(225)
    policy[tag % 15] = 1.0
    return TrainingSample(state, policy, value, mask, 0, tag, 0)


@unittest.skipIf(torch is None, 'requires torch')
class BalancedSamplingTest(unittest.TestCase):
    def _buffer(self, black_games: int, white_games: int, capacity: int = 1000):
        buffer = ReplayBuffer(capacity)
        tag = 0
        for winner, count in (('black', black_games), ('white', white_games)):
            for _ in range(count):
                # one black-to-move and one white-to-move sample per game
                sign = 1.0 if winner == 'black' else -1.0
                buffer.extend([_sample(True, sign, tag), _sample(False, -sign, tag + 1)])
                tag += 2
        return buffer

    def test_winner_groups_follow_side_to_move_and_target(self):
        buffer = self._buffer(black_games=9, white_games=1)
        black, white = buffer.winner_groups()
        self.assertEqual((len(black), len(white)), (18, 2))
        self.assertEqual(sorted(black + white), list(range(20)))

    def test_half_of_every_batch_from_each_winner(self):
        buffer = self._buffer(black_games=49, white_games=1)   # 98% black wins
        black, white = map(set, buffer.winner_groups())
        indices, info = buffer.sample_indices_balanced(32, rng=Random(1))
        self.assertTrue(info['balanced'])
        self.assertEqual(sum(i in black for i in indices), 16)
        self.assertEqual(sum(i in white for i in indices), 16)
        # deterministic given the rng
        self.assertEqual(indices, buffer.sample_indices_balanced(32, rng=Random(1))[0])

    def test_one_sided_buffer_falls_back_to_uniform(self):
        buffer = self._buffer(black_games=5, white_games=0)
        indices, info = buffer.sample_indices_balanced(8, rng=Random(2))
        self.assertFalse(info['balanced'])
        self.assertEqual(indices, buffer.sample_indices(8, rng=Random(2)))

    def test_groups_track_the_fifo_window(self):
        buffer = self._buffer(black_games=3, white_games=3, capacity=6)   # oldest evicted
        black, white = buffer.winner_groups()
        self.assertEqual((len(black), len(white)), (0, 6))


@unittest.skipIf(torch is None, 'requires torch')
class NewRunAndHeadToHeadTest(unittest.TestCase):
    def test_new_run_orchestration_and_round_robin(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from run_stage8_training import build_parser, orchestrate

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=2, keep_every=1, balanced_sampling=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / 'arm'
                args = build_parser().parse_args([
                    '--run-dir', str(run_dir), '--new-run', '--anchor', '0', '--light-every', '1',
                    '--heavy-every', '2', '--light-opponents', 'random', '--light-pairs', '1',
                    '--heavy-opponents', 'random', '--heavy-pairs', '1', '--skip-probes'])
                self.assertEqual(orchestrate(run_dir, config, args, log=lambda m: None), 2)
                out = run_dir / 'external_eval'
                # generation 0 = the new run's init checkpoint
                for name in ('gen000_light.json', 'gen000_heavy.json', 'gen001_light.json',
                             'gen002_light.json', 'gen002_heavy.json'):
                    self.assertTrue((out / name).is_file(), name)
                events = [json.loads(line) for line in
                          (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                generations = [e for e in events if e['type'] == 'generation']
                self.assertEqual(len(generations), 2)
                self.assertIn('balanced_sampling', generations[-1])

                ckpt = run_dir / 'checkpoints'
                result = subprocess.run(
                    [sys.executable, str(ROOT / 'scripts' / 'run_stage8_head_to_head.py'),
                     '--checkpoint', f"g1={ckpt / 'checkpoint_gen001.pt'}",
                     '--checkpoint', f"g2={ckpt / 'checkpoint_gen002.pt'}",
                     '--pairs', '1', '--output', str(Path(tmp) / 'h2h.json')],
                    capture_output=True, text=True, timeout=600,
                    env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
                self.assertEqual(result.returncode, 0, result.stderr)
                h2h = json.loads((Path(tmp) / 'h2h.json').read_text(encoding='utf-8'))
                match = h2h['matches'][0]['summary']
                self.assertEqual(match['games'], 2)
                self.assertEqual(match['a_wins'] + match['a_losses'] + match['draws'], 2)
                self.assertEqual(h2h['totals']['g1']['games'], 2)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
