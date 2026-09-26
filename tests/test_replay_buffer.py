import random
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.replay_buffer import ReplayBuffer, TrainingSample


def make_sample(tag: int) -> 'TrainingSample':
    """Distinguishable synthetic sample; legal action = tag % 225."""
    state = torch.zeros((6, 15, 15), dtype=torch.float32)
    state[0].view(-1)[tag % 225] = 1
    state[4].fill_(1)
    mask = torch.zeros(225, dtype=torch.bool)
    mask[tag % 225] = True
    state[5] = mask.reshape(15, 15).float()
    policy = mask.float()
    return TrainingSample(state, policy, float(tag % 3 - 1), mask, generation_id=tag // 10,
                          game_id=tag % 10, ply=tag)


@unittest.skipIf(torch is None, 'requires torch')
class ReplayBufferTest(unittest.TestCase):
    def test_fifo_capacity_and_order(self):
        buffer = ReplayBuffer(4)
        buffer.extend(make_sample(i) for i in range(6))
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.total_added, 6)
        batch = buffer.get(range(4))
        self.assertEqual(batch.provenance[:, 2].tolist(), [2, 3, 4, 5])  # oldest first
        buffer.append(make_sample(6))
        self.assertEqual(buffer.get(range(4)).provenance[:, 2].tolist(), [3, 4, 5, 6])

    def test_batch_contents_and_shapes(self):
        buffer = ReplayBuffer(8)
        samples = [make_sample(i) for i in range(3)]
        buffer.extend(samples)
        batch = buffer.get([2, 0])
        self.assertEqual(tuple(batch.states.shape), (2, 6, 15, 15))
        self.assertEqual(batch.states.dtype, torch.float32)
        self.assertEqual(tuple(batch.values.shape), (2, 1))
        self.assertTrue(torch.equal(batch.states[0], samples[2].state))
        self.assertTrue(torch.equal(batch.policies[1], samples[0].policy))
        self.assertTrue(torch.equal(batch.legal_masks[0], samples[2].legal_mask))
        self.assertEqual(batch.values[:, 0].tolist(), [samples[2].value, samples[0].value])

    def test_non_binary_state_rejected(self):
        sample = make_sample(1)
        sample.state[0, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            ReplayBuffer(2).append(sample)

    def test_sampling_is_deterministic_with_external_rng(self):
        buffer = ReplayBuffer(16)
        buffer.extend(make_sample(i) for i in range(10))
        a = buffer.sample_indices(32, rng=random.Random(5))
        b = buffer.sample_indices(32, rng=random.Random(5))
        self.assertEqual(a, b)
        self.assertTrue(all(0 <= i < 10 for i in a))
        self.assertGreater(len(a), len(set(a)))  # with replacement
        with self.assertRaises(TypeError):
            buffer.sample_indices(4, rng=None)

    def test_sampling_does_not_touch_global_random(self):
        buffer = ReplayBuffer(16)
        buffer.extend(make_sample(i) for i in range(10))
        before = random.getstate()
        buffer.sample_indices(64, rng=random.Random(1))
        self.assertEqual(before, random.getstate())

    def test_empty_buffer_sampling_rejected(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(4).sample_indices(1, rng=random.Random(0))

    def test_state_dict_round_trip_after_wraparound(self):
        buffer = ReplayBuffer(5)
        buffer.extend(make_sample(i) for i in range(12))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'buffer.pt'
            torch.save(buffer.state_dict(), path)
            restored = ReplayBuffer(5)
            restored.load_state_dict(torch.load(path, weights_only=True))
        self.assertTrue(buffer.equals(restored))
        self.assertEqual(restored.total_added, 12)
        # Logical indices map to the same samples after reload.
        indices = buffer.sample_indices(20, rng=random.Random(3))
        self.assertEqual(indices, restored.sample_indices(20, rng=random.Random(3)))
        self.assertTrue(torch.equal(buffer.get(indices).states, restored.get(indices).states))
        # Order continues identically after further appends.
        buffer.extend(make_sample(i) for i in range(12, 15))
        restored.extend(make_sample(i) for i in range(12, 15))
        self.assertTrue(buffer.equals(restored))

    def test_equals_detects_order(self):
        a, b = ReplayBuffer(4), ReplayBuffer(4)
        a.extend([make_sample(1), make_sample(2)])
        b.extend([make_sample(2), make_sample(1)])
        self.assertFalse(a.equals(b))

    def test_capacity_mismatch_rejected(self):
        buffer = ReplayBuffer(4)
        buffer.append(make_sample(0))
        with self.assertRaises(ValueError):
            ReplayBuffer(8).load_state_dict(buffer.state_dict())


if __name__ == '__main__':
    unittest.main()
