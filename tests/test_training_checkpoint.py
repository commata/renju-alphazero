import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.dataset import build_batch
    from training.replay_buffer import TrainingSample
    from training.trainer import train_step
    from training.training_checkpoint import (
        CheckpointCompatibilityError, INIT_NAME, LATEST_NAME, build_checkpoint, copy_atomic,
        generation_checkpoint_name, load_checkpoint_payload, load_training_state,
        prune_checkpoints, save_atomic)
    from training.training_state import COMPONENT_RNGS, init_training_state


def test_config(**changes):
    config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
    for dotted, value in changes.items():
        node = config
        *parents, leaf = dotted.split('__')
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config


def synthetic_sample(tag):
    state = torch.zeros((6, 15, 15))
    state[4].fill_(1)
    mask = torch.zeros(225, dtype=torch.bool)
    mask[[tag % 225, (tag + 7) % 225]] = True
    state[5] = mask.reshape(15, 15).float()
    return TrainingSample(state, mask.float() / 2, float(tag % 3 - 1), mask, 0, tag, 0)


def advanced_state():
    torch.set_num_threads(1)
    state = init_training_state(test_config())
    state.buffer.extend(synthetic_sample(i) for i in range(30))
    for _ in range(3):
        batch = build_batch(state.buffer, 8, sample_rng=state.sample_rng,
                            augment_rng=state.augment_rng, augment=True)
        train_step(state.model, state.optimizer, batch, 1.0)
        state.global_step += 1
    state.self_play_rng.getrandbits(64)
    state.generation = 2
    return state


def assert_states_equal(test, a, b):
    for (name, x), (_, y) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        test.assertTrue(torch.equal(x, y), name)
    oa, ob = a.optimizer.state_dict(), b.optimizer.state_dict()
    test.assertEqual(oa['param_groups'], ob['param_groups'])
    for key in oa['state']:
        for field, value in oa['state'][key].items():
            test.assertTrue(torch.equal(value, ob['state'][key][field]), (key, field))
    test.assertTrue(a.buffer.equals(b.buffer))
    test.assertEqual((a.generation, a.global_step), (b.generation, b.global_step))
    for name in COMPONENT_RNGS:
        test.assertEqual(getattr(a, name).getstate(), getattr(b, name).getstate(), name)


@unittest.skipIf(torch is None, 'requires torch')
class TrainingCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_restores_everything(self):
        state = advanced_state()
        path = self.dir / generation_checkpoint_name(2)
        save_atomic(path, build_checkpoint(state))
        global_before = (random.getstate(), torch.get_rng_state())
        random.random()
        torch.rand(3)  # disturb global RNGs; load must restore them
        restored = load_training_state(path)
        self.assertEqual(random.getstate(), global_before[0])
        self.assertTrue(torch.equal(torch.get_rng_state(), global_before[1]))
        assert_states_equal(self, state, restored)
        self.assertTrue(restored.model.training)
        # Both continue identically.
        for s in (state, restored):
            batch = build_batch(s.buffer, 8, sample_rng=s.sample_rng,
                                augment_rng=s.augment_rng, augment=True)
            train_step(s.model, s.optimizer, batch, 1.0)
        assert_states_equal(self, state, restored)

    def test_payload_is_weights_only_loadable_with_metadata(self):
        state = advanced_state()
        path = self.dir / 'c.pt'
        digest = save_atomic(path, build_checkpoint(state))
        payload = torch.load(path, weights_only=True)
        for key in ('model_state_dict', 'optimizer_state_dict', 'replay_buffer', 'global_rng',
                    'component_rng', 'config', 'critical_config_hash', 'git_commit',
                    'git_dirty', 'device', 'model_config', 'model_contract',
                    'python_version', 'torch_version'):
            self.assertIn(key, payload)
        self.assertIsNone(payload['scheduler_state_dict'])
        self.assertEqual(payload['generation'], 2)
        self.assertEqual(len(digest), 64)

    def test_failed_save_preserves_existing_file(self):
        path = self.dir / LATEST_NAME
        save_atomic(path, {'value': 1})
        original = path.read_bytes()

        def broken_save(obj, handle):
            handle.write(b'partial')
            raise OSError('disk full')

        with patch('torch.save', broken_save):
            with self.assertRaises(OSError):
                save_atomic(path, {'value': 2})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), [LATEST_NAME])

    def test_failed_copy_preserves_latest(self):
        source, latest = self.dir / 'gen.pt', self.dir / LATEST_NAME
        save_atomic(source, {'value': 2})
        save_atomic(latest, {'value': 1})
        original = latest.read_bytes()
        with patch('shutil.copyfileobj', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                copy_atomic(source, latest)
        self.assertEqual(latest.read_bytes(), original)
        copy_atomic(source, latest)
        self.assertEqual(latest.read_bytes(), source.read_bytes())

    def test_config_mismatch_is_rejected(self):
        path = self.dir / 'c.pt'
        save_atomic(path, build_checkpoint(advanced_state()))
        for change in ({'optimizer__lr': 0.5}, {'model__channels': 16},
                       {'training__replay_capacity': 999}, {'seed': 1}):
            with self.assertRaises(ConfigError, msg=str(change)):
                load_training_state(path, test_config(**change))
        # Execution-control values may change.
        restored = load_training_state(path, test_config(training__generations=9))
        self.assertEqual(restored.config['training']['generations'], 9)

    def test_architecture_and_format_mismatch_are_rejected(self):
        payload = build_checkpoint(advanced_state())
        payload['model_config'] = dict(payload['model_config'], channels=16)
        path = self.dir / 'arch.pt'
        save_atomic(path, payload)
        with self.assertRaises(CheckpointCompatibilityError):
            load_training_state(path)
        payload = build_checkpoint(advanced_state())
        payload['format_version'] = 'stage6-training-checkpoint-v0'
        save_atomic(path, payload)
        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint_payload(path)
        payload = build_checkpoint(advanced_state())
        payload['model_contract'] = dict(payload['model_contract'], encoder_version='other')
        save_atomic(path, payload)
        with self.assertRaises(CheckpointCompatibilityError):
            load_checkpoint_payload(path)

    def test_prune_keeps_init_latest_and_newest(self):
        for name in [INIT_NAME, LATEST_NAME] + [generation_checkpoint_name(g) for g in range(1, 6)]:
            (self.dir / name).write_bytes(b'x')
        removed = prune_checkpoints(self.dir, 2)
        self.assertEqual(sorted(p.name for p in removed),
                         [generation_checkpoint_name(g) for g in (1, 2, 3)])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         sorted([INIT_NAME, LATEST_NAME, generation_checkpoint_name(4),
                                 generation_checkpoint_name(5)]))


if __name__ == '__main__':
    unittest.main()
