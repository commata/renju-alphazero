from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    import model.checkpoint as checkpoint_module
    from model.checkpoint import _git_provenance, load_checkpoint, save_checkpoint
    from model.config import ModelConfig
    from model.network import PolicyValueNet


@unittest.skipIf(torch is None, 'install the neural extra')
class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'model.pt'
        self.config = ModelConfig(channels=8, blocks=1)
        self.model = PolicyValueNet(self.config)
        save_checkpoint(self.path, self.model)

    def test_eval_roundtrip(self):
        torch.manual_seed(42)
        self.model.eval()
        x = torch.randn(2, 6, 15, 15)
        other = load_checkpoint(self.path, self.config)
        self.assertFalse(other.training)
        with torch.inference_mode():
            for a, b in zip(self.model(x), other(x)):
                self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=0))
        data = torch.load(self.path, weights_only=True)
        self.assertEqual(data['torch_version'], str(torch.__version__))
        self.assertIn('git_commit', data)
        self.assertIn('git_dirty', data)
        self.assertTrue(data['git_dirty'] is None or type(data['git_dirty']) is bool)

        # Git for Windows emits non-ASCII worktree paths as UTF-8. Keep that
        # decoding explicit so a Korean path does not fall back to cp949.
        source = Path(checkpoint_module.__file__).resolve()
        root = source.parents[2]
        with patch.object(
            checkpoint_module.subprocess,
            'check_output',
            side_effect=[str(root) + '\n', 'deadbeef\n', ' M tracked.py\n'],
        ) as mocked:
            self.assertEqual(_git_provenance(), ('deadbeef', True))
        self.assertEqual(len(mocked.call_args_list), 3)
        for git_call in mocked.call_args_list:
            self.assertEqual(git_call.kwargs['encoding'], 'utf-8')
            self.assertEqual(git_call.kwargs['errors'], 'strict')

    def test_metadata_mismatches(self):
        original = torch.load(self.path, weights_only=True)
        for key in ('checkpoint_format_version', 'encoder_version', 'action_index_version',
                    'input_plane_names', 'torch_version', 'git_commit', 'git_dirty'):
            data = deepcopy(original)
            data[key] = -99
            torch.save(data, self.path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_checkpoint(self.path, self.config)
        for key in original['model_config']:
            data = deepcopy(original)
            data['model_config'][key] += 1
            torch.save(data, self.path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_checkpoint(self.path, self.config)

    def test_missing_and_extra_weights_rejected(self):
        original = torch.load(self.path, weights_only=True)
        for extra in (False, True):
            data = deepcopy(original)
            if extra:
                data['model_state']['unexpected'] = torch.zeros(1)
            else:
                data['model_state'].pop(next(iter(data['model_state'])))
            torch.save(data, self.path)
            with self.assertRaises(ValueError):
                load_checkpoint(self.path, self.config)

    def test_expected_config_required(self):
        with self.assertRaises(ValueError):
            load_checkpoint(self.path)
