"""Stage 8-C: device transfer for training and the accelerator smoke gate (CPU dry run)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.config import load_config
    from training.loop import run_training
    from training.replay_buffer import Batch
    from training.trainer import batch_to_device


@unittest.skipIf(torch is None, 'requires torch')
class BatchToDeviceTest(unittest.TestCase):
    def _batch(self):
        return Batch(torch.zeros(2, 6, 15, 15), torch.zeros(2, 225), torch.zeros(2, 1),
                     torch.ones(2, 225, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.int64))

    def test_same_device_is_a_no_op(self):
        batch = self._batch()
        self.assertIs(batch_to_device(batch, 'cpu'), batch)
        self.assertIs(batch_to_device(batch, torch.device('cpu')), batch)

    def test_other_device_moves_loss_tensors_only(self):
        batch = self._batch()
        moved = batch_to_device(batch, 'meta')   # a real second device available everywhere
        for name in ('states', 'policies', 'values', 'legal_masks'):
            self.assertEqual(getattr(moved, name).device.type, 'meta', name)
        self.assertIs(moved.provenance, batch.provenance)


@unittest.skipIf(torch is None, 'requires torch')
class SmokeGateDryRunTest(unittest.TestCase):
    def test_cpu_dry_run_passes_every_check(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from check_stage8_gpu import run_checks

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training']['generations'] = 1
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_training(config, run_dir=Path(tmp) / 'run', log=lambda m: None)
                checkpoint = Path(tmp) / 'run' / 'checkpoints' / 'latest.pt'
                report = run_checks(checkpoint, 'cpu', steps=3, positions=16, batch_sizes=[1, 2],
                                    repeats=1, log=lambda m: None)
        finally:
            torch.set_num_threads(threads)
        self.assertTrue(report['ok'], report['checks'])
        checks = report['checks']
        self.assertEqual(checks['inference_parity']['b1_vs_cpu'], {'prior': 0.0, 'value': 0.0})
        self.assertTrue(checks['training']['device']['finite'])
        self.assertEqual(checks['training']['first_loss_diff_vs_cpu'], 0.0)
        self.assertTrue(checks['checkpoint_roundtrip']['ok'])
        self.assertTrue(checks['deterministic_algorithms']['ok'])
        self.assertEqual([r['batch'] for r in checks['batch_benchmark']['rows']['device']], [1, 2])


if __name__ == '__main__':
    unittest.main()
