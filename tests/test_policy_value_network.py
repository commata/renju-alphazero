import unittest
from pathlib import Path
import importlib.util

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.network import PolicyValueNet
    from model.masking import policy_loss, value_loss


@unittest.skipIf(torch is None, 'requires torch')
class NetworkTest(unittest.TestCase):
    def test_forward_backward_optimizer(self):
        torch.manual_seed(42)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        model = PolicyValueNet()
        x = torch.randn(3, 6, 15, 15)
        logits, value = model(x)
        self.assertEqual(tuple(logits.shape), (3, 225))
        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((value.abs() <= 1).all())
        self.assertFalse(torch.allclose(logits.sum(-1), torch.ones(3)))
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, :20] = False
        target = mask.float() / mask.sum(-1, keepdim=True)
        pl = policy_loss(logits, target, mask)
        vl = value_loss(value, torch.tensor([[-1.], [0.], [1.]]))
        total = pl + vl
        for loss in (pl, vl, total):
            self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.trunk[0].weight.detach().clone()
        total.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.trunk[0].weight))

    def test_bad_input_shape(self):
        model = PolicyValueNet()
        for shape in ((6, 15, 15), (1, 5, 15, 15), (0, 6, 15, 15)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_tiny_dataset_reproducible_and_legal(self):
        path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_tiny_overfit.py'
        spec = importlib.util.spec_from_file_location('tiny_overfit', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        first = module.make_dataset(42, 32)
        second = module.make_dataset(42, 32)
        for a, b in zip(first[:4], second[:4]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(first[4], second[4])
        x, mask, policy, value, _ = first
        self.assertTrue(mask.any(-1).all())
        self.assertTrue((policy[~mask] == 0).all())
        self.assertTrue((policy.sum(-1) == 1).all())
        self.assertTrue(torch.equal(x[:, 5].flatten(1).bool(), mask))
        self.assertEqual(set(value.flatten().tolist()), {-1., 0., 1.})
