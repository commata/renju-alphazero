import random
import unittest
from unittest.mock import patch

from renju import Game
from training.config import ConfigError, resolve_config

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.config import ModelConfig, coordinate_to_action
    from model.encoding import encode_game
    from model.evaluator import PolicyValueEvaluator
    from model.masking import legal_moves_to_mask
    from model.network import PolicyValueNet
    from training.replay_buffer import Batch
    from training.trainer import (TrainingDivergenceError, build_optimizer, compute_losses,
                                  inference_mode_for, train_step)

TINY = dict(channels=8, blocks=1, policy_channels=2, value_channels=1, value_hidden=8)


def make_batch(size=8, seed=0):
    rng = random.Random(seed)
    states, policies, values, masks = [], [], [], []
    for index in range(size):
        game = Game()
        for _ in range(3 + index % 7):
            game.play(*rng.choice(game.legal_moves()))
        moves = game.legal_moves()
        mask = legal_moves_to_mask(moves)
        policy = torch.zeros(225)
        policy[coordinate_to_action(*rng.choice(moves))] = 1
        states.append(encode_game(game, mask))
        policies.append(policy)
        values.append([float(index % 3 - 1)])
        masks.append(mask)
    return Batch(torch.stack(states), torch.stack(policies), torch.tensor(values),
                 torch.stack(masks), torch.zeros(size, 3, dtype=torch.int64))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(ModelConfig(**TINY))


@unittest.skipIf(torch is None, 'requires torch')
class TrainStepTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = resolve_config({'model': TINY})
        self.model = tiny_model()
        self.optimizer = build_optimizer(self.model, self.config)
        self.batch = make_batch()

    def test_finite_metrics_and_weights_change(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        metrics = train_step(self.model, self.optimizer, self.batch, 1.0)
        for key in ('policy_loss', 'value_loss', 'total_loss', 'grad_norm', 'lr'):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])), key)
        self.assertEqual(metrics['lr'], 0.001)
        self.assertAlmostEqual(metrics['total_loss'],
                               metrics['policy_loss'] + metrics['value_loss'], places=5)
        changed = any(not torch.equal(a, b) for a, b in zip(before, self.model.parameters()))
        self.assertTrue(changed)
        self.assertTrue(self.model.training)

    def test_tiny_fixed_dataset_overfits(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        first = train_step(self.model, optimizer, self.batch, 1.0)['total_loss']
        for _ in range(60):
            last = train_step(self.model, optimizer, self.batch, 1.0)['total_loss']
        self.assertLess(last, first * 0.5)

    def test_illegal_logits_do_not_affect_loss(self):
        logits = torch.randn(8, 225)
        values = torch.zeros(8, 1)
        base = compute_losses(logits, values, self.batch, 1.0)[2]
        changed = logits.clone()
        changed[~self.batch.legal_masks] = 1e4 * torch.randn(int((~self.batch.legal_masks).sum()))
        self.assertTrue(torch.equal(base, compute_losses(changed, values, self.batch, 1.0)[2]))
        # Gradient never flows into illegal logits.
        logits.requires_grad_(True)
        compute_losses(logits, values, self.batch, 1.0)[2].backward()
        self.assertEqual(float(logits.grad[~self.batch.legal_masks].abs().sum()), 0.0)

    def test_value_weight_scales_value_loss(self):
        metrics = train_step(self.model, self.optimizer, self.batch, 0.5)
        self.assertAlmostEqual(metrics['total_loss'],
                               metrics['policy_loss'] + 0.5 * metrics['value_loss'], places=5)

    def test_l2_coeff_adds_parameter_norm(self):
        params = sum(float(p.detach().pow(2).sum()) for p in self.model.parameters())
        metrics = train_step(self.model, torch.optim.SGD(self.model.parameters(), lr=0.0),
                             self.batch, 1.0, l2_coeff=0.01)
        self.assertAlmostEqual(metrics['total_loss'] - metrics['policy_loss']
                               - metrics['value_loss'], 0.01 * params, places=3)

    def test_nan_weight_raises_before_step(self):
        with torch.no_grad():
            next(self.model.parameters()).view(-1)[0] = float('nan')
        before = [p.detach().clone() for p in self.model.parameters()]
        with patch.object(self.optimizer, 'step') as step:
            with self.assertRaises(TrainingDivergenceError):
                train_step(self.model, self.optimizer, self.batch, 1.0)
            step.assert_not_called()
        for a, b in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(a.nan_to_num(), b.detach().nan_to_num()))

    def test_nonfinite_grad_norm_raises_before_step(self):
        with patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(float('inf'))):
            with patch.object(self.optimizer, 'step') as step:
                with self.assertRaises(TrainingDivergenceError):
                    train_step(self.model, self.optimizer, self.batch, 1.0)
                step.assert_not_called()

    def test_grad_clip(self):
        metrics = train_step(self.model, self.optimizer, self.batch, 1.0, grad_clip=1e-6)
        self.assertGreater(metrics['grad_norm'], 1e-6)  # pre-clip norm is reported
        clipped = sum(float(p.grad.pow(2).sum()) for p in self.model.parameters()) ** 0.5
        self.assertLessEqual(clipped, 1.01e-6)

    def test_weight_decay_and_l2_together_is_config_error(self):
        with self.assertRaises(ConfigError):
            resolve_config({'optimizer': {'weight_decay': 1e-4}, 'loss': {'l2_coeff': 1e-4}})
        bad = resolve_config({'model': TINY})
        bad['loss']['l2_coeff'] = 1e-4  # bypass config validation
        with self.assertRaises(ValueError):
            build_optimizer(self.model, bad)

    def test_optimizer_matches_config(self):
        self.assertIsInstance(self.optimizer, torch.optim.Adam)
        self.assertEqual(self.optimizer.param_groups[0]['weight_decay'], 0.0001)
        sgd = build_optimizer(self.model, resolve_config(
            {'optimizer': {'name': 'sgd', 'lr': 0.1, 'momentum': 0.9}}))
        self.assertIsInstance(sgd, torch.optim.SGD)
        self.assertEqual(sgd.param_groups[0]['momentum'], 0.9)


@unittest.skipIf(torch is None, 'requires torch')
class ModeContractTest(unittest.TestCase):
    def test_inference_mode_for_restores_training_mode(self):
        model = tiny_model()
        model.train()
        with inference_mode_for(model):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            PolicyValueEvaluator(model)  # sets eval() on the shared model
        self.assertTrue(model.training)
        self.assertTrue(torch.is_grad_enabled())
        model.eval()
        with inference_mode_for(model):
            pass
        self.assertFalse(model.training)

    def test_inference_mode_for_restores_on_error(self):
        model = tiny_model()
        model.train()
        with self.assertRaises(KeyError):
            with inference_mode_for(model):
                raise KeyError
        self.assertTrue(model.training)

    def test_batchnorm_statistics_frozen_in_eval_context(self):
        model = tiny_model()
        model.train()
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        with inference_mode_for(model):
            model(make_batch(4).states)
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(buffers[key], value), key)
        model(make_batch(4).states)  # train mode updates BN running stats
        self.assertTrue(any(not torch.equal(buffers[k], v) for k, v in model.named_buffers()))


if __name__ == '__main__':
    unittest.main()
