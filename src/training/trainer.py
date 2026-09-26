"""Optimizer construction, one training step, and the train/eval mode contract.

Losses reuse the Stage 4 helpers ``policy_loss`` (masked cross entropy with illegal
log-probs zeroed before multiplication, so no ``0 * -inf``) and ``value_loss`` (MSE on
``[B,1]``). ``optimizer.weight_decay`` is an optimizer setting, not a loss term; the
explicit ``l2_coeff * ||theta||^2`` loss term is separate and never combined with it.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite

import torch
from torch import nn

from model.masking import policy_loss, value_loss

from .replay_buffer import Batch


class TrainingDivergenceError(RuntimeError):
    """Non-finite model output, loss or gradient norm; raised before optimizer.step()."""


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    o = config['optimizer']
    if o['weight_decay'] and config['loss']['l2_coeff']:
        raise ValueError('use either optimizer.weight_decay or loss.l2_coeff, not both')
    if o['name'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=o['lr'], weight_decay=o['weight_decay'])
    if o['name'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=o['lr'], momentum=o['momentum'],
                               weight_decay=o['weight_decay'])
    raise ValueError(f"unsupported optimizer: {o['name']}")


def compute_losses(logits: torch.Tensor, values: torch.Tensor, batch: Batch,
                   value_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy = policy_loss(logits, batch.policies, batch.legal_masks)
    value = value_loss(values, batch.values)
    return policy, value, policy + value_weight * value


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: Batch,
               value_weight: float, grad_clip: float | None = None,
               l2_coeff: float = 0.0) -> dict:
    model.train()
    logits, values = model(batch.states)
    if not (torch.isfinite(logits).all() and torch.isfinite(values).all()):
        raise TrainingDivergenceError('non-finite model output during training')
    policy, value, total = compute_losses(logits, values, batch, value_weight)
    if l2_coeff:
        total = total + l2_coeff * sum(p.pow(2).sum() for p in model.parameters())
    if not torch.isfinite(total):
        raise TrainingDivergenceError(f'non-finite loss: {total.item()}')
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip if grad_clip else float('inf')))
    if not isfinite(grad_norm):
        raise TrainingDivergenceError(f'non-finite gradient norm: {grad_norm}')
    optimizer.step()
    return {'policy_loss': policy.item(), 'value_loss': value.item(),
            'total_loss': total.item(), 'grad_norm': grad_norm,
            'lr': float(optimizer.param_groups[0]['lr'])}


@contextmanager
def inference_mode_for(model: nn.Module) -> Iterator[nn.Module]:
    """Self-play / evaluation: eval mode + no_grad, restoring the previous mode after.

    ``PolicyValueEvaluator`` also calls ``model.eval()`` on construction, so the restore
    here is what keeps a shared training model from silently staying in eval mode.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield model
    finally:
        model.train(was_training)
