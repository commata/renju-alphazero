"""Canonical boolean action masks; legality is owned exclusively by Game."""
from collections.abc import Iterable

import torch
from torch.nn import functional as F

from .config import ACTION_COUNT, coordinate_to_action


def legal_moves_to_mask(moves: Iterable[tuple[int, int]], *, device=None) -> torch.Tensor:
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool, device=device)
    indices = [coordinate_to_action(row, col) for row, col in moves]
    mask[indices] = True
    return mask


def validate_legal_mask(mask: torch.Tensor, shape: tuple[int, ...]) -> None:
    if mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"legal mask must be bool with shape {shape}")


def mask_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim not in (1, 2) or logits.shape[-1] != ACTION_COUNT or logits.numel() == 0:
        raise ValueError("logits must have shape [225] or [B,225], B > 0")
    validate_legal_mask(legal_mask, tuple(logits.shape))
    if logits.device != legal_mask.device:
        raise ValueError("logits and mask must use the same device")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating point values")
    if not legal_mask.any(dim=-1).all():
        raise ValueError("each state must have at least one legal action")
    return logits.masked_fill(~legal_mask, -torch.inf)


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    return F.softmax(mask_policy_logits(logits, legal_mask), dim=-1)


def policy_loss(logits: torch.Tensor, target_policy: torch.Tensor,
                legal_mask: torch.Tensor) -> torch.Tensor:
    masked = mask_policy_logits(logits, legal_mask)
    if logits.ndim != 2 or target_policy.shape != logits.shape:
        raise ValueError("policy targets and logits must have shape [B,225]")
    if target_policy.device != logits.device or not target_policy.is_floating_point():
        raise ValueError("policy targets must be floating point on the logits device")
    if not torch.isfinite(target_policy).all() or (target_policy < 0).any():
        raise ValueError("policy targets must be finite and non-negative")
    if (target_policy.masked_select(~legal_mask) != 0).any():
        raise ValueError("illegal action target mass must be zero")
    if not torch.allclose(target_policy.sum(-1), torch.ones_like(target_policy.sum(-1)),
                          atol=1e-5, rtol=0):
        raise ValueError("policy target rows must sum to one")
    # Clear illegal -inf before multiplication: zero times -inf would be NaN.
    log_probs = F.log_softmax(masked, dim=-1).masked_fill(~legal_mask, 0)
    return -(target_policy * log_probs).sum(-1).mean()


def value_loss(predicted_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    if (predicted_value.ndim != 2 or predicted_value.shape[1] != 1
            or predicted_value.shape[0] == 0 or target_value.shape != predicted_value.shape):
        raise ValueError("values must have matching shape [B,1], B > 0")
    if predicted_value.device != target_value.device:
        raise ValueError("values must use the same device")
    for value in (predicted_value, target_value):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("values must be finite floating point")
    if (target_value.abs() > 1).any():
        raise ValueError("value targets must be in [-1,1]")
    return F.mse_loss(predicted_value, target_value)
