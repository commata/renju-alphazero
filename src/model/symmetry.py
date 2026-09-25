"""D4: ids 0..3 rotate CCW; ids 4..7 mirror columns then rotate CCW.

Row increases downwards. CCW maps (r,c) to (N-1-c,r); mirror maps
(r,c) to (r,N-1-c). All leading tensor dimensions are preserved.
"""
import torch

from .config import ACTION_COUNT, BOARD_SIZE, action_to_coordinate, coordinate_to_action

SYMMETRIES = tuple(range(8))


def _validate(symmetry: int) -> None:
    if type(symmetry) is not int or symmetry not in SYMMETRIES:
        raise ValueError('symmetry must be an integer in [0,7]')


def inverse_symmetry(symmetry: int) -> int:
    _validate(symmetry)
    return (-symmetry) % 4 if symmetry < 4 else symmetry


def transform_coordinate(row: int, col: int, symmetry: int) -> tuple[int, int]:
    _validate(symmetry)
    coordinate_to_action(row, col)
    if symmetry >= 4:
        col = BOARD_SIZE - 1 - col
    for _ in range(symmetry % 4):
        row, col = BOARD_SIZE - 1 - col, row
    return row, col


def transform_action(action: int, symmetry: int) -> int:
    return coordinate_to_action(*transform_coordinate(*action_to_coordinate(action), symmetry))


def transform_spatial(tensor: torch.Tensor, symmetry: int) -> torch.Tensor:
    """Transform boards or encoded planes with trailing shape [15,15]."""
    _validate(symmetry)
    if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError('spatial tensor must end in [15,15]')
    if symmetry >= 4:
        tensor = tensor.flip(-1)
    return torch.rot90(tensor, symmetry % 4, (-2, -1)).clone()


def transform_policy(policy: torch.Tensor, symmetry: int) -> torch.Tensor:
    """Permute single/batched policy targets (or logits), without normalization."""
    if policy.ndim not in (1, 2) or policy.shape[-1] != ACTION_COUNT:
        raise ValueError('policy must have shape [225] or [B,225]')
    spatial = policy.reshape(*policy.shape[:-1], BOARD_SIZE, BOARD_SIZE)
    return transform_spatial(spatial, symmetry).reshape(policy.shape)


def transform_mask(mask: torch.Tensor, symmetry: int) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise ValueError('mask must be bool')
    return transform_policy(mask, symmetry)
