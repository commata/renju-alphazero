"""Relative six-plane encoding; single state [6,15,15], stack for batches."""
import torch

from renju import BLACK, Game
from .config import ACTION_COUNT, BOARD_SIZE, INPUT_PLANE_NAMES
from .masking import legal_moves_to_mask, validate_legal_mask


def encode_game(game: Game, legal_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Reuse a caller-owned mask for this exact state without recomputing legality.

    Terminal boards are representable, but their empty mask cannot be softmaxed.
    Value perspective always follows game.to_play, including terminal boards.
    """
    if legal_mask is None:
        legal_mask = legal_moves_to_mask(game.legal_moves())
    validate_legal_mask(legal_mask, (ACTION_COUNT,))
    board = torch.tensor(game.board, device=legal_mask.device)
    if tuple(board.shape) != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("board must be 15x15")
    if game.to_play not in (BLACK, -BLACK):
        raise ValueError("invalid player")
    planes = torch.zeros((len(INPUT_PLANE_NAMES), BOARD_SIZE, BOARD_SIZE),
                         dtype=torch.float32, device=legal_mask.device)
    planes[0] = board == game.to_play
    planes[1] = board == -game.to_play
    if game.history:
        row, col = game.history[-1]
        planes[2, row, col] = 1
    planes[3].fill_(game.to_play == BLACK)
    planes[4].fill_(1)
    planes[5] = legal_mask.reshape(BOARD_SIZE, BOARD_SIZE)
    return planes
