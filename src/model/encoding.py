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


def encode_batch(boards, to_play, last_moves, legal_masks: torch.Tensor) -> torch.Tensor:
    """``encode_game`` for B states at once (CPU tensors; Stage 8 batched inference).

    ``boards``: B nested 15x15 lists, ``to_play``: B players, ``last_moves``: B
    ``(row, col)`` or ``None``, ``legal_masks``: bool [B, 225]. Equal, plane for plane,
    to stacking ``encode_game`` of each state.
    """
    count = len(boards)
    validate_legal_mask(legal_masks, (count, ACTION_COUNT))
    board = torch.tensor(boards, dtype=torch.int64, device=legal_masks.device)
    if tuple(board.shape) != (count, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("boards must be Bx15x15")
    player = torch.tensor(to_play, dtype=torch.int64, device=legal_masks.device)
    if not bool(((player == BLACK) | (player == -BLACK)).all()):
        raise ValueError("invalid player")
    side = player.view(count, 1, 1)
    planes = torch.zeros((count, len(INPUT_PLANE_NAMES), BOARD_SIZE, BOARD_SIZE),
                         dtype=torch.float32, device=legal_masks.device)
    planes[:, 0] = board == side
    planes[:, 1] = board == -side
    for index, move in enumerate(last_moves):
        if move is not None:
            planes[index, 2, move[0], move[1]] = 1
    planes[:, 3] = (side == BLACK).float().expand(count, BOARD_SIZE, BOARD_SIZE)
    planes[:, 4] = 1
    planes[:, 5] = legal_masks.reshape(count, BOARD_SIZE, BOARD_SIZE)
    return planes
