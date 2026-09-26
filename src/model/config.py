"""Frozen encoder/action contracts and explicit network configuration."""
from dataclasses import dataclass

BOARD_SIZE = 15
ACTION_COUNT = BOARD_SIZE ** 2
INPUT_PLANE_NAMES = (
    "current_stones", "opponent_stones", "last_move", "current_is_black",
    "board_constant", "legal_actions",
)
ENCODER_VERSION = "renju-relative-6p-v1"
ACTION_INDEX_VERSION = "row-major-15x15-v1"
CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class ModelConfig:
    board_size: int = BOARD_SIZE
    input_planes: int = len(INPUT_PLANE_NAMES)
    channels: int = 64
    blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden: int = 64

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.board_size != BOARD_SIZE or self.input_planes != len(INPUT_PLANE_NAMES):
            raise ValueError("The v1 encoder requires a 15x15 board and 6 planes")


def coordinate_to_action(row: int, col: int) -> int:
    if any(type(x) is not int or not 0 <= x < BOARD_SIZE for x in (row, col)):
        raise ValueError("row and col must be integers in [0, 14]")
    return row * BOARD_SIZE + col


def action_to_coordinate(action: int) -> tuple[int, int]:
    if type(action) is not int or not 0 <= action < ACTION_COUNT:
        raise ValueError("action must be an integer in [0, 224]")
    return divmod(action, BOARD_SIZE)
