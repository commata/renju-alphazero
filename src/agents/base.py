"""Minimal interface shared by baseline and future search agents."""
from typing import Protocol

from renju import Game


class Agent(Protocol):
    name: str

    def select_move(self, game: Game) -> tuple[int, int]:
        """Return a legal move without changing game; raise if none exists."""
        ...
