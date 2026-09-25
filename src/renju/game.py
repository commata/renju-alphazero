"""Mutable game state with validated moves and reversible history."""
from __future__ import annotations

from .rules import BLACK, WHITE, EMPTY, SIZE, DIRECTIONS, forbidden_reason, inside, run_length


class IllegalMove(ValueError):
    pass


OPENING_MOVE = (SIZE // 2, SIZE // 2)


class Game:
    def __init__(self):
        self.board = [[EMPTY] * SIZE for _ in range(SIZE)]
        self.to_play = BLACK
        self.winner: int | None = None
        self.done = False
        self.history: list[tuple[int, int]] = []

    def _is_initial_position(self) -> bool:
        return not self.history and all(cell == EMPTY for row in self.board for cell in row)

    def legal_moves(self) -> list[tuple[int, int]]:
        if self.done:
            return []
        if self._is_initial_position():
            return [OPENING_MOVE]
        return [(r, c) for r in range(SIZE) for c in range(SIZE)
                if self.board[r][c] == EMPTY and
                (self.to_play == WHITE or forbidden_reason(self.board, r, c) is None)]

    def has_legal_move(self) -> bool:
        """Return as soon as one legal move exists without building the full list."""
        if self.done:
            return False
        if self._is_initial_position():
            return True
        if self.to_play == WHITE:
            return any(EMPTY in row for row in self.board)
        for row in range(SIZE):
            for col in range(SIZE):
                if self.board[row][col] == EMPTY and forbidden_reason(self.board, row, col) is None:
                    return True
        return False

    def play(self, row: int, col: int) -> None:
        if self.done:
            raise IllegalMove("이미 종료된 대국입니다")
        if self._is_initial_position() and (row, col) != OPENING_MOVE:
            raise IllegalMove("첫 수는 흑이 바둑판 정중앙에 두어야 합니다")
        if not inside(row, col) or self.board[row][col] != EMPTY:
            raise IllegalMove("범위를 벗어났거나 이미 돌이 있습니다")
        if self.to_play == BLACK:
            reason = forbidden_reason(self.board, row, col)
            if reason:
                raise IllegalMove(f"흑 금수: {reason}")
        player = self.to_play
        self.board[row][col] = player
        self.history.append((row, col))
        lengths = (run_length(self.board, row, col, dr, dc) for dr, dc in DIRECTIONS)
        if any(n == 5 if player == BLACK else n >= 5 for n in lengths):
            self.winner = player
            self.done = True
        else:
            self.to_play = -player
            if not self.has_legal_move():
                self.done = True

    def undo(self) -> tuple[int, int]:
        if not self.history:
            raise IllegalMove("되돌릴 착수가 없습니다")
        row, col = self.history.pop()
        self.to_play = self.board[row][col]
        self.board[row][col] = EMPTY
        self.done = False
        self.winner = None
        return row, col
