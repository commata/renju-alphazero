"""Renju move classification on a 15x15 board. Coordinates are (row, column)."""
from __future__ import annotations

SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, -1
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def inside(row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE


def _build_line_info() -> dict[tuple[int, int, int, int], tuple[tuple[tuple[int, int], ...], int]]:
    """Precompute immutable board geometry for every cell and direction."""
    result = {}
    for row in range(SIZE):
        for col in range(SIZE):
            for dr, dc in DIRECTIONS:
                r, c = row, col
                while inside(r - dr, c - dc):
                    r -= dr
                    c -= dc
                coords = []
                while inside(r, c):
                    coords.append((r, c))
                    r += dr
                    c += dc
                line = tuple(coords)
                result[(row, col, dr, dc)] = (line, line.index((row, col)))
    return result


_LINE_INFO = _build_line_info()


def run_length(board: list[list[int]], row: int, col: int, dr: int, dc: int) -> int:
    color = board[row][col]
    count = 1
    for sign in (-1, 1):
        r, c = row + sign * dr, col + sign * dc
        while inside(r, c) and board[r][c] == color:
            count += 1
            r += sign * dr
            c += sign * dc
    return count


def _fours(board: list[list[int]], move: tuple[int, int], dr: int, dc: int):
    """Four stone sets containing move, with at least one exact-five completion."""
    coords, move_index = _LINE_INFO[(*move, dr, dc)]
    result = set()
    start_min = max(0, move_index - 4)
    start_max = min(move_index, len(coords) - 5)
    for start in range(start_min, start_max + 1):
        window = coords[start:start + 5]
        black = tuple(p for p in window if board[p[0]][p[1]] == BLACK)
        blanks = [p for p in window if board[p[0]][p[1]] == EMPTY]
        if len(black) == 4 and len(blanks) == 1:
            r, c = blanks[0]
            board[r][c] = BLACK
            exact = run_length(board, r, c, dr, dc) == 5
            board[r][c] = EMPTY
            if exact:
                result.add(frozenset(black))
    return result


def _straight_four_after_extension(
    board: list[list[int]],
    move: tuple[int, int],
    extension: tuple[int, int],
    dr: int,
    dc: int,
) -> bool:
    """Whether an already placed extension makes a straight four containing move."""
    coords, move_index = _LINE_INFO[(*move, dr, dc)]
    _, extension_index = _LINE_INFO[(*extension, dr, dc)]
    later = max(move_index, extension_index)
    earlier = min(move_index, extension_index)
    start_min = max(0, later - 3)
    start_max = min(earlier, len(coords) - 4)

    for start in range(start_min, start_max + 1):
        group = coords[start:start + 4]
        if not all(board[row][col] == BLACK for row, col in group):
            continue
        before = (group[0][0] - dr, group[0][1] - dc)
        after = (group[-1][0] + dr, group[-1][1] + dc)
        if not inside(*before) or not inside(*after):
            continue
        if board[before[0]][before[1]] != EMPTY or board[after[0]][after[1]] != EMPTY:
            continue

        board[before[0]][before[1]] = BLACK
        first = run_length(board, before[0], before[1], dr, dc) == 5
        board[before[0]][before[1]] = EMPTY

        board[after[0]][after[1]] = BLACK
        second = run_length(board, after[0], after[1], dr, dc) == 5
        board[after[0]][after[1]] = EMPTY
        if first and second:
            return True
    return False


def _open_three(board: list[list[int]], move: tuple[int, int], dr: int, dc: int) -> bool:
    """Check a three, recursively requiring its straight-four extension to be legal."""
    coords, index = _LINE_INFO[(*move, dr, dc)]
    for pos in coords[max(0, index - 4):index + 5]:
        r, c = pos
        if board[r][c] != EMPTY:
            continue

        board[r][c] = BLACK
        try:
            if any(run_length(board, r, c, vr, vc) == 5 for vr, vc in DIRECTIONS):
                continue
            # Recurse only after the candidate really makes the required straight four.
            # Keeping ancestor stones on the board makes each recursive step progress.
            if not _straight_four_after_extension(board, move, pos, dr, dc):
                continue
            if _forbidden_after_black_move(board, r, c) is None:
                return True
        finally:
            board[r][c] = EMPTY
    return False


def _forbidden_after_black_move(board: list[list[int]], row: int, col: int) -> str | None:
    """Classify a black stone that is already present at (row, col)."""
    lengths = [run_length(board, row, col, dr, dc) for dr, dc in DIRECTIONS]
    if 5 in lengths:
        return None
    if any(length >= 6 for length in lengths):
        return "장목"

    fours = 0
    for dr, dc in DIRECTIONS:
        fours += len(_fours(board, (row, col), dr, dc))
        if fours >= 2:
            return "사사"

    threes = 0
    for dr, dc in DIRECTIONS:
        if _open_three(board, (row, col), dr, dc):
            threes += 1
            if threes >= 2:
                return "삼삼"
    return None


def forbidden_reason(board: list[list[int]], row: int, col: int) -> str | None:
    """Classify a prospective black move. Board is restored before returning."""
    if not inside(row, col) or board[row][col] != EMPTY:
        raise ValueError("빈 보드의 범위 내 좌표가 필요합니다")
    board[row][col] = BLACK
    try:
        return _forbidden_after_black_move(board, row, col)
    finally:
        board[row][col] = EMPTY
