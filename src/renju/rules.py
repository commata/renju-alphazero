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
_NEIGHBORS4 = {
    key: coords[max(0, index - 4):index] + coords[index + 1:index + 5]
    for key, (coords, index) in _LINE_INFO.items()
}
# Split of _NEIGHBORS4 into the cells within +/-3 and the (at most two) cells at +/-4.
_NEAR3 = {
    key: coords[max(0, index - 3):index] + coords[index + 1:index + 4]
    for key, (coords, index) in _LINE_INFO.items()
}
_EDGE4 = {
    key: tuple(coords[k] for k in (index - 4, index + 4) if 0 <= k < len(coords))
    for key, (coords, index) in _LINE_INFO.items()
}


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
    # A four-cell group containing move and one new extension needs two
    # existing black stones, both within three cells of move. This necessary
    # condition only rejects impossible directions; recursive legality follows.
    near = coords[max(0, index - 3):index] + coords[index + 1:index + 4]
    if sum(board[r][c] == BLACK for r, c in near) < 2:
        return False
    for pos in coords[max(0, index - 4):index + 5]:
        r, c = pos
        if board[r][c] != EMPTY:
            continue

        board[r][c] = BLACK
        try:
            # A Renju three must grow into a straight four without the extension
            # already making five in any direction.
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


def _axis_counts(board: list[list[int]], row: int, col: int) -> tuple[tuple[int, int], ...]:
    """Per axis, other black stones within +/-4 and within +/-3 of (row, col).

    Blockers are ignored, so these only overcount. They give necessary conditions
    taken from the exact classifiers below: five/overline through the point needs
    four others within +/-4, a four (``_fours``) needs three others in one five-cell
    window, and ``_open_three`` itself returns False with fewer than two within +/-3.
    """
    result = []
    for dr, dc in DIRECTIONS:
        key = (row, col, dr, dc)
        near = 0
        for r, c in _NEAR3[key]:
            if board[r][c] == BLACK:
                near += 1
        wide = near
        for r, c in _EDGE4[key]:
            if board[r][c] == BLACK:
                wide += 1
        result.append((wide, near))
    return tuple(result)


def _classify_black_stone(board: list[list[int]], row: int, col: int,
                          counts: tuple[tuple[int, int], ...]) -> str | None:
    """Exact classification of a placed black stone; axes skipped only when impossible.

    Same result as evaluating every axis: run_length only where five/overline is
    possible, ``_fours`` only where a four is possible, and ``_open_three`` only on
    candidate axes and only while two threes are still reachable.
    """
    lengths = [run_length(board, row, col, dr, dc)
               for (dr, dc), (wide, _) in zip(DIRECTIONS, counts) if wide >= 4]
    # Under RIF 9.2/9.3, a black move that simultaneously makes an exact five
    # wins; forbidden patterns matter only when the move does not make five.
    if 5 in lengths:
        return None
    if any(length >= 6 for length in lengths):
        return "장목"

    fours = 0
    for (dr, dc), (wide, _) in zip(DIRECTIONS, counts):
        if wide >= 3:
            fours += len(_fours(board, (row, col), dr, dc))
            if fours >= 2:
                return "사사"

    candidates = [direction for direction, (_, near) in zip(DIRECTIONS, counts) if near >= 2]
    threes = 0
    for index, (dr, dc) in enumerate(candidates):
        if threes + len(candidates) - index < 2:
            return None
        if _open_three(board, (row, col), dr, dc):
            threes += 1
            if threes >= 2:
                return "삼삼"
    return None


def _forbidden_after_black_move(board: list[list[int]], row: int, col: int) -> str | None:
    """Classify a black stone that is already present at (row, col)."""
    return _classify_black_stone(board, row, col, _axis_counts(board, row, col))


def _quiet_counts(counts: tuple[tuple[int, int], ...]) -> bool:
    """True proves the point is not forbidden; False is inconclusive.

    No axis with three others within +/-4 excludes five, overline and every four;
    at most one axis with two others within +/-3 leaves at most one open three.
    """
    two_stone_axes = 0
    for wide, near in counts:
        if wide > 2:
            return False
        if near == 2:
            two_stone_axes += 1
            if two_stone_axes > 1:
                return False
    return True


def _quiet_black_point(board: list[list[int]], row: int, col: int) -> bool:
    """Prove safety, or defer to the exact classifier (False is inconclusive)."""
    return _quiet_counts(_axis_counts(board, row, col))


def forbidden_reason(board: list[list[int]], row: int, col: int) -> str | None:
    """Classify a prospective black move. Board is restored before returning."""
    if not inside(row, col) or board[row][col] != EMPTY:
        raise ValueError("빈 보드의 범위 내 좌표가 필요합니다")
    counts = _axis_counts(board, row, col)
    if _quiet_counts(counts):
        return None
    board[row][col] = BLACK
    try:
        return _classify_black_stone(board, row, col, counts)
    finally:
        board[row][col] = EMPTY


def legal_black_points(board: list[list[int]]) -> list[tuple[int, int]]:
    """Row-major empty points that are not forbidden for black.

    Same list as filtering every empty point with ``forbidden_reason``. The per-axis
    counts of ``_axis_counts`` are accumulated once from the black stones (the
    +/-4 and +/-3 neighbourhoods are symmetric), so quiet points cost a table lookup
    and only the rest reach the exact classifier.
    """
    wide = [[0] * (SIZE * SIZE) for _ in DIRECTIONS]
    near = [[0] * (SIZE * SIZE) for _ in DIRECTIONS]
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] != BLACK:
                continue
            for axis, (dr, dc) in enumerate(DIRECTIONS):
                key = (row, col, dr, dc)
                axis_wide, axis_near = wide[axis], near[axis]
                for r, c in _NEAR3[key]:
                    axis_wide[r * SIZE + c] += 1
                    axis_near[r * SIZE + c] += 1
                for r, c in _EDGE4[key]:
                    axis_wide[r * SIZE + c] += 1
    result = []
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] != EMPTY:
                continue
            index = row * SIZE + col
            counts = tuple((wide[axis][index], near[axis][index]) for axis in range(len(DIRECTIONS)))
            if not _quiet_counts(counts):
                board[row][col] = BLACK
                try:
                    reason = _classify_black_stone(board, row, col, counts)
                finally:
                    board[row][col] = EMPTY
                if reason is not None:
                    continue
            result.append((row, col))
    return result
