"""Root-only compound threats; the existing engine owns all black legality."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations

from renju import BLACK, WHITE, EMPTY, Game
from renju.rules import DIRECTIONS, inside
from .mcts import Move, _is_legal_for_player, _wins_for_player
from .mcts_v5 import WINDOWS, _threat_windows


BY_CELL = {p: tuple(w for w in WINDOWS if p in w)
           for p in ((r, c) for r in range(15) for c in range(15))}


@contextmanager
def placed(game: Game, player: int, move: Move):
    """Geometry probes intentionally leave turn, terminal flags and history alone."""
    r, c = move
    if not inside(r, c) or game.board[r][c] != EMPTY:
        raise ValueError("temporary placement requires an empty board cell")
    game.board[r][c] = player
    try:
        yield
    finally:
        game.board[r][c] = EMPTY


@dataclass(frozen=True)
class Threat:
    axis: Move
    stones: frozenset[Move]
    continuations: frozenset[Move]
    defenses: frozenset[Move]


@dataclass(frozen=True)
class Compound:
    move: Move
    kinds: frozenset[str]
    fours: tuple[Threat, ...]
    threes: tuple[Threat, ...]

    @property
    def defense_points(self) -> frozenset[Move]:
        return frozenset({self.move}).union(*(t.defenses for t in self.fours + self.threes))


def _axis(window) -> Move:
    return window[1][0] - window[0][0], window[1][1] - window[0][1]


def structural_candidates(game: Game, player: int) -> list[Move]:
    axes: dict[Move, set[Move]] = {}
    for window in _threat_windows(game, player, 2):
        for p in window:
            if game.board[p[0]][p[1]] == EMPTY:
                axes.setdefault(p, set()).add(_axis(window))
    return sorted(p for p, directions in axes.items() if len(directions) >= 2)


def fours_at(game: Game, player: int, anchor: Move) -> tuple[Threat, ...]:
    groups: dict[tuple, set[Move]] = {}
    for window in BY_CELL[anchor]:
        stones = frozenset(p for p in window if game.board[p[0]][p[1]] == player)
        blanks = [p for p in window if game.board[p[0]][p[1]] == EMPTY]
        if len(stones) != 4 or len(blanks) != 1:
            continue
        p = blanks[0]
        if _wins_for_player(game, player, p) and _is_legal_for_player(game, player, p):
            groups.setdefault((_axis(window), stones), set()).add(p)
    return tuple(Threat(axis, stones, frozenset(points), frozenset(points))
                 for (axis, stones), points in groups.items())


def threes_at(game: Game, player: int, anchor: Move, *, exact=True) -> tuple[Threat, ...]:
    groups: dict[tuple, tuple[set, set]] = {}
    r, c = anchor
    for dr, dc in DIRECTIONS:
        for offset in range(-3, 1):
            group = tuple((r + (offset+i)*dr, c + (offset+i)*dc) for i in range(4))
            ends = ((group[0][0]-dr, group[0][1]-dc),
                    (group[-1][0]+dr, group[-1][1]+dc))
            if not all(inside(*p) for p in (*group, *ends)):
                continue
            if any(game.board[x][y] != EMPTY for x, y in ends):
                continue
            stones = frozenset(p for p in group if game.board[p[0]][p[1]] == player)
            blanks = [p for p in group if game.board[p[0]][p[1]] == EMPTY]
            if len(stones) != 3 or len(blanks) != 1 or anchor not in stones:
                continue
            extension = blanks[0]
            if exact and not _is_legal_for_player(game, player, extension):
                continue
            with placed(game, player, extension):
                if not all(_wins_for_player(game, player, p)
                           and _is_legal_for_player(game, player, p) for p in ends):
                    continue
            extensions, defenses = groups.setdefault(((dr, dc), stones), (set(), set()))
            extensions.add(extension)
            defenses.update((*ends, extension))
    return tuple(Threat(axis, stones, frozenset(ext), frozenset(defense))
                 for (axis, stones), (ext, defense) in groups.items())


def _independent(first: Threat, second: Threat) -> bool:
    # Conservative initial scope: separate axes; no double counting sliding windows.
    return first.axis != second.axis and not first.defenses.intersection(second.defenses)


def _valid_43(game: Game, player: int, anchor: Move, four: Threat, three: Threat) -> bool:
    if not _independent(four, three):
        return False
    # A four reply can also block a crossing line that made a three extension 44.
    # Validate the actual continuation AFTER that forced reply, not on a pass board.
    for block in sorted(four.continuations):
        if not _is_legal_for_player(game, -player, block):
            continue
        with placed(game, -player, block):
            if not any(t.stones == three.stones and t.axis == three.axis
                       for t in threes_at(game, player, anchor)):
                return False
    return True


def compound_at(game: Game, player: int, move: Move) -> Compound | None:
    if not inside(*move) or not _is_legal_for_player(game, player, move):
        return None
    if _wins_for_player(game, player, move):
        return None
    with placed(game, player, move):
        fours = fours_at(game, player, move)
        threes = threes_at(game, player, move, exact=not bool(fours))
        kinds = set()
        if any(_valid_43(game, player, move, f, t) for f in fours for t in threes):
            kinds.add('43')
        if player == WHITE:
            if any(_independent(a, b) for a, b in combinations(fours, 2)):
                kinds.add('44')
            if any(_independent(a, b) for a, b in combinations(threes, 2)):
                kinds.add('33')
        return Compound(move, frozenset(kinds), fours, threes) if kinds else None


def compound_moves(game: Game, player: int) -> dict[Move, Compound]:
    if game.done:
        return {}
    result = {}
    for move in structural_candidates(game, player):
        compound = compound_at(game, player, move)
        if compound is not None:
            result[move] = compound
    return result


def black_legal_43_moves(game: Game) -> list[Move]:
    return [m for m, t in compound_moves(game, BLACK).items() if '43' in t.kinds]


black_immediate_43_creators = black_legal_43_moves


def white_43_moves(game: Game) -> list[Move]:
    return [m for m, t in compound_moves(game, WHITE).items() if '43' in t.kinds]


def white_44_moves(game: Game) -> list[Move]:
    return [m for m, t in compound_moves(game, WHITE).items() if '44' in t.kinds]


def white_33_moves(game: Game) -> list[Move]:
    return [m for m, t in compound_moves(game, WHITE).items() if '33' in t.kinds]
