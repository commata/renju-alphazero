"""Exact drop-in for ``search.mcts_v321._fast_winning_extensions_in_direction`` (docs/web-ser5.md §3.9).

The frozen function places a stone on each of the 8 cells within 4 of the anchor and scans a run
from each. Only two cells can extend the anchor's run: the first non-stone cell at each end of the
run of ``player`` stones through the anchor. A farther empty cell leaves that end cell empty
between itself and the anchor, so its run never contains the anchor. For each end cell that is
empty and at most 4 away, the run after placing there is the anchor's run, plus 1, plus the
stones beyond the end cell; black wins with exactly 5, white with 5 or more (the frozen
``_is_winning_run``). An anchor that is not ``player``'s stone gives no extension (the frozen
function skips step 0 and no other run can contain an empty or opponent anchor). Same set,
checked on every cell, direction and colour of random positions (tests).

``install()`` rebinds the name in every loaded module that imported the frozen function
(``search.mcts_v321``, ``search.mcts_v5``, ``search.mcts_v4``); this process only.
"""
from __future__ import annotations

import sys

from renju import BLACK, EMPTY, SIZE
from search import mcts_v321

FROZEN = mcts_v321._fast_winning_extensions_in_direction


def winning_extensions_in_direction(board, player: int, anchor, dr: int, dc: int) -> set:
    row, col = anchor
    if board[row][col] != player:
        return set()
    back = 0
    br, bc = row - dr, col - dc
    while 0 <= br < SIZE and 0 <= bc < SIZE and board[br][bc] == player:
        back += 1
        br -= dr
        bc -= dc
    ahead = 0
    ar, ac = row + dr, col + dc
    while 0 <= ar < SIZE and 0 <= ac < SIZE and board[ar][ac] == player:
        ahead += 1
        ar += dr
        ac += dc
    result = set()
    for er, ec, offset, sr, sc in ((br, bc, back + 1, -dr, -dc), (ar, ac, ahead + 1, dr, dc)):
        if offset > 4 or not (0 <= er < SIZE and 0 <= ec < SIZE) or board[er][ec] != EMPTY:
            continue
        beyond = 0
        xr, xc = er + sr, ec + sc
        while 0 <= xr < SIZE and 0 <= xc < SIZE and board[xr][xc] == player:
            beyond += 1
            xr += sr
            xc += sc
        length = back + ahead + 2 + beyond
        if (length == 5) if player == BLACK else (length >= 5):
            result.add((er, ec))
    return result


def _bindings(target):
    return [(module, name) for module in list(sys.modules.values())
            if module is not None and getattr(module, '__name__', None) != __name__
            for name, value in list(getattr(module, '__dict__', {}).items()) if value is target]


def installed() -> bool:
    return mcts_v321._fast_winning_extensions_in_direction is not FROZEN


def install() -> int:
    bound = _bindings(FROZEN)
    for module, name in bound:
        setattr(module, name, winning_extensions_in_direction)
    return len(bound)


def uninstall() -> int:
    bound = _bindings(winning_extensions_in_direction)
    for module, name in bound:
        setattr(module, name, FROZEN)
    return len(bound)
