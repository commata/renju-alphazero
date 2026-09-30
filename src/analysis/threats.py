"""Bounded threat proofs for offline analysis: VCF and depth-limited VCT.

Three outcomes, always from the point of view of the side that just moved
(the opponent is to move):

- ``UNSAFE``: the opponent has a proven forced win of the searched class.
- ``SAFE``: every searched line was exhausted and none wins for the opponent.
- ``UNKNOWN``: a budget or depth cut prevented a proof either way.

Searched class
--------------
``vct_depth = 0`` is VCF: the opponent wins by an immediate five or by the frozen
V7 VCF solver (``search.mcts_v7._find_vcf_with_stats``) with ``max_fours`` set
above what the empty board can hold, so its only cut is the node budget it
reports. That solver skips lines where the defender's forced block itself makes a
four, so SAFE means "no VCF of that class".

``vct_depth = d > 0`` also lets the opponent play up to ``d`` quiet moves (open
threes, broken threes, anything) before the VCF. Every legal quiet move ``h`` is
tried and every reply of ours is actually played, so forbidden-point effects are
handled by the engine for both colours: a black defender's own stone can turn its
needed block into a forbidden point, and a white defender's stone can remove a
black forbidden point. A move ``h`` is refuted as soon as one reply is SAFE at
depth ``d - 1`` (replies nearest the stones first), which keeps the cost close to
one VCF probe per quiet move.

``prune_quiet=True`` restores the older null-move pruning (skip ``h`` when the
opponent has no VCF after ``h`` if we passed). It is faster but NOT sound for SAFE
because of the forbidden-point effects above; use it only for rough analysis,
never for labels or probes.

Our own fours are handled everywhere: after our four the opponent must block
(unless the only block is a black forbidden point), and the position after the
block is a fresh decision for us, bounded by ``four_chain``.

UNSAFE results carry a witness: ``('five', move)``, ``('vcf', moves)``,
``('threat', move)`` (then every reply of ours is UNSAFE at depth ``d - 1``) or
``('block', move)`` (our four is blocked there and every reply loses).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from renju import EMPTY, SIZE, Game
from search.mcts_v5 import _winning_moves
from search.mcts_v7 import _find_vcf_with_stats

SAFE, UNKNOWN, UNSAFE = 'SAFE', 'UNKNOWN', 'UNSAFE'
VCF_WIN, VCF_NONE, VCF_UNKNOWN = 'WIN', 'NONE', 'UNKNOWN'

Move = tuple[int, int]


def decision_status(counts: Counter) -> str:
    """A decision is SAFE if any move is SAFE, UNSAFE only if every move is UNSAFE."""
    if counts[SAFE]:
        return SAFE
    return UNKNOWN if counts[UNKNOWN] else UNSAFE


def _board_key(game: Game) -> tuple:
    return tuple(tuple(row) for row in game.board)


def _unbounded_max_fours(game: Game) -> int:
    # Every four in a VCF line spends an attacker stone and a defender block.
    empties = sum(row.count(EMPTY) for row in game.board)
    return empties // 2 + 1


def ordered_moves(game: Game) -> list[Move]:
    """Legal moves, nearest to the stones (and the last move) first.

    Order never changes a full classification; it only lets a SAFE reply be found
    early when the caller stops at the first one.
    """
    legal = game.legal_moves()
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if game.board[r][c] != EMPTY]
    if not stones:
        return legal
    last = game.history[-1] if game.history else stones[0]

    def key(move):
        near = min(max(abs(move[0] - r), abs(move[1] - c)) for r, c in stones)
        return (near > 2, near, max(abs(move[0] - last[0]), abs(move[1] - last[1])), move)

    return sorted(legal, key=key)


@dataclass
class ThreatSolver:
    node_limit: int = 100_000
    four_chain: int = 6
    prune_quiet: bool = False
    vcf_calls: int = 0
    vcf_exhausted: int = 0
    _vcf_cache: dict = field(default_factory=dict, repr=False)
    _after_cache: dict = field(default_factory=dict, repr=False)

    # -- VCF ---------------------------------------------------------------
    def vcf(self, game: Game, attacker: int) -> tuple[str, tuple[Move, ...]]:
        """VCF for ``attacker`` on the current board (board-based: ``to_play`` is ignored)."""
        key = (_board_key(game), attacker)
        cached = self._vcf_cache.get(key)
        if cached is not None:
            return cached
        self.vcf_calls += 1
        found, _, exhausted = _find_vcf_with_stats(
            game, attacker, max_fours=_unbounded_max_fours(game), node_limit=self.node_limit)
        if found is not None:
            result = (VCF_WIN, tuple(found.attack_moves))
        elif exhausted:
            self.vcf_exhausted += 1
            result = (VCF_UNKNOWN, ())
        else:
            result = (VCF_NONE, ())
        self._vcf_cache[key] = result
        return result

    # -- statuses ----------------------------------------------------------
    def after_move(self, game: Game, vct_depth: int = 0, chain: int = 0) -> tuple[str, tuple]:
        """Status of the side that just moved; the opponent is ``game.to_play``."""
        key = (_board_key(game), game.to_play, vct_depth, chain)
        cached = self._after_cache.get(key)
        if cached is None:
            cached = self._after_move(game, vct_depth, chain)
            self._after_cache[key] = cached
        return cached

    def _after_move(self, game: Game, vct_depth: int, chain: int) -> tuple[str, tuple]:
        opponent = game.to_play
        me = -opponent
        if game.done:
            # Only the side that just moved can end the game on its own move (a five, or
            # a full board). ``Game.play`` does not switch ``to_play`` after a five, so
            # ``opponent`` above is the mover here and must not be compared with the winner.
            return SAFE, ()
        wins = _winning_moves(game, opponent)
        if wins:
            return UNSAFE, ('five', wins[0])
        ours = _winning_moves(game, me)
        if len(ours) >= 2:
            return SAFE, ()  # one block cannot stop two legal fives
        if len(ours) == 1:
            block = ours[0]
            if block not in game.legal_moves():
                return SAFE, ()  # the only block is a black forbidden point
            if chain >= self.four_chain:
                return UNKNOWN, ()
            game.play(*block)
            try:
                counts, _ = self.decision(game, vct_depth, chain + 1, stop_at_safe=True)
            finally:
                game.undo()
            status = decision_status(counts)
            return status, (('block', block) if status == UNSAFE else ())

        vcf, line = self.vcf(game, opponent)
        if vcf == VCF_WIN:
            return UNSAFE, ('vcf', line)
        worst = SAFE if vcf == VCF_NONE else UNKNOWN
        if vct_depth == 0:
            return worst, ()

        for threat in ordered_moves(game):
            game.play(*threat)
            try:
                if game.done:
                    continue  # an immediate five was already excluded above
                if self.prune_quiet and self.vcf(game, opponent)[0] == VCF_NONE:
                    continue  # unsound shortcut, see module docstring
                counts, _ = self.decision(game, vct_depth - 1, 0, stop_at_safe=True)
            finally:
                game.undo()
            status = decision_status(counts)
            if status == UNSAFE:
                return UNSAFE, ('threat', threat)
            if status == UNKNOWN:
                worst = UNKNOWN
        return worst, ()

    def decision(self, game: Game, vct_depth: int = 0, chain: int = 0, *,
                 stop_at_safe: bool = False) -> tuple[Counter, dict[Move, tuple[str, tuple]]]:
        """Classify every legal move of ``game.to_play`` (stop early at a SAFE one if asked)."""
        per_move = {}
        for move in ordered_moves(game):
            game.play(*move)
            try:
                per_move[move] = self.after_move(game, vct_depth, chain)
            finally:
                game.undo()
            if stop_at_safe and per_move[move][0] == SAFE:
                break
        return Counter(status for status, _ in per_move.values()), per_move

    def classify(self, game: Game, moves, vct_depth: int = 0) -> dict[Move, tuple[str, tuple]]:
        """Status of each given move of ``game.to_play`` (the game is restored)."""
        result = {}
        for move in moves:
            game.play(*move)
            try:
                result[move] = self.after_move(game, vct_depth)
            finally:
                game.undo()
        return result
