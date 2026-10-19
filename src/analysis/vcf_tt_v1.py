"""H7-E1 candidate: the frozen VCF search with a call-local table of exact no-win states (docs/h6-value.md).

``find_vcf_with_stats`` has the signature and return values of ``search.mcts_v7._find_vcf_with_stats``
and runs the same recursion (same candidates in the same order, same node accounting), with one change:
inside one call it remembers every state whose search returned no VCF without a node-limit cut, and
answers a later visit of exactly that state with "no VCF" at no node cost.

State key: (board, attacker, remaining fours = max_fours - fours_used), exact (no D4, no "at most this
many fours" generalisation; that is a later step). The recursion depends on nothing else: the VCF helpers
read neither ``history`` nor ``to_play``, and stones are only added, so a key cannot recur inside its own
search.

Why this is exact: a stored state's subtree returned None without a cut, so the frozen recursion would
return None there again. Skipping it changes no candidate order and no first VCF found; only the node
count drops. Hence, whenever the frozen call finishes without a cut, this call returns the same result
and witness with no more nodes. Results can differ only where the frozen call was cut, which is the
point: there the freed nodes may finish the search. Wins and cut results are never stored.

The table lives for one call only (no sharing across calls, decisions, games or workers), so results do
not depend on scheduling. Helpers are read from ``search.mcts_v7`` at call time, so the exact ``--fast``
speed-ups apply unchanged. Nothing here edits a frozen file; ``scripts/h7_e1_vcf_table.py`` swaps this in
for the V8-C solver only.
"""
from __future__ import annotations

from dataclasses import dataclass

from search import mcts_v7 as v7


@dataclass
class TableStats:
    """Totals over the calls made with one ``make_find``: table stores and hits."""
    calls: int = 0
    stores: int = 0
    hits: int = 0


def _key(game, attacker, remaining):
    return tuple(map(tuple, game.board)), attacker, remaining


def _recursive(game, attacker, *, max_fours, fours_used, state, table):
    if state.exhausted:
        return None
    key = _key(game, attacker, max_fours - fours_used)
    if key in table:
        table[None] += 1  # hit counter
        return None
    result = _body(game, attacker, max_fours=max_fours, fours_used=fours_used, state=state, table=table)
    if result is None and not state.exhausted:
        table[key] = True
    return result


def _body(game, attacker, *, max_fours, fours_used, state, table):
    """``search.mcts_v7._find_vcf_recursive`` after its exhausted check, recursing through the table."""
    immediate = sorted(v7._winning_moves(game, attacker))
    if immediate:
        return (immediate[0],), (), fours_used
    if fours_used >= max_fours:
        return None
    defender = -attacker
    for move in v7._window_candidates(game, attacker, 3):
        if not state.take_node():
            return None
        if not v7._is_legal_for_player(game, attacker, move):
            continue
        if v7._wins_for_player(game, attacker, move):
            return (move,), (), fours_used
        with v7.placed(game, attacker, move):
            completions = v7._legal_completions(game, attacker, move)
            if not completions:
                continue
            if v7._winning_moves(game, defender):
                continue
            if len(completions) >= 2:
                return (move,), completions, fours_used + 1
            block = completions[0]
            if not v7._is_legal_for_player(game, defender, block):
                return (move,), completions, fours_used + 1
            if v7._wins_for_player(game, defender, block):
                continue
            with v7.placed(game, defender, block):
                if v7._reverse_four_after_block(game, defender, block):
                    continue
                child = _recursive(game, attacker, max_fours=max_fours, fours_used=fours_used + 1,
                                   state=state, table=table)
                if child is None:
                    continue
                attack_moves, child_completions, child_fours = child
                return ((move,) + attack_moves,
                        tuple(sorted(set(completions).union(child_completions))), child_fours)
    return None


def make_find(stats: TableStats | None = None):
    """A drop-in ``_find_vcf_with_stats`` with a fresh table per call; totals go to ``stats``."""

    def find(game, attacker, *, max_fours, node_limit):
        if attacker not in (v7.BLACK, v7.WHITE):
            raise ValueError('attacker must be BLACK or WHITE')
        if type(max_fours) is not int or max_fours < 1:
            raise ValueError('max_fours must be a positive integer')
        if type(node_limit) is not int or node_limit < 1:
            raise ValueError('node_limit must be a positive integer')
        state = v7._VCFState(node_limit=node_limit)
        table = {None: 0}
        found = _recursive(game, attacker, max_fours=max_fours, fours_used=0, state=state, table=table)
        if stats is not None:
            stats.calls += 1
            stats.hits += table[None]
            stats.stores += len(table) - 1
        if found is None:
            return None, state.nodes, state.exhausted
        attack_moves, completion_points, fours = found
        return (v7.VCFResult(first_move=attack_moves[0], fours=fours, nodes=state.nodes, attack_moves=attack_moves,
                             completion_points=completion_points), state.nodes, state.exhausted)
    return find


find_vcf_with_stats = make_find()


def verify_line(game, attacker, attack_moves, max_fours) -> bool:
    """Replay a VCF line by the recursion's own rules, independent of any table: every attacker move but
    the last makes a four with one completion that the defender must and may block (no defender five, no
    reverse four); the last move makes five, or a four the defender cannot stop. ``game`` is restored."""
    defender = -attacker
    if not attack_moves or len(attack_moves) > max_fours + 1:
        return False
    undo = []
    try:
        for i, move in enumerate(attack_moves):
            last = i == len(attack_moves) - 1
            if v7._winning_moves(game, attacker):  # a five was already on offer: the line ends here
                return last and move in v7._winning_moves(game, attacker)
            if not v7._is_legal_for_player(game, attacker, move):
                return False
            if v7._wins_for_player(game, attacker, move):
                return last
            ctx = v7.placed(game, attacker, move)
            ctx.__enter__()
            undo.append(ctx)
            completions = v7._legal_completions(game, attacker, move)
            if not completions or v7._winning_moves(game, defender):
                return False
            block = completions[0]
            if len(completions) >= 2 or not v7._is_legal_for_player(game, defender, block):
                return last
            if last or v7._wins_for_player(game, defender, block):
                return False
            ctx = v7.placed(game, defender, block)
            ctx.__enter__()
            undo.append(ctx)
            if v7._reverse_four_after_block(game, defender, block):
                return False
        return False
    finally:
        for ctx in reversed(undo):
            ctx.__exit__(None, None, None)
