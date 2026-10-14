"""``search.mcts_v7._find_vcf_with_stats`` with a deadline checked on every node.

The frozen recursion (``_find_vcf_recursive``) asks its state object for every node
(``take_node``), so a state subclass that raises ``SearchTimeout`` reaches inside it without a
copy of the search. The exception leaves through the recursion's ``placed`` context managers,
which restore the board; there is nothing else to restore (no history, turn or cache is touched).
With ``deadline=None`` this is the frozen function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from search.mcts_v7 import VCFResult, _find_vcf_recursive, _find_vcf_with_stats, _VCFState

from .deadline import check


@dataclass
class _DeadlineVCFState(_VCFState):
    deadline: object = field(default=None, repr=False)

    def take_node(self) -> bool:
        check(self.deadline)
        return super().take_node()


def find_vcf_with_stats(game, attacker: int, *, max_fours: int, node_limit: int, deadline=None):
    """Same return value as the frozen function; raises ``SearchTimeout`` at the deadline."""
    if deadline is None:
        return _find_vcf_with_stats(game, attacker, max_fours=max_fours, node_limit=node_limit)
    # Argument checks: the frozen function's own (it raises before searching on bad input).
    if type(node_limit) is not int or node_limit < 1 or type(max_fours) is not int or max_fours < 1:
        return _find_vcf_with_stats(game, attacker, max_fours=max_fours, node_limit=node_limit)
    state = _DeadlineVCFState(node_limit=node_limit, deadline=deadline)
    found = _find_vcf_recursive(game, attacker, max_fours=max_fours, fours_used=0, state=state)
    if found is None:
        return None, state.nodes, state.exhausted
    attack_moves, completion_points, fours = found
    result = VCFResult(first_move=attack_moves[0], fours=fours, nodes=state.nodes,
                       attack_moves=attack_moves, completion_points=completion_points)
    return result, state.nodes, state.exhausted
