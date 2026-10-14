"""V8 bounded solvers with the move deadline (docs/web-ser5.md §3.2).

``DeadlineSolver`` is ``analysis.mcts_v8._BudgetedSolver`` and ``DeadlineSelectiveSolver`` is
``analysis.selective_vct.SelectiveSolver`` plus one rule: at the deadline the solver marks itself
exhausted and raises ``SearchTimeout`` (a ``_BudgetExhausted``), which ``_bounded`` turns into
UNKNOWN exactly like a node-budget cut. So a deadline cut

- is never cached: the exception leaves ``vcf`` before its cache write and ``after_move`` before
  its own (``ThreatSolver.after_move`` stores only a returned value);
- never becomes SAFE / WIN / UNSAFE: those need a returned search, and nothing returns;
- stops the fair scheduler (``_proven`` loops while ``not solver.exhausted``) instead of letting
  it retry pending moves that would each time out at once;
- leaves the game as it was: every play in the solver is undone in a ``finally`` and the VCF
  probes restore their stones in ``placed``.

The deadline is checked when ``vcf`` is called (cache hits included: decision enumeration over
cached positions costs time the node budget does not see), on every VCF node and on every
``after_move``. With ``deadline=None`` the solvers search exactly as the frozen ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis.mcts_v8 import _BudgetedSolver
from analysis.selective_vct import threat_moves
from analysis.threats import VCF_NONE, VCF_UNKNOWN, VCF_WIN, _board_key, _unbounded_max_fours

from .deadline import SearchTimeout, check
from .vcf import find_vcf_with_stats


@dataclass
class DeadlineSolver(_BudgetedSolver):
    deadline: object = field(default=None, repr=False)
    timed_out: bool = False

    def _check_deadline(self) -> None:
        try:
            check(self.deadline)
        except SearchTimeout:
            self._timeout()

    def _timeout(self):
        self.timed_out = True
        self.exhausted = True  # stops ``_proven``; ``_bounded`` returns UNKNOWN from now on
        raise SearchTimeout

    def after_move(self, game, vct_depth: int = 0, chain: int = 0):
        self._check_deadline()
        return super().after_move(game, vct_depth, chain)

    def vcf(self, game, attacker: int):
        # ``_BudgetedSolver.vcf`` with the deadline-aware VCF search; budget logic unchanged.
        self._check_deadline()
        key = (_board_key(game), attacker)
        cached = self._vcf_cache.get(key)
        if cached is not None:
            return cached
        remaining = min(self._cap, self.node_budget) - self.nodes_used
        if self.vcf_calls >= min(self._call_cap, self.call_limit) or remaining < 1:
            self._stop()
        limit = min(self.node_limit, remaining)
        self.vcf_calls += 1
        try:
            found, nodes, cut = find_vcf_with_stats(
                game, attacker, max_fours=_unbounded_max_fours(game), node_limit=limit, deadline=self.deadline)
        except SearchTimeout:
            self._timeout()
        self.nodes_used += nodes
        if found is not None:
            result = (VCF_WIN, tuple(found.attack_moves))
        elif cut and limit < self.node_limit:
            self._stop()
        elif cut:
            self.vcf_exhausted += 1
            result = (VCF_UNKNOWN, ())
        else:
            result = (VCF_NONE, ())
        self._vcf_cache[key] = result
        return result


@dataclass
class DeadlineSelectiveSolver(DeadlineSolver):
    """``SelectiveSolver`` (attacker quiet moves: threat moves only) with the deadline."""

    def quiet_moves(self, game):
        return threat_moves(game)
