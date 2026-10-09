"""E2 opponent: a V8 agent that also plays selective depth-2 attacks (docs/mcts-v8-teacher.md §12.25 E2).

The question E2 needs an opponent for: when the tested engine plays a move that loses to a
depth-2 VCT ("at most two quiet moves, then VCF"), is it punished? ``v8:full`` attacks only
to depth 1 (V8-B), so it rarely punishes such a move and the VCT2 veto shows no effect.

``own_vct2_attack`` looks for a move ``m`` of the side to move such that every reply of the
opponent loses within depth 1 (VCF, or one quiet move + VCF):

- candidates are V8-B's (moves that make a four or an open three, ``_attack_candidates``);
- after ``m`` every legal reply is played (``ThreatSolver.decision``), so a WIN is a proof;
- our quiet move at depth 1 is restricted to threat moves (``SelectiveSolver``), which only
  makes the search incomplete ("no WIN found" is not "no depth-2 win"), never unsound;
- budgets are node and call counts (``VCT2_ATTACK_BUDGET``), shared fairly over the
  candidates exactly as V8-B shares its own (``_first_proven``). Only a proven WIN is played.

``VCT2AttackAgent`` wraps a V8 agent and does not change it: the base agent chooses first; when
its route is one where it found no win of its own (``ATTACK_ROUTES``: Stage 4/5 defenses and
the tree) the depth-2 attack runs, and a proven WIN replaces the base move. A depth-1 attack
(V8-B, route ``own_vct``), an own VCF and every forced Stage 1-3 move stand as the base chose
them. With no WIN the base move is played, so the wrapper never makes the base weaker in its
own class.
"""
from __future__ import annotations

from time import perf_counter

from renju import Game
from search.mcts_v5 import _RootContext

from .mcts_v8 import REFUTED, WIN, SearchDiagnostics, _attack_candidates, _first_proven
from .selective_vct import SelectiveSolver
from .threats import SAFE, UNKNOWN, UNSAFE, decision_status

Move = tuple[int, int]
VCT2_ATTACK_BUDGET = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 200_000}
ATTACK_ROUTES = ('stage4', 'stage5', 'tree')
ROUTE = 'own_vct2'


def _depth2_attack_check(solver: SelectiveSolver, game: Game) -> str:
    """After our move: WIN if every reply loses within depth 1, REFUTED if one survives."""
    if game.done:
        return WIN if game.winner is not None else REFUTED
    if not game.legal_moves():
        return REFUTED
    counts, _ = solver.decision(game, 1, stop_at_safe=True)
    return {UNSAFE: WIN, SAFE: REFUTED, UNKNOWN: UNKNOWN}[decision_status(counts)]


def own_vct2_attack(game: Game, budget: dict | None = None) -> tuple[Move | None, dict]:
    """The first proven depth-2 attack of ``game.to_play`` (or None) and its diagnostics."""
    budget = budget or VCT2_ATTACK_BUDGET
    started = perf_counter()
    candidates = _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))
    solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])

    def evaluate(g, move, share, call_share):
        return solver._bounded(g, move, share, call_share, lambda after: _depth2_attack_check(solver, after))

    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, evaluate, WIN)
    info = {
        'status': WIN if chosen is not None else '',
        'move': list(chosen) if chosen is not None else None,
        'rank': candidates.index(chosen) + 1 if chosen is not None else 0,
        'candidates': len(candidates),
        'checked': [[list(m), s] for m, s in statuses.items()],
        'calls': solver.vcf_calls, 'nodes': solver.nodes_used, 'exhausted': solver.exhausted,
        'seconds': round(perf_counter() - started, 4),
    }
    return chosen, info


class VCT2AttackAgent:
    """A V8 agent plus ``own_vct2_attack`` on ``ATTACK_ROUTES`` (the E2 opponent)."""

    def __init__(self, base, budget: dict | None = None):
        self.base = base
        self.budget = dict(budget or VCT2_ATTACK_BUDGET)
        self.name = f'{base.name}+vct2atk'
        self.attack_info: dict = {}

    @property
    def diagnostics(self):
        return self.base.diagnostics

    def select_move(self, game: Game) -> Move:
        move = self.base.select_move(game)
        self.attack_info = {}
        route = self.base.diagnostics.v8_route
        if route not in ATTACK_ROUTES:
            return move
        attack, info = own_vct2_attack(game, self.budget)
        self.attack_info = {'base_route': route, 'base_move': list(move), **info}
        if attack is None:
            return move
        diag = self.base.diagnostics
        diag.v8_route = ROUTE
        diag.v8_changed = attack != move
        return attack
