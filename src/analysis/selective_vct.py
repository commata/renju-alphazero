"""S2-3 selective VCT2 detector (docs/mcts-v8-teacher.md §12.18, §12.23).

Question: after ``move`` by the side to move, does the opponent have a forced win of the
class "at most two quiet moves, then VCF"?

- Depths 0 and 1 use the full bounded solver (``analysis.mcts_v8._BudgetedSolver``, every
  legal quiet move), the same class as V8-A/V8-C.
- Depth 2 uses ``SelectiveSolver``: the attacker's quiet moves are restricted to threat
  moves (a four or an open/split three for the attacker, ``_fast_pattern_features_for_move``).
  The defender's replies are always every legal move (``ThreatSolver.decision``), so a
  PROVEN_LOSS is a proof; pruning only the attacker makes the search incomplete, never
  unsound.

Statuses (never "SAFE"):

    PROVEN_LOSS              lost at ``depth`` (0, 1: full class; 2: found by the selective search)
    NO_TARGETED_VCT2_FOUND   depth <= 1 exhausted without a loss and the selective depth-2
                             search found none; deeper or non-threat wins are not excluded
    UNKNOWN                  a budget or a single-VCF node limit cut the search (``cause``)

``verify_witness`` re-checks a depth-2 PROVEN_LOSS with the full ``ThreatSolver`` along
the witness threat (every defender reply at depth 1, all quiet moves of the attacker).
Budgets are node and call counts only (no time cut), so results are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from renju import Game
from search.mcts_v321 import _fast_pattern_features_for_move

from .mcts_v8 import _BudgetedSolver
from .threats import SAFE, UNKNOWN, UNSAFE, ThreatSolver, decision_status, ordered_moves

Move = tuple[int, int]
PROVEN_LOSS, NO_TARGETED_VCT2_FOUND = 'PROVEN_LOSS', 'NO_TARGETED_VCT2_FOUND'
DEFAULT_BUDGET = {
    'full': {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000},       # depth 0-1 (V8-C root)
    'selective': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 1_000_000},  # depth 2
}


def threat_moves(game: Game) -> list[Move]:
    """Legal moves of ``game.to_play`` that make a four or an open (or split) three, nearest first."""
    player = game.to_play
    out = []
    for move in ordered_moves(game):
        features = _fast_pattern_features_for_move(game, player, move, assume_legal=True)
        if features.four_directions or features.open_three_directions:
            out.append(move)
    return out


@dataclass
class SelectiveSolver(_BudgetedSolver):
    """``_BudgetedSolver`` whose attacker tries only threat moves as quiet moves."""

    def quiet_moves(self, game: Game) -> list[Move]:
        return threat_moves(game)


def _solver(cls, budget):
    return cls(node_limit=budget['node_limit'], call_limit=budget['call_limit'], node_budget=budget['node_budget'])


def _run(solver, game, move, depth):
    """Status and witness of the side playing ``move`` at ``depth`` (game restored)."""
    result = {}

    def check(g):
        status, witness = solver.after_move(g, depth)
        result['witness'] = witness
        return status

    status = solver._bounded(game, move, None, None, check)
    return status, result.get('witness', ())


def _cause(solvers) -> str:
    if any(s.exhausted for s in solvers):
        return 'budget'
    if any(s.vcf_exhausted for s in solvers):
        return 'vcf_node_limit'
    return 'chain_or_other'


def _stats(solvers) -> dict:
    return {'vcf_calls': sum(s.vcf_calls for s in solvers), 'nodes_used': sum(s.nodes_used for s in solvers)}


def classify(game: Game, move: Move, budget: dict | None = None) -> dict:
    """Detector result for ``move`` of ``game.to_play`` (the game is restored)."""
    budget = budget or DEFAULT_BUDGET
    started = perf_counter()
    full = _solver(_BudgetedSolver, budget['full'])
    for depth in (0, 1):
        status, witness = _run(full, game, move, depth)
        if status in (UNSAFE, UNKNOWN):
            seconds = round(perf_counter() - started, 3)
            out = {'seconds': seconds, 'stage_seconds': {'full': seconds, 'selective': 0.0},
                   'stage_nodes': {'full': full.nodes_used, 'selective': 0}, **_stats([full])}
            if status == UNSAFE:
                return {'status': PROVEN_LOSS, 'depth': depth, 'witness': _plain(witness), **out}
            return {'status': UNKNOWN, 'depth': None, 'cause': _cause([full]), 'stage': f'full_depth{depth}', **out}
    full_seconds = perf_counter() - started
    selective = _solver(SelectiveSolver, budget['selective'])
    status, witness = _run(selective, game, move, 2)
    # The depth 0-1 stage is the class V8-C already checks on the tree route; the selective
    # stage is the cost the detector adds there.
    out = {'seconds': None, 'stage_nodes': {'full': full.nodes_used, 'selective': selective.nodes_used},
           **_stats([full, selective])}
    if status == UNSAFE:
        out.update(status=PROVEN_LOSS, depth=2, witness=_plain(witness))
    elif status == UNKNOWN:
        out.update(status=UNKNOWN, depth=None, cause=_cause([selective]), stage='selective_depth2')
    else:
        out.update(status=NO_TARGETED_VCT2_FOUND, depth=None)
    total = perf_counter() - started
    out['seconds'] = round(total, 3)
    out['stage_seconds'] = {'full': round(full_seconds, 3), 'selective': round(total - full_seconds, 3)}
    return out


def _plain(witness):
    """JSON-friendly witness (kind, then moves as lists)."""
    if not witness:
        return []
    kind, payload = witness[0], witness[1]
    if kind == 'vcf':
        return [kind, [list(m) for m in payload]]
    return [kind, list(payload)]


def verify_witness(game: Game, move: Move, result: dict, node_limit: int = 20_000) -> bool | None:
    """Re-check a depth-2 PROVEN_LOSS with the full solver along its witness threat.

    Returns True when the full search confirms that after ``move`` and the witness threat
    every defender reply loses within depth 1, False when it does not, and None when the
    full search could not finish (an UNKNOWN reply). Depth 0/1 results come from the full
    solver already and are returned as True.
    """
    if result.get('status') != PROVEN_LOSS:
        raise ValueError('only PROVEN_LOSS results carry a witness')
    if result['depth'] < 2:
        return True
    kind, payload = result['witness']
    solver = ThreatSolver(node_limit=node_limit)
    game.play(*move)
    try:
        if kind == 'threat':
            game.play(*payload)
            try:
                counts, _ = solver.decision(game, 1, stop_at_safe=True)
            finally:
                game.undo()
            status = decision_status(counts)
        else:  # 'block' / 'five' / 'vcf' at depth 2: re-run the full class on the position itself
            status, _ = solver.after_move(game, 2)
    finally:
        game.undo()
    return {UNSAFE: True, SAFE: False}.get(status)
