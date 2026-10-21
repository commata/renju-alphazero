"""H7-E4 candidate: the E2 selective VCT2 attack with a configurable first-move set and trigger.

Development module (docs/h6-value.md, "H7-E4 DEV 설계 수정안" and its approval). Not an engine yet: the
selected configuration is frozen later in a new engine file; ``analysis.vct2_attack`` (E2) is unchanged.

The attack is ``vct2_attack.own_vct2_attack`` with two options:

    candidates  'four_three'  V8-B's first moves (``mcts_v8._attack_candidates``: a four or an open three)
                'vcf_threat'  those, then the local VCF-threat moves: legal moves within Chebyshev distance 2
                              of a stone, not already candidates, after which the attacker has a VCF if the
                              defender passed (board VCF at the solver's per-call node limit). Ordered by the
                              V3.2.1 priority, then the V5 root key. Their detection is paid from the budget.
    trigger     'all'         every position the wrapper is asked about
                'threat'      only when the opponent's last move made a four or an open three for it

Soundness is the E2 module's: a WIN needs every legal reply of the defender to lose within depth 1
(``_depth2_attack_check``); the first-move set and the trigger only change what is tried.
"""
from __future__ import annotations

from time import perf_counter

from renju import EMPTY, SIZE, Game
from search.mcts_v321 import _fast_pattern_features_for_move, _v321_priority_score
from search.mcts_v5 import _RootContext

from .mcts_v8 import REFUTED, WIN, SearchDiagnostics, _attack_candidates, _first_proven
from .selective_vct import SelectiveSolver
from .threats import VCF_WIN
from .vct2_attack import ATTACK_ROUTES, _depth2_attack_check

Move = tuple[int, int]
CANDIDATES = ('four_three', 'vcf_threat')
TRIGGERS = ('all', 'threat')
BUDGETS = {'200k': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 200_000},
           '400k': {'node_limit': 20_000, 'call_limit': 40_000, 'node_budget': 400_000}}
# The DEV grid (fixed before DEV): budget x first-move candidates x trigger; routes fixed to E2's.
GRID = tuple({'name': f'{b}/{c}/{t}', 'budget': b, 'candidates': c, 'trigger': t}
             for b in BUDGETS for c in CANDIDATES for t in TRIGGERS)
ROUTES = ATTACK_ROUTES


def _local(game: Game, move: Move, radius: int = 2) -> bool:
    r0, c0 = move
    for r in range(max(0, r0 - radius), min(SIZE, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(SIZE, c0 + radius + 1)):
            if game.board[r][c] != EMPTY:
                return True
    return False


def opponent_threat(game: Game) -> bool:
    """Did the opponent's last move make a four or an open three for it? (the game is restored)"""
    if not game.history:
        return False
    last = game.history[-1]
    player = game.board[last[0]][last[1]]
    game.undo()
    try:
        features = _fast_pattern_features_for_move(game, player, last, assume_legal=True)
    finally:
        game.play(*last)
    return features.four_directions > 0 or features.open_three_directions > 0


def vcf_threat_moves(game: Game, solver, exclude, context: _RootContext) -> list[Move]:
    """Local legal moves (not in ``exclude``) after which ``game.to_play`` has a VCF if the defender passed."""
    me, skip, out = game.to_play, set(exclude), []

    def check(after):
        if after.done:
            return WIN if after.winner is not None else REFUTED
        return WIN if solver.vcf(after, me)[0] == VCF_WIN else REFUTED

    for move in context.legal:
        if move in skip or not _local(game, move):
            continue
        if solver.exhausted:
            break
        if solver._bounded(game, move, None, None, check) == WIN:
            out.append(move)
    return sorted(out, key=lambda m: (-_v321_priority_score(game, m), context.key(game, m)))


def e4_attack(game: Game, config: dict) -> tuple[Move | None, dict]:
    """The first proven depth-2 attack of ``game.to_play`` under ``config`` (or None) and its diagnostics."""
    budget = BUDGETS[config['budget']]
    started = perf_counter()
    context = _RootContext(game.legal_moves(), SearchDiagnostics())
    solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    candidates = _attack_candidates(game, context)
    base = len(candidates)
    if config['candidates'] == 'vcf_threat':
        candidates = candidates + vcf_threat_moves(game, solver, candidates, context)
    detection = {'calls': solver.vcf_calls, 'nodes': solver.nodes_used}

    def evaluate(g, move, share, call_share):
        return solver._bounded(g, move, share, call_share, lambda after: _depth2_attack_check(solver, after))

    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, evaluate, WIN) if candidates else None
    return chosen, {
        'status': WIN if chosen is not None else '', 'move': list(chosen) if chosen is not None else None,
        'rank': candidates.index(chosen) + 1 if chosen is not None else 0,
        'candidates': len(candidates), 'vcf_threat_candidates': len(candidates) - base,
        'detection': detection, 'calls': solver.vcf_calls, 'nodes': solver.nodes_used,
        'exhausted': solver.exhausted, 'seconds': round(perf_counter() - started, 4)}


def triggered(game: Game, config: dict, base_route: str) -> bool:
    """Whether the wrapper runs the attack: the base route is one of E2's and the trigger holds."""
    if base_route not in ROUTES:
        return False
    return config['trigger'] == 'all' or opponent_threat(game)
