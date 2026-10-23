"""H7-E4-v2: the E4 attack with cheaper activation (docs/h6-value.md, "E4-v2 설계 확정").

Development module. ``analysis.e4_attack`` (E4-v1, frozen with its candidate) is used as it is; this module
adds, in front of and inside it:

    G1 opponent-VCF gate   (fixed, every configuration) no attack when the opponent has a VCF if we pass
                           (board VCF at ``GATE_NODE_LIMIT`` nodes). A cut VCF (UNKNOWN) does not stop it.
    C  cooldown            (axis) after an own attack that used its whole budget without a WIN, the next own
                           move does not attack; the flag is cleared at that move whatever it does
                           (``Cooldown``: one bit per game). A gated or skipped move, and an attack that
                           finished without a WIN, set nothing.
    K  killer replies      (axis) inside the attack, the defender replies that refuted an earlier check are
                           tried first (``KillerSolver``: latest 8 per depth and side to move). The reply set is
                           unchanged, so WIN / REFUTED mean what they mean without it; only the order, and so
                           what a budget reaches, changes.
    budget                 (axis) 200k nodes / 20k calls, or 150k / 15k (20k nodes per VCF call)
The first moves (fours, open threes, local VCF threats), the proof (``vct2_attack._depth2_attack_check``: every
defender reply), the fair-share schedule and the routes (stage4 / stage5 / tree) are E4-v1's.
"""
from __future__ import annotations

from collections import Counter
from time import perf_counter

from renju import Game
from search.mcts_v5 import _RootContext

from . import e4_attack
from .mcts_v8 import WIN, SearchDiagnostics, _attack_candidates, _first_proven
from .selective_vct import SelectiveSolver
from .threats import SAFE, _unbounded_max_fours, ordered_moves
from .vct2_attack import _depth2_attack_check

Move = tuple[int, int]
ROUTES = e4_attack.ROUTES
GATE_NODE_LIMIT = 20_000
BUDGETS = {'200k': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 200_000},
           '150k': {'node_limit': 20_000, 'call_limit': 15_000, 'node_budget': 150_000}}
KILLERS = 8
# The E4-v2 DEV grid (fixed before DEV): cooldown x killer replies x budget; G1 in all of them.
GRID = tuple({'name': f"{b}/{'cd' if c else 'nocd'}/{'killer' if k else 'plain'}", 'budget': b, 'cooldown': c,
              'killer': k} for b in BUDGETS for c in (False, True) for k in (False, True))


class KillerSolver(SelectiveSolver):
    """``SelectiveSolver`` whose decisions try the latest refuting replies first (same reply set)."""

    def decision(self, game: Game, vct_depth: int = 0, chain: int = 0, *, stop_at_safe: bool = False):
        killers = self.__dict__.setdefault('killers', {})
        key = (vct_depth, game.to_play)
        legal = ordered_moves(game)
        present = set(legal)
        first = [m for m in killers.get(key, ()) if m in present]
        chosen = set(first)
        per_move = {}
        for move in first + [m for m in legal if m not in chosen]:
            game.play(*move)
            try:
                per_move[move] = self.after_move(game, vct_depth, chain)
            finally:
                game.undo()
            if stop_at_safe and per_move[move][0] == SAFE:
                recent = killers.setdefault(key, [])
                if move in recent:
                    recent.remove(move)
                recent.insert(0, move)
                del recent[KILLERS:]
                break
        return Counter(status for status, _ in per_move.values()), per_move


def opponent_vcf(game: Game) -> dict:
    """G1: the opponent's VCF on this board if we passed: 'WIN', 'NONE' or 'UNKNOWN' (cut), and its cost."""
    from search.mcts_v7 import _find_vcf_with_stats  # looked up at call time (the fast modes rebind it)

    found, nodes, cut = _find_vcf_with_stats(game, -game.to_play, max_fours=_unbounded_max_fours(game),
                                            node_limit=GATE_NODE_LIMIT)
    status = 'WIN' if found is not None else 'UNKNOWN' if cut else 'NONE'
    return {'status': status, 'nodes': nodes, 'calls': 1}


def attack(game: Game, budget: str, killer: bool) -> tuple[Move | None, dict]:
    """E4-v1's attack (VCF-threat first moves) with the budget and, if asked, killer reply ordering."""
    values = BUDGETS[budget]
    started = perf_counter()
    context = _RootContext(game.legal_moves(), SearchDiagnostics())
    cls = KillerSolver if killer else SelectiveSolver
    solver = cls(node_limit=values['node_limit'], call_limit=values['call_limit'], node_budget=values['node_budget'])
    candidates = _attack_candidates(game, context)
    base = len(candidates)
    candidates = candidates + e4_attack.vcf_threat_moves(game, solver, candidates, context)

    def evaluate(g, move, share, call_share):
        return solver._bounded(g, move, share, call_share, lambda after: _depth2_attack_check(solver, after))

    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, evaluate, WIN) if candidates else None
    return chosen, {'move': list(chosen) if chosen is not None else None, 'candidates': len(candidates),
                    'vcf_threat_candidates': len(candidates) - base, 'calls': solver.vcf_calls,
                    'nodes': solver.nodes_used, 'exhausted': solver.exhausted,
                    'seconds': round(perf_counter() - started, 4)}


class Cooldown:
    """One bit per game: set by an exhausted attack without a WIN, consumed at the next own move."""

    def __init__(self):
        self.history: tuple | None = None  # the game history of the move that set the flag

    def blocks(self, game: Game) -> bool:
        """True if this move must not attack; the flag is cleared either way (one move only). Called at every
        own move; a later move than the next own one is never blocked (exactly two plies on the same line)."""
        prev, self.history = self.history, None
        if prev is None:
            return False
        now = tuple(game.history)
        return len(now) == len(prev) + 2 and now[:len(prev)] == prev  # the next own move of the same game

    def exhausted(self, game: Game) -> None:
        self.history = tuple(game.history)


def decide(game: Game, config: dict, base_route: str, cooldown: Cooldown | None = None) -> tuple[Move | None, dict]:
    """The E4-v2 step after the base move: route, cooldown, G1, then the attack. (move or None, record)."""
    # the bit is consumed at every own move, before the route decides (a forced move clears it too)
    blocked = config['cooldown'] and cooldown is not None and cooldown.blocks(game)
    if base_route not in ROUTES:
        return None, {'ran': False, 'skip': 'route'}
    if blocked:
        return None, {'ran': False, 'skip': 'cooldown'}
    gate = opponent_vcf(game)
    if gate['status'] == 'WIN':
        return None, {'ran': False, 'skip': 'opponent_vcf', 'gate': gate}
    move, info = attack(game, config['budget'], config['killer'])
    if config['cooldown'] and cooldown is not None and move is None and info['exhausted']:
        cooldown.exhausted(game)
    return move, {'ran': True, 'gate': gate, **info}
