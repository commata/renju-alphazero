"""MCTS-v8 teacher engine: the frozen V7 flow plus bounded VCT1 safety.

Design: docs/mcts-v8-teacher.md. V8 lives in ``analysis`` because it needs the
depth-1 VCT solver (``analysis.threats``) and is an offline teacher/evaluation
engine, never part of the AlphaZero search or training path.

Implemented modules (the rest of the design is not yet here):

- V8-1 skeleton: with every module off, ``mcts_search_v8`` makes exactly the
  moves of ``search.mcts_v7.mcts_search_v7`` (same random stream), and records
  the root visits of tree-searched moves (V8-F root record).
- V8-B (``own_vct_attack``): after V7's own VCF (M1) finds nothing, moves that
  make a four or an open three are tried as depth-1 VCT attacks. A move is
  played only when every opponent reply is proven lost to our VCF (WIN); an
  unfinished proof (UNKNOWN) is never played. The candidate list is a speed
  heuristic and is incomplete: "no WIN found" does not mean "no VCT1 win".
- V8-C (``root_vct_safety``): on the tree route V7's own tree runs unchanged
  (same root, same random stream, so the same move as V7), then that move is
  checked for depth-1 VCT safety. If it is proven SAFE it is played. Otherwise
  the other root children are checked in tree preference order (visits, then
  mean value) and the first proven SAFE one is played; with none proven SAFE
  V7's move stands unless it is proven UNSAFE, in which case the first
  unrefuted child that does not lose at once is played. That is the
  ``root_vct_mode="aggressive"`` rule (the default): an unfinished check on
  V7's move (UNKNOWN) is enough to switch to a proven-SAFE lower child. With
  ``root_vct_mode="veto"`` only a proven loss (UNSAFE) of V7's move switches,
  to the best-ranked child that is not itself proven UNSAFE, with the children
  checked exactly as in aggressive mode (design §12.3, §12.8).
- V8-A (``stage_vct_safety``): Stage 4 and single-point Stage 5 forced defenses
  are checked with a depth-1 VCT proof in V7 order and the first proven SAFE one
  is played. If every forced defense is proven UNSAFE the check widens to the
  V6 root candidates. A per-move budget on VCF calls and total VCF nodes bounds
  the cost; an unfinished proof is UNKNOWN and is never treated as SAFE.

V8-A, V8-B and V8-C each have their own solver, cache and budget, and share one
fair budget scheduler (``_proven``).

SAFE here means "no VCF and no quiet-move + VCF win for the opponent" within the
frozen VCF solver's class (see ``analysis.threats``), not "no forced win". The
design doc calls it NOT_REFUTED_VCT1 and UNSAFE PROVEN_LOSS_VCT1 (§12.2); the
stored strings stay "SAFE"/"UNSAFE" so earlier result files remain comparable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from random import Random
from time import perf_counter

from renju import EMPTY, Game
from search.mcts import MCTSNode, _backpropagate, _select_child
from search.mcts_v3 import _can_expand, _pop_ranked_untried
from search.mcts_v321 import (
    _fast_pattern_features_for_move, _rollout_v321, _search_candidates_v321, _v321_priority_score,
)
from search.mcts_v5 import (
    _RootContext, _double_threat_moves, _forced_v5_move, _threat_windows,
    _unstoppable_four_moves, _validate_v5_config, _winning_moves,
)
from search.mcts_v6 import PRIORITY, _root_candidates_v6
from search.mcts_v7 import (
    V7_FINAL, SearchDiagnostics as V7Diagnostics, _find_vcf_with_stats,
    _penalize_within_tiers, _stage4_v7_move, _validate_v7_config, _vcf_safety_tiers,
)
from search.threat_patterns import placed

from .puct_v8 import PUCT_PRIORS, PUCTStats, prior_entropy, search_tree_puct
from .threats import (
    SAFE, UNKNOWN, UNSAFE, VCF_NONE, VCF_UNKNOWN, VCF_WIN, ThreatSolver, _board_key,
    _unbounded_max_fours, decision_status,
)

Move = tuple[int, int]

# Design-doc names of the V8-A/V8-C statuses (§12.2): a SAFE move is only "not refuted
# by a depth-1 VCT within the budget", an UNSAFE move is a proven loss in that class.
NOT_REFUTED_VCT1, PROVEN_LOSS_VCT1 = SAFE, UNSAFE
ROOT_VCT_MODES = ("aggressive", "veto")
TREE_MODES = ("v5", "puct")

# V8-B attack statuses (the side to move attacks; kept apart from V8-A's SAFE/UNSAFE).
WIN, REFUTED = 'WIN', 'REFUTED'

# Development defaults. V8_PLAY / V8_TEACHER budgets are fixed at V8-5 (design §4.5)
# and the whole dict is frozen at V8-6; until then these values may change.
V8_DEFAULTS = {
    **V7_FINAL,
    "stage_vct_safety": True,
    "vct_vcf_node_limit": 20_000,
    "vct_call_limit": 3_000,
    "vct_node_budget": 200_000,
    "own_vct_attack": True,
    "attack_vcf_node_limit": 20_000,
    "attack_call_limit": 10_000,  # 3,000 ran out on 164445 ply 15 (§4.2.9); nodes bound the time
    "attack_node_budget": 200_000,
    "root_vct_safety": True,
    "root_vcf_node_limit": 20_000,
    "root_call_limit": 10_000,
    # V8-C root gate (§4.3.5): 3/3 needs both the 4-child cap (budget is not spread over ~17
    # children) and 400k nodes (one V7 move alone needed 128,917 nodes to be proven SAFE).
    "root_node_budget": 400_000,
    "root_max_children": 4,  # 0 = every root child; N = the tree's top N first, wider only if all UNSAFE
    # "aggressive": switch unless V7's move is proven SAFE (V8-C as measured in §11.14);
    # "veto": switch only when V7's move is a proven loss (§12.3).
    "root_vct_mode": "aggressive",
    # H4 (§12.13): order the tree's root candidates by a policy (``root_policy`` callable,
    # supplied by the caller, e.g. hybrid.h4_policy). Off by default: V8 needs no torch.
    "root_policy_order": False,
    "root_policy_extra": 0,  # add up to N policy top moves missing from the V6 root candidates
    # H5 (§12.16): the tree route's tree. "v5" = the V5 widening tree (V7's search);
    # "puct" = analysis.puct_v8 on the same action sets and rollouts, prior ``puct_prior``
    # ("uniform" / "heuristic" / "policy"; "policy" needs the ``root_policy`` callable).
    "tree_mode": "v5",
    "puct_prior": "uniform",
    "puct_c": 1.5,
    # S3-VCT2 (§12.24): after V8-C on the tree route, a selective depth-2 check
    # (analysis.selective_vct) of the played move; a proven loss switches to the next
    # tree-ranked child that is not proven lost (UNKNOWN allowed). Off by default.
    "root_vct2_check": False,
    "vct2_node_limit": 20_000,
    "vct2_call_limit": 20_000,
    "vct2_node_budget": 10_000,
    "vct2_max_children": 4,
}
_V8_KEYS = tuple(key for key in V8_DEFAULTS if key not in V7_FINAL)


@dataclass
class SearchDiagnostics(V7Diagnostics):
    v8_route: str = ""
    v8_v7_move: Move | None = None
    v8_changed: bool = False
    v8_vct_checked: tuple[tuple[Move, str], ...] = ()
    v8_vct_widened: bool = False
    v8_vct_calls: int = 0
    v8_vct_nodes: int = 0
    v8_vct_budget_exhausted: bool = False
    v8_vct_seconds: float = 0.0
    v8_root_visits: tuple[tuple[Move, int, float], ...] = ()
    v8_attack_status: str = ""
    v8_attack_move: Move | None = None
    v8_attack_rank: int = 0
    v8_attack_candidates: int = 0
    v8_attack_checked: tuple[tuple[Move, str], ...] = ()
    v8_attack_calls: int = 0
    v8_attack_nodes: int = 0
    v8_attack_budget_exhausted: bool = False
    v8_attack_seconds: float = 0.0
    v8_root_checked: tuple[tuple[Move, str], ...] = ()
    v8_root_rank: int = 0  # tree preference rank of the played move (1 = V7's move)
    v8_root_calls: int = 0
    v8_root_nodes: int = 0
    v8_root_budget_exhausted: bool = False
    v8_root_seconds: float = 0.0
    # Why V8-C changed V7's move: "proven_loss" (V7's move UNSAFE) or "unknown"
    # (aggressive mode only: V7's move unresolved, a lower child proven SAFE).
    v8_root_switch: str = ""
    # H4 policy at the tree root (empty when no policy is used).
    v8_policy_seconds: float = 0.0
    v8_policy_added: tuple[Move, ...] = ()       # policy moves added to the V6 candidates (before tiers)
    v8_policy_added_kept: tuple[Move, ...] = ()  # ... that survived the VCF safety tiers
    v8_policy_added_opened: int = 0              # ... that the tree opened as root children
    v8_policy_displaced: int = 0                 # V6 moves pushed from the first-N opening ranks
    v8_policy_rank: int = 0                      # policy rank of the tree's move among legal moves (1 = top)
    v8_policy_prob: float = 0.0
    v8_root_order_rank: int = 0                  # rank of the tree's move in the root order (1 = first)
    # Tree route: which tree ran and its cost (H5); filled for both tree modes.
    v8_tree_mode: str = ""
    v8_tree_seconds: float = 0.0
    v8_tree_simulations: int = 0
    # H5 PUCT only.
    v8_puct_prior: str = ""
    v8_puct_nn_calls: int = 0
    v8_puct_prior_fallbacks: int = 0
    v8_puct_prior_entropy: float = 0.0           # root prior, normalized (1 = uniform)
    v8_puct_prior_top: Move | None = None        # root child with the highest prior
    v8_puct_prior_top_prob: float = 0.0
    v8_vct2_checked: tuple[tuple[Move, str], ...] = ()  # S3-VCT2: selective depth-2 status per checked move
    v8_vct2_check_nodes: tuple[int, ...] = ()  # nodes of each check above, same order (diagnostics only)
    v8_vct2_switched: bool = False
    v8_vct2_nodes: int = 0
    v8_vct2_seconds: float = 0.0


class _BudgetExhausted(Exception):
    pass


@dataclass
class _BudgetedSolver(ThreatSolver):
    """ThreatSolver with a per-decision budget on uncached VCF calls and total VCF nodes.

    A single call is capped at ``node_limit`` (a cut there is an ordinary UNKNOWN,
    as in ``ThreatSolver``). A call cut short by the remaining total budget raises
    instead, so a budget cut is never cached or mistaken for a finished search.
    """

    call_limit: int = 0
    node_budget: int = 0
    nodes_used: int = 0
    exhausted: bool = field(default=False, repr=False)
    _cap: int = field(default=0, repr=False)  # absolute node cap for the current status_after
    _call_cap: int = field(default=0, repr=False)  # absolute VCF call cap for the current status_after
    last_cut: bool = field(default=False, repr=False)  # last status_after stopped by a budget/share cut

    def vcf(self, game: Game, attacker: int):
        key = (_board_key(game), attacker)
        cached = self._vcf_cache.get(key)
        if cached is not None:
            return cached
        remaining = min(self._cap, self.node_budget) - self.nodes_used
        if self.vcf_calls >= min(self._call_cap, self.call_limit) or remaining < 1:
            self._stop()
        limit = min(self.node_limit, remaining)
        self.vcf_calls += 1
        found, nodes, cut = _find_vcf_with_stats(
            game, attacker, max_fours=_unbounded_max_fours(game), node_limit=limit)
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

    def _stop(self):
        if self.vcf_calls >= self.call_limit or self.nodes_used >= self.node_budget:
            self.exhausted = True
        raise _BudgetExhausted

    def _bounded(self, game: Game, move: Move, share, call_share, check) -> str:
        """Play ``move``, run ``check(game)`` under the shares, undo.

        ``share`` / ``call_share`` cap the VCF nodes / uncached VCF calls this
        call may add (default: the rest of the budget). A cut returns UNKNOWN
        with ``last_cut`` set; finished sub-results stay cached, so a later call
        on the same move resumes cheaply.
        """
        self.last_cut = True
        if self.exhausted:
            return UNKNOWN
        self._cap = self.node_budget if share is None else self.nodes_used + share
        self._call_cap = self.call_limit if call_share is None else self.vcf_calls + call_share
        game.play(*move)
        try:
            status = check(game)
            self.last_cut = False
            return status
        except _BudgetExhausted:
            return UNKNOWN
        finally:
            game.undo()

    def status_after(self, game: Game, move: Move, share: int | None = None,
                     call_share: int | None = None) -> str:
        """V8-A: VCT1 status (SAFE/UNSAFE/UNKNOWN) of ``move`` for ``game.to_play``."""
        return self._bounded(game, move, share, call_share, lambda g: self.after_move(g, 1)[0])

    def attack_status(self, game: Game, move: Move, share: int | None = None,
                      call_share: int | None = None) -> str:
        """V8-B: WIN if every reply to ``move`` loses to our five/VCF, REFUTED if one survives.

        Same meaning as ``tactical_labels.proves_threat``, but stops at the first
        surviving reply (``stop_at_safe``), which does not change the status.
        """
        return self._bounded(game, move, share, call_share, self._attack_check)

    def _attack_check(self, game: Game) -> str:
        if game.done:
            # ``Game.play`` keeps ``to_play`` on the mover after a five: a five by the
            # attacking move is a WIN, a full board is not.
            return WIN if game.winner is not None else REFUTED
        if not game.legal_moves():
            return REFUTED  # no reply to refute, but no win either (full board)
        counts, _ = self.decision(game, 0, stop_at_safe=True)
        return {UNSAFE: WIN, SAFE: REFUTED, UNKNOWN: UNKNOWN}[decision_status(counts)]


def _validate_v8_config(*, stage_vct_safety, vct_vcf_node_limit, vct_call_limit, vct_node_budget,
                        own_vct_attack, attack_vcf_node_limit, attack_call_limit,
                        attack_node_budget, root_vct_safety, root_vcf_node_limit,
                        root_call_limit, root_node_budget, root_max_children=0,
                        root_vct_mode="aggressive", root_policy_order=False, root_policy_extra=0,
                        tree_mode="v5", puct_prior="uniform", puct_c=1.5, root_vct2_check=False,
                        vct2_node_limit=20_000, vct2_call_limit=20_000, vct2_node_budget=10_000,
                        vct2_max_children=4):
    for name, value in (("stage_vct_safety", stage_vct_safety), ("own_vct_attack", own_vct_attack),
                        ("root_vct_safety", root_vct_safety)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool")
    for name, value in (("vct_vcf_node_limit", vct_vcf_node_limit),
                        ("vct_call_limit", vct_call_limit),
                        ("vct_node_budget", vct_node_budget),
                        ("attack_vcf_node_limit", attack_vcf_node_limit),
                        ("attack_call_limit", attack_call_limit),
                        ("attack_node_budget", attack_node_budget),
                        ("root_vcf_node_limit", root_vcf_node_limit),
                        ("root_call_limit", root_call_limit),
                        ("root_node_budget", root_node_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(root_max_children, bool) or not isinstance(root_max_children, int) or root_max_children < 0:
        raise ValueError("root_max_children must be a non-negative integer")
    if root_vct_mode not in ROOT_VCT_MODES:
        raise ValueError(f"root_vct_mode must be one of {ROOT_VCT_MODES}")
    if not isinstance(root_policy_order, bool):
        raise ValueError("root_policy_order must be a bool")
    if isinstance(root_policy_extra, bool) or not isinstance(root_policy_extra, int) or root_policy_extra < 0:
        raise ValueError("root_policy_extra must be a non-negative integer")
    if tree_mode not in TREE_MODES:
        raise ValueError(f"tree_mode must be one of {TREE_MODES}")
    if puct_prior not in PUCT_PRIORS:
        raise ValueError(f"puct_prior must be one of {PUCT_PRIORS}")
    if isinstance(puct_c, bool) or not isinstance(puct_c, (int, float)) or not puct_c > 0:
        raise ValueError("puct_c must be a positive number")
    if not isinstance(root_vct2_check, bool):
        raise ValueError("root_vct2_check must be a bool")
    for name, value in (("vct2_node_limit", vct2_node_limit), ("vct2_call_limit", vct2_call_limit),
                        ("vct2_node_budget", vct2_node_budget), ("vct2_max_children", vct2_max_children)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if tree_mode == "puct" and (root_policy_order or root_policy_extra):
        # H5 keeps the action set fixed across priors (§12.16): no H4 options on PUCT.
        raise ValueError("tree_mode='puct' cannot be combined with root_policy_order / root_policy_extra")


def _stage4_order(game: Game, context: _RootContext, v7_move: Move) -> list[Move]:
    """Stage 4 defenses in V7 preference order, V7's own choice first."""
    player, opponent = game.to_play, -game.to_play
    creators = set(_unstoppable_four_moves(game, opponent))
    defenses = set(creators)
    for window in _threat_windows(game, opponent):
        if creators.intersection(window):
            defenses.update(p for p in window if game.board[p[0]][p[1]] == EMPTY)
    defenses.intersection_update(context.legal)
    rows = []
    for move in defenses:
        with placed(game, player, move):
            rows.append((len(_unstoppable_four_moves(game, opponent)),
                         len(_double_threat_moves(game, opponent)),
                         context.key(game, move), move))
    ordered = [row[-1] for row in sorted(rows)]
    return [v7_move] + [move for move in ordered if move != v7_move]


def _not_immediately_lost(game: Game, move: Move) -> bool:
    """Cheap sanity check for an unproven fallback: no opponent five or unstoppable four after it."""
    player, opponent = game.to_play, -game.to_play
    with placed(game, player, move):
        return not _winning_moves(game, opponent) and not _unstoppable_four_moves(game, opponent)


def _first_safe(game, moves, statuses, solver) -> Move | None:
    """V8-A: the first proven VCT1-SAFE move (see ``_first_proven``)."""
    return _first_proven(game, moves, statuses, solver, solver.status_after, SAFE)


def _first_proven(game, moves, statuses, solver, evaluate, target) -> Move | None:
    """The first move proven ``target`` (see ``_proven``), or None."""
    found = _proven(game, moves, statuses, solver, evaluate, target, want=1)
    return found[0] if found else None


def _proven(game, moves, statuses, solver, evaluate, target, *, want: int) -> list[Move]:
    """Check ``moves`` in order until ``want`` of them are proven ``target``.

    Returns the proven moves in the order they were proven (possibly fewer than
    ``want`` when the budget runs out or the moves run out).

    ``evaluate(game, move, node_share, call_share)`` is the solver's bounded
    status function (V8-A ``status_after`` -> SAFE, V8-B ``attack_status`` ->
    WIN). "First" means first proven: within a round moves are checked in the
    given order, but a later move proven in an earlier round wins.

    Both budgets (VCF nodes and uncached VCF calls) are spent in fair rounds.
    The first target share is half of an equal split and doubles each round,
    but whenever a full equal split can still give every pending move at least
    one node/call, every move gets at least one of each. This avoids the
    ``n <= remaining < 2n`` rounding hole where a zero half-share used to give
    the first move the whole remainder.

    If either budget can no longer give every pending move even one unit, no
    candidate gets the leftover exclusively. Instead every pending move gets
    one final zero-budget structural pass; only checks that finish without a
    new VCF node/call can resolve. The budget is then reported as exhausted
    and the caller falls back among unrefuted moves. Finished sub-results
    remain cached across rounds, and an UNKNOWN that finished at the per-VCF
    node limit is not retried.
    """
    found: list[Move] = []
    pending = [m for m in moves if statuses.get(m, UNKNOWN) == UNKNOWN]
    if not pending:
        return found
    share = (solver.node_budget - solver.nodes_used) // (2 * len(pending)) if pending else 0
    call_share = (solver.call_limit - solver.vcf_calls) // (2 * len(pending)) if pending else 0
    while pending and not solver.exhausted:
        fair_share = (solver.node_budget - solver.nodes_used) // len(pending)
        fair_call_share = (solver.call_limit - solver.vcf_calls) // len(pending)

        if fair_share < 1 or fair_call_share < 1:
            for move in pending:
                if solver.exhausted:
                    break
                statuses[move] = evaluate(game, move, 0, 0)
                if statuses[move] == target:
                    found.append(move)
                    if len(found) >= want:
                        return found
            # The remainder cannot be shared fairly, so it is never spent: report the
            # budget as exhausted (diagnostics, gate statistics, later calls).
            solver.exhausted = True
            break

        share = min(max(share, 1), fair_share)
        call_share = min(max(call_share, 1), fair_call_share)
        cut = []
        for move in pending:
            if solver.exhausted:
                break
            statuses[move] = evaluate(game, move, share, call_share)
            if statuses[move] == target:
                found.append(move)
                if len(found) >= want:
                    return found
                continue
            if solver.last_cut:
                cut.append(move)
        pending = cut
        share *= 2
        call_share *= 2
    return found


def _stage_vct_move(game, context, v7_move, defenses, diag, solver, *,
                    candidate_limit, neighborhood_radius) -> Move:
    """V8-A: first proven VCT1-SAFE forced defense, widening to the root if all lose.

    With no proven SAFE move the fallback never prefers a proven loss:
    V7's move if it is not proven UNSAFE, otherwise the first unrefuted
    (UNKNOWN or unchecked) forced defense, then the first unrefuted (UNKNOWN or
    never checked) root candidate, each passing ``_not_immediately_lost``.
    Only when all of those are proven UNSAFE (or fail the check) does V7's
    choice stand. The root is built whenever every forced defense is proven
    UNSAFE, even if the budget ran out, so it can still serve as the fallback.
    """
    statuses: dict[Move, str] = {}
    root: list[Move] = []
    chosen = _first_safe(game, defenses, statuses, solver)
    if chosen is None and all(statuses.get(m) == UNSAFE for m in defenses):
        diag.v8_vct_widened = True
        root, _, _ = _root_candidates_v6(game, context, candidate_limit, neighborhood_radius)
        root = [m for m in root if m not in statuses]
        chosen = _first_safe(game, root, statuses, solver)
    if chosen is None:
        if statuses.get(v7_move, UNKNOWN) != UNSAFE:
            chosen = v7_move
        else:
            fallback = [m for m in defenses if statuses.get(m, UNKNOWN) == UNKNOWN]
            fallback += [m for m in root if statuses.get(m, UNKNOWN) == UNKNOWN]
            chosen = next((m for m in fallback if _not_immediately_lost(game, m)), v7_move)
    diag.v8_vct_checked = tuple(statuses.items())
    return chosen


def _attack_candidates(game: Game, context: _RootContext) -> list[Move]:
    """V8-B candidates: legal moves making a four or an open three (speed heuristic, incomplete).

    Ordered by the V3.2.1 priority score (highest first), then the V5 root key.
    ``context.legal`` excludes black forbidden points.
    """
    player = game.to_play
    moves = []
    for move in context.legal:
        features = _fast_pattern_features_for_move(game, player, move, assume_legal=True)
        if features.four_directions > 0 or features.open_three_directions > 0:
            moves.append(move)
    return sorted(moves, key=lambda m: (-_v321_priority_score(game, m), context.key(game, m)))


def _own_vct_attack(game, context, diag, *, node_limit, call_limit, node_budget) -> Move | None:
    """V8-B: the first proven depth-1 VCT attack, or None (never an unproven one)."""
    started = perf_counter()
    candidates = _attack_candidates(game, context)
    diag.v8_attack_candidates = len(candidates)
    solver = _BudgetedSolver(node_limit=node_limit, call_limit=call_limit, node_budget=node_budget)
    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, solver.attack_status, WIN)
    diag.v8_attack_checked = tuple(statuses.items())
    diag.v8_attack_calls = solver.vcf_calls
    diag.v8_attack_nodes = solver.nodes_used
    diag.v8_attack_budget_exhausted = solver.exhausted
    diag.v8_attack_seconds = perf_counter() - started
    if chosen is not None:
        diag.v8_attack_status = WIN
        diag.v8_attack_move = chosen
        diag.v8_attack_rank = candidates.index(chosen) + 1
    return chosen


def _verify_root_choice(game, chosen, children, diag, *, node_limit, call_limit, node_budget,
                        max_children=0, mode="aggressive") -> Move:
    """V8-C: keep V7's tree move if it is proven VCT1-SAFE, otherwise the first proven SAFE child.

    Children are tried in the tree's own preference order (visits, mean value).
    V7's move first gets half of the budget on its own, so in the common case
    (it is SAFE) nothing else is checked and V8 plays exactly V7's move. Then
    the top ``max_children`` children (all when 0), V7's move included while
    still undecided, share the rest fairly (``_first_proven``); if every one of
    them is proven UNSAFE the check widens to the remaining children. With no
    proven SAFE child the fallback never prefers a proven loss: V7's move unless
    it is proven UNSAFE, else the first unrefuted (UNKNOWN or never checked)
    child that does not lose at once.

    ``mode="veto"`` keeps V7's move unless it is proven UNSAFE (an UNKNOWN never
    switches). When it is proven UNSAFE, the children are checked exactly as in
    aggressive mode (same fair shares, same widening, same budget), but the
    played move is the best-ranked child that is not proven UNSAFE (UNKNOWN or
    never checked included) and does not lose at once, not the proven SAFE one;
    V7's move stands if every child is proven UNSAFE. The two modes therefore
    differ only in the selection rule, never in how the budget is spent (§12.8).
    """
    started = perf_counter()
    solver = _BudgetedSolver(node_limit=node_limit, call_limit=call_limit, node_budget=node_budget)
    order = _root_order(chosen, children)
    statuses = {chosen: solver.status_after(game, chosen, node_budget // 2, call_limit // 2)}
    if statuses[chosen] == SAFE or (mode == "veto" and statuses[chosen] != UNSAFE):
        return _record_root(diag, solver, statuses, order, chosen, chosen, started)
    final = _search_root_children(game, order, statuses, solver, max_children)
    if mode == "veto":
        final = next((m for m in order[1:] if statuses.get(m, UNKNOWN) != UNSAFE
                      and _not_immediately_lost(game, m)), chosen)
    elif final is None:
        if statuses[chosen] != UNSAFE:
            final = chosen
        else:
            final = next((m for m in order[1:] if statuses.get(m, UNKNOWN) == UNKNOWN
                          and _not_immediately_lost(game, m)), chosen)
    return _record_root(diag, solver, statuses, order, chosen, final, started)


def _search_root_children(game, order, statuses, solver, max_children) -> Move | None:
    """First proven SAFE child: the top ``max_children`` share the budget fairly, widening if all are UNSAFE."""
    head = order[:max_children] if max_children else order
    final = _first_proven(game, head, statuses, solver, solver.status_after, SAFE)
    if final is None and len(head) < len(order) and all(statuses.get(m) == UNSAFE for m in head):
        final = _first_proven(game, order[len(head):], statuses, solver, solver.status_after, SAFE)
    return final


def _root_order(chosen, children) -> list[Move]:
    return [chosen] + [c.move for c in sorted(children, key=lambda c: (-c.visits, -c.mean_value, c.move))
                       if c.move != chosen]


def _record_root(diag, solver, statuses, order, chosen, final, started) -> Move:
    diag.v8_root_checked = tuple(statuses.items())
    diag.v8_root_rank = order.index(final) + 1
    diag.v8_root_calls = solver.vcf_calls
    diag.v8_root_nodes = solver.nodes_used
    diag.v8_root_budget_exhausted = solver.exhausted
    diag.v8_root_seconds = perf_counter() - started
    if final != chosen:
        diag.v8_root_switch = "proven_loss" if statuses[chosen] == UNSAFE else "unknown"
    return final


def _vct2_veto(game, chosen, children, diag, *, budget, max_children) -> Move:
    """S3-VCT2: keep ``chosen`` unless a selective depth-2 search proves it lost.

    Each check is a fresh ``SelectiveSolver`` with ``budget`` (attacker: threat moves only,
    defender: every reply), so a PROVEN loss is a proof and anything else (no win found,
    budget cut) is not. On a proven loss the next children in tree order (visits, mean
    value), up to ``max_children`` in all, are checked; the first one that is not proven
    lost, not refuted by V8-C (UNSAFE in ``v8_root_checked``) and not lost at once is
    played. If none qualifies, ``chosen`` stands (as V8-C does when every child is lost).
    """
    from .selective_vct import SelectiveSolver  # selective_vct imports this module

    started = perf_counter()
    root_status = dict(diag.v8_root_checked)
    statuses, nodes, per_check = {}, 0, []

    def proven_lost(move) -> bool:
        nonlocal nodes
        solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                                 node_budget=budget['node_budget'])
        status = solver._bounded(game, move, None, None, lambda g: solver.after_move(g, 2)[0])
        nodes += solver.nodes_used
        per_check.append(solver.nodes_used)
        statuses[move] = status
        return status == UNSAFE

    final = chosen
    if proven_lost(chosen):
        for move in _root_order(chosen, children)[1:max_children]:
            if root_status.get(move) == UNSAFE or not _not_immediately_lost(game, move):
                continue
            if not proven_lost(move):
                final = move
                break
    diag.v8_vct2_checked = tuple(statuses.items())
    diag.v8_vct2_check_nodes = tuple(per_check)
    diag.v8_vct2_switched = final != chosen
    diag.v8_vct2_nodes = nodes
    diag.v8_vct2_seconds = perf_counter() - started
    return final


def root_opening_count(simulations: int, initial_width: int, total: int) -> int:
    """Root children the V5 tree opens in ``simulations`` (root expansion has priority)."""
    opened = 0
    for visits in range(simulations):
        if opened < min(total, initial_width + int(sqrt(visits))):
            opened += 1
    return opened


def prepare_root_moves(game, context, diag, *, candidate_limit, neighborhood_radius,
                       safety_vcf_max_fours, safety_vcf_node_limit, safety_precheck_node_limit,
                       safety_total_node_limit, self_forbidden_min_white,
                       policy_scores: dict | None = None, order_by_policy=False, extra=0):
    """The tree's root move list: V6 candidates -> V7 VCF safety tiers -> M3 penalty.

    H4 (§12.13), only when ``policy_scores`` is given:
    - ``extra``: up to ``extra`` of the policy's top legal moves that are not V6 candidates
      are appended BEFORE the safety tiers, so they get exactly the same VCF check
      (a move the opponent's VCF refutes is dropped like any other);
    - ``order_by_policy``: inside each safety tier, moves are sorted by policy probability,
      ties by the V7 order. Tiers are never mixed: the policy cannot lift an unverified move
      above a verified one.
    Returns (moves, score, reasons, added) with ``added`` = policy-added moves.
    """
    moves, score, reasons = _root_candidates_v6(game, context, candidate_limit, neighborhood_radius)
    added: list[Move] = []
    if policy_scores is not None and extra:
        present, legal = set(moves), set(context.legal)
        ranked = sorted((m for m in policy_scores if m in legal), key=lambda m: (-policy_scores[m], m))
        added = [m for m in ranked[:extra] if m not in present]
        moves = list(moves) + added
    tiers = _vcf_safety_tiers(
        game, context, moves, diag,
        max_fours=safety_vcf_max_fours, node_limit=safety_vcf_node_limit,
        precheck_node_limit=safety_precheck_node_limit, total_node_limit=safety_total_node_limit,
    )
    ordered = _penalize_within_tiers(game, tiers, diag, minimum_white=self_forbidden_min_white)
    if policy_scores is not None and order_by_policy:
        tier_of = {m: i for i, tier in enumerate(tiers) for m in tier}
        position = {m: i for i, m in enumerate(ordered)}
        ordered = sorted(ordered, key=lambda m: (tier_of[m], -policy_scores.get(m, 0.0), position[m]))
    return ordered, score, reasons, added


def _search_tree_v8(game, root_moves, simulations, exploration, candidate_limit,
                    initial_width, neighborhood_radius, priority_top_k, random):
    """``search.mcts_v5._search_v5_tree`` with the root children returned.

    The loop and every random call are identical, so the chosen move is the
    same as the frozen function's for the same random state.
    """
    random = random or Random()
    root = MCTSNode(parent=None, move=None, player_just_moved=None, untried_moves=root_moves[:])
    state = deepcopy(game)
    root_history_length = len(state.history)
    for _ in range(simulations):
        node = root
        while not state.done and not _can_expand(node, initial_width) and node.children:
            node = _select_child(node, exploration)
            state.play(*node.move)
        if not state.done and _can_expand(node, initial_width):
            move = _pop_ranked_untried(node, priority_top_k, random)
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node, move=move, player_just_moved=player,
                untried_moves=([] if state.done else
                               _search_candidates_v321(state, candidate_limit, neighborhood_radius)),
            )
            node.children.append(child)
            node = child
        winner = (state.winner if state.done else
                  _rollout_v321(state, random, candidate_limit, neighborhood_radius, priority_top_k))
        _backpropagate(node, winner)
        while len(state.history) > root_history_length:
            state.undo()
    max_visits = max(child.visits for child in root.children)
    candidates = [child for child in root.children if child.visits == max_visits]
    max_value = max(child.mean_value for child in candidates)
    candidates = [child for child in candidates if child.mean_value == max_value]
    chosen = random.choice(candidates)
    return chosen.move, root.children


def mcts_search_v8(
    game: Game, *, simulations=50, tactical_simulations=100,
    tactical_score_threshold=1800, exploration=sqrt(2), candidate_limit=20,
    initial_width=8, neighborhood_radius=2, priority_top_k=8,
    own_vcf_max_fours=10, own_vcf_node_limit=5000,
    safety_vcf_max_fours=10, safety_vcf_node_limit=4000,
    safety_precheck_node_limit=8000, safety_total_node_limit=8000,
    self_forbidden_min_white=3,
    stage_vct_safety=True, vct_vcf_node_limit=20_000, vct_call_limit=3_000,
    vct_node_budget=200_000,
    own_vct_attack=True, attack_vcf_node_limit=20_000, attack_call_limit=10_000,
    attack_node_budget=200_000,
    root_vct_safety=True, root_vcf_node_limit=20_000, root_call_limit=10_000,
    root_node_budget=400_000, root_max_children=4, root_vct_mode="aggressive",
    root_policy_order=False, root_policy_extra=0, tree_mode="v5", puct_prior="uniform", puct_c=1.5,
    root_vct2_check=False, vct2_node_limit=20_000, vct2_call_limit=20_000, vct2_node_budget=10_000,
    vct2_max_children=4,
    root_policy=None, random: Random | None = None, diagnostics: SearchDiagnostics | None = None,
) -> Move:
    """V7 decision flow (``search.mcts_v7.mcts_search_v7``) with the V8 modules."""
    _validate_v5_config(
        simulations=simulations, tactical_simulations=tactical_simulations,
        tactical_score_threshold=tactical_score_threshold, exploration=exploration,
        candidate_limit=candidate_limit, initial_width=initial_width,
        neighborhood_radius=neighborhood_radius, priority_top_k=priority_top_k,
    )
    _validate_v7_config(
        own_vcf_max_fours=own_vcf_max_fours, own_vcf_node_limit=own_vcf_node_limit,
        safety_vcf_max_fours=safety_vcf_max_fours, safety_vcf_node_limit=safety_vcf_node_limit,
        safety_precheck_node_limit=safety_precheck_node_limit,
        safety_total_node_limit=safety_total_node_limit,
        self_forbidden_min_white=self_forbidden_min_white,
    )
    _validate_v8_config(stage_vct_safety=stage_vct_safety, vct_vcf_node_limit=vct_vcf_node_limit,
                        vct_call_limit=vct_call_limit, vct_node_budget=vct_node_budget,
                        own_vct_attack=own_vct_attack, attack_vcf_node_limit=attack_vcf_node_limit,
                        attack_call_limit=attack_call_limit, attack_node_budget=attack_node_budget,
                        root_vct_safety=root_vct_safety, root_vcf_node_limit=root_vcf_node_limit,
                        root_call_limit=root_call_limit, root_node_budget=root_node_budget,
                        root_max_children=root_max_children, root_vct_mode=root_vct_mode,
                        root_policy_order=root_policy_order, root_policy_extra=root_policy_extra,
                        tree_mode=tree_mode, puct_prior=puct_prior, puct_c=puct_c,
                        root_vct2_check=root_vct2_check, vct2_node_limit=vct2_node_limit,
                        vct2_call_limit=vct2_call_limit, vct2_node_budget=vct2_node_budget,
                        vct2_max_children=vct2_max_children)
    uses_policy = root_policy_order or root_policy_extra > 0
    if (uses_policy or (tree_mode == "puct" and puct_prior == "policy")) and root_policy is None:
        # Fail fast: a policy arm must never silently run as the baseline.
        raise ValueError("root_policy_order / root_policy_extra / puct_prior='policy' need a root_policy callable")
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.__dict__.update(vars(SearchDiagnostics()))
    context = _RootContext(game.legal_moves(), diag)

    forced = _forced_v5_move(game, context=context)
    forced_stage = diag.forced_policy_stage
    if forced is not None and forced_stage in (1, 2, 3):
        diag.v8_route = f"stage{forced_stage}"
        diag.v8_v7_move = forced
        return forced

    module_started = perf_counter()
    own_vcf, own_nodes, _ = _find_vcf_with_stats(
        game, game.to_play, max_fours=own_vcf_max_fours, node_limit=own_vcf_node_limit,
    )
    diag.v7_own_vcf_nodes = own_nodes
    if own_vcf is not None:
        diag.v7_own_vcf_found = True
        diag.v7_own_vcf_length = own_vcf.fours
        diag.forced_policy_stage = None
        diag.v7_module_seconds = perf_counter() - module_started
        diag.v8_route = "own_vcf"
        diag.v8_v7_move = own_vcf.first_move
        return own_vcf.first_move

    if own_vct_attack:
        attack = _own_vct_attack(game, context, diag, node_limit=attack_vcf_node_limit,
                                 call_limit=attack_call_limit, node_budget=attack_node_budget)
        if attack is not None:
            # V7's own move is not computed here (it would need the full V7 search);
            # v8_v7_move stays None and the move counts as changed.
            diag.forced_policy_stage = None
            diag.v7_module_seconds = perf_counter() - module_started
            diag.v8_route = "own_vct"
            diag.v8_changed = True
            return attack

    if forced is not None:
        if forced_stage == 4:
            forced = _stage4_v7_move(
                game, context, forced, diag,
                max_fours=safety_vcf_max_fours, node_limit=safety_vcf_node_limit,
                total_node_limit=safety_total_node_limit,
            )
        diag.v7_module_seconds = perf_counter() - module_started
        diag.v8_route = f"stage{forced_stage}"
        diag.v8_v7_move = forced
        if not stage_vct_safety:
            return forced
        started = perf_counter()
        solver = _BudgetedSolver(node_limit=vct_vcf_node_limit, call_limit=vct_call_limit,
                                 node_budget=vct_node_budget)
        order = _stage4_order(game, context, forced) if forced_stage == 4 else [forced]
        chosen = _stage_vct_move(game, context, forced, order, diag, solver,
                                 candidate_limit=candidate_limit,
                                 neighborhood_radius=neighborhood_radius)
        diag.v8_vct_calls = solver.vcf_calls
        diag.v8_vct_nodes = solver.nodes_used
        diag.v8_vct_budget_exhausted = solver.exhausted
        diag.v8_vct_seconds = perf_counter() - started
        diag.v8_changed = chosen != forced
        return chosen

    policy_scores = None
    if uses_policy:
        policy_started = perf_counter()
        policy_scores = dict(root_policy(game))
        diag.v8_policy_seconds = perf_counter() - policy_started
    safety = dict(candidate_limit=candidate_limit, neighborhood_radius=neighborhood_radius,
                  safety_vcf_max_fours=safety_vcf_max_fours, safety_vcf_node_limit=safety_vcf_node_limit,
                  safety_precheck_node_limit=safety_precheck_node_limit,
                  safety_total_node_limit=safety_total_node_limit,
                  self_forbidden_min_white=self_forbidden_min_white)
    moves, score, reasons, added = prepare_root_moves(
        game, context, diag, **safety, policy_scores=policy_scores,
        order_by_policy=root_policy_order, extra=root_policy_extra)
    if uses_policy:
        baseline, _, _, _ = prepare_root_moves(game, context, SearchDiagnostics(), **safety)
        n_open = root_opening_count(simulations, initial_width, len(baseline))
        diag.v8_policy_added = tuple(added)
        diag.v8_policy_added_kept = tuple(m for m in added if m in moves)
        diag.v8_policy_displaced = len(set(baseline[:n_open]) - set(moves[:n_open]))
    diag.v7_module_seconds = perf_counter() - module_started

    tactical = tactical_score_threshold is not None and score >= tactical_score_threshold
    budget = tactical_simulations if tactical else simulations
    diag.best_root_tactical_score = score
    diag.selected_simulations = budget
    diag.simulation_mode = "tactical" if tactical else "normal"
    diag.root_candidates = tuple(moves)
    tree_started = perf_counter()
    diag.v8_tree_mode = tree_mode
    diag.v8_tree_simulations = budget
    if tree_mode == "puct":
        stats = PUCTStats()
        chosen, children = search_tree_puct(
            game, moves, budget, c_puct=puct_c, prior=puct_prior,
            policy=root_policy if puct_prior == "policy" else None,
            candidate_limit=candidate_limit, neighborhood_radius=neighborhood_radius,
            priority_top_k=priority_top_k, random=random, stats=stats,
        )
        top = max(moves, key=lambda m: (stats.root_prior[m], -moves.index(m)))
        diag.v8_puct_prior = puct_prior
        diag.v8_puct_nn_calls = stats.nn_calls
        diag.v8_puct_prior_fallbacks = stats.prior_fallbacks
        diag.v8_puct_prior_entropy = prior_entropy([stats.root_prior[m] for m in moves])
        diag.v8_puct_prior_top = top
        diag.v8_puct_prior_top_prob = stats.root_prior[top]
    else:
        chosen, children = _search_tree_v8(
            game, moves, budget, exploration, candidate_limit,
            initial_width, neighborhood_radius, priority_top_k, random,
        )
    diag.v8_tree_seconds = perf_counter() - tree_started
    v7_move = chosen
    if uses_policy:
        opened = {c.move for c in children}
        diag.v8_policy_added_opened = sum(m in opened for m in diag.v8_policy_added_kept)
        ranked = sorted(policy_scores, key=lambda m: (-policy_scores[m], m))
        diag.v8_policy_rank = ranked.index(chosen) + 1 if chosen in policy_scores else 0
        diag.v8_policy_prob = float(policy_scores.get(chosen, 0.0))
        diag.v8_root_order_rank = moves.index(chosen) + 1
    if root_vct_safety:
        chosen = _verify_root_choice(game, chosen, children, diag, node_limit=root_vcf_node_limit,
                                     call_limit=root_call_limit, node_budget=root_node_budget,
                                     max_children=root_max_children, mode=root_vct_mode)
    if root_vct2_check:
        chosen = _vct2_veto(game, chosen, children, diag,
                            budget={'node_limit': vct2_node_limit, 'call_limit': vct2_call_limit,
                                    'node_budget': vct2_node_budget},
                            max_children=vct2_max_children)
    selected = sorted(reasons.get(chosen, ()), key=lambda k: (-PRIORITY[k], k))
    diag.v6_selected_reasons = tuple(selected)
    diag.v6_selected_threat_type = selected[0] if selected else None
    diag.v8_route = "tree"
    # Without a policy: same root and random stream as V7, so this is V7's move.
    # With a policy (H4): the tree's move on the policy-ordered root (before V8-C).
    diag.v8_v7_move = v7_move
    diag.v8_changed = chosen != v7_move
    diag.v8_root_visits = tuple((c.move, c.visits, c.mean_value) for c in children)
    return chosen
