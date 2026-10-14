"""TB-RT30 move search: the TB-Engine-v1 flow under one absolute deadline (docs/web-ser5.md §3.3, §3.4).

``rt30_search`` is ``analysis.mcts_v8.mcts_search_v8`` for the TB-Engine-v1 configuration
(PUCT tree, no H4 root-policy options), with these differences only:

- V8-B, V8-A, V8-C and VCT2 use the deadline solvers (``rt30.solver``); the frozen helpers that
  build their own solver (``_own_vct_attack``, ``_verify_root_choice``, ``_vct2_veto``) are copied
  here with the solver injected, everything else is called as frozen;
- every phase shares the move's absolute deadline, minus what the later phases reserve
  (``RT30_TIMING``); V8-B is also capped at ``attack_seconds`` from its start;
- the PUCT tree has no deadline inside (it stays deterministic) but is entered only when the
  remaining time covers ``puct_reserve``; VCT2 is skipped below ``vct2_min``;
- an emergency move is made before any search and ``progress(stage, move)`` reports every new
  best-so-far move, so a supervisor always holds a playable move.

With ``deadline=None`` nothing is cut or skipped and the moves and diagnostics are those of
TB-Engine-v1 (``scripts/run_rt30_replay.py`` arm ``rt``). The V7 modules (own VCF 5k nodes,
VCF safety 8k) run without a deadline; their time is inside the reserves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from analysis.mcts_v8 import (
    WIN, SearchDiagnostics, _attack_candidates, _first_proven, _not_immediately_lost, _record_root,
    _root_order, _search_root_children, _stage4_order, _stage_vct_move, _validate_v8_config,
    _V8_KEYS, V8_DEFAULTS, prepare_root_moves,
)
from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.puct_v8 import PUCTStats, prior_entropy, search_tree_puct
from analysis.threats import SAFE, UNKNOWN, UNSAFE, ordered_moves
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import PRIORITY
from search.mcts_v7 import _find_vcf_with_stats, _stage4_v7_move

from .deadline import Deadline
from .solver import DeadlineSelectiveSolver, DeadlineSolver

Move = tuple[int, int]
ATTACK_ORDERS = ('frozen', 'threes_first')
ATTACK_SCHEDULERS = ('fair', 'widen', 'widen3')
RT30_TIMING = {
    'final_margin': 1.5,    # kept free at the end of every phase: response, IPC, clock jitter
    'attack_seconds': 4.0,  # V8-B cap from its own start
    'puct_reserve': 6.0,    # PUCT is entered only with this much left (set from the mini-PC p99)
    'vct2_reserve': 4.0,    # V8-C leaves this for VCT2
    'vct2_min': 2.0,        # VCT2 runs only with this much left
}


@dataclass
class RTInfo:
    """What the deadline did on one move (kept apart from the frozen diagnostics)."""
    deadline: bool = False
    remaining_at_start: float | None = None
    emergency: Move | None = None
    timeouts: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    best: list[tuple[str, Move]] = field(default_factory=list)
    seconds: float = 0.0


def limited(deadline, *, reserve: float = 0.0, cap: float | None = None):
    """``deadline`` moved ``reserve`` seconds earlier and at most ``cap`` seconds from now.

    Only a ``Deadline`` is moved; ``None`` stays None and test clocks (``CheckLimit``) are shared as is.
    """
    if not isinstance(deadline, Deadline):
        return deadline
    at = deadline.at - reserve
    if cap is not None:
        at = min(at, deadline.clock() + cap)
    return Deadline(at, deadline.clock)


def remaining(deadline) -> float:
    return float('inf') if deadline is None else deadline.remaining()


def emergency_move(game, context, policy=None) -> Move:
    """A playable move before any search: the policy's best (nearest-the-stones order without a
    policy) among the first 20 candidates that do not lose at once; else the first legal one."""
    near = [m for m in ordered_moves(game) if m in set(context.legal)] or list(context.legal)
    scores = policy(game) if policy is not None else {}
    ranked = sorted(near, key=lambda m: -scores.get(m, 0.0))  # stable: ties keep the distance order
    return next((m for m in ranked[:20] if _not_immediately_lost(game, m)), ranked[0])


def attack_order(game, candidates: list[Move], order: str) -> list[Move]:
    """V8-B candidate order. ``frozen``: V8's own (V3.2.1 priority). ``threes_first``: moves making
    an open three first, V8's order inside each group (in the E3-A records the VCT1 winning move is an
    open-three move far more often than a four: own VCF has already looked for the four wins)."""
    if order == 'frozen':
        return candidates
    if order == 'threes_first':
        from search.mcts_v321 import _fast_pattern_features_for_move

        player = game.to_play
        threes = {m: _fast_pattern_features_for_move(game, player, m, assume_legal=True).open_three_directions
                  for m in candidates}
        return sorted(candidates, key=lambda m: (-threes[m], candidates.index(m)))
    raise ValueError(f'attack order must be one of {ATTACK_ORDERS}')


def _widening_proven(game, candidates, statuses, solver, evaluate, target, stages=None):
    """Progressive widening: the fair scheduler on growing heads of the order.

    ``stages`` is [(head size, share of what is left)], the last stage taking all candidates and the
    rest; default 1, 2, 4, ... with half of what is left each. The share is of the remaining *time*
    when the solver has a deadline (under a cap a node slice would outlast it), else of the remaining
    nodes and calls. A stage cut by its slice leaves the solver usable for the next stage; finished
    sub-results stay cached, so a later stage resumes earlier ones cheaply.
    """
    if stages is None:
        stages, k = [], 1
        while k < len(candidates):
            stages.append((k, 0.5))
            k *= 2
    stages = [(min(size, len(candidates)), share) for size, share in stages if size < len(candidates)]
    stages.append((len(candidates), 1.0))
    outer = solver.deadline
    for index, (size, share) in enumerate(stages):
        last = index == len(stages) - 1
        saved = (solver.node_budget, solver.call_limit)
        if not last:
            if isinstance(outer, Deadline):
                now = outer.clock()
                solver.deadline = Deadline(min(outer.at, now + (outer.at - now) * share), outer.clock)
            else:
                solver.node_budget = solver.nodes_used + int((saved[0] - solver.nodes_used) * share)
                solver.call_limit = solver.vcf_calls + int((saved[1] - solver.vcf_calls) * share)
        try:
            found = _first_proven(game, candidates[:size], statuses, solver, evaluate, target)
        finally:
            solver.deadline = outer
            solver.node_budget, solver.call_limit = saved
        if found is not None:
            return found
        if not last:
            outer_done = outer is not None and isinstance(outer, Deadline) and outer.expired()
            if solver.timed_out and not outer_done:
                solver.timed_out = False  # only this stage's time slice ran out
            if solver.timed_out:
                return None
            if solver.nodes_used < solver.node_budget and solver.vcf_calls < solver.call_limit:
                solver.exhausted = False  # only this stage's node slice was used up
            if solver.exhausted:
                return None
    return None


ATTACK_STAGES = {'widen': None, 'widen3': [(1, 0.4), (3, 0.6)]}


def _own_vct_attack(game, context, diag, *, node_limit, call_limit, node_budget, deadline,
                    order: str = 'frozen', scheduler: str = 'fair'):
    """``analysis.mcts_v8._own_vct_attack`` with a deadline solver (and, opt-in, another candidate
    order or scheduler; ``frozen`` / ``fair`` are V8's own)."""
    if scheduler not in ATTACK_SCHEDULERS:
        raise ValueError(f'attack scheduler must be one of {ATTACK_SCHEDULERS}')
    started = perf_counter()
    candidates = attack_order(game, _attack_candidates(game, context), order)
    diag.v8_attack_candidates = len(candidates)
    solver = DeadlineSolver(node_limit=node_limit, call_limit=call_limit, node_budget=node_budget, deadline=deadline)
    statuses: dict[Move, str] = {}
    if scheduler == 'fair':
        chosen = _first_proven(game, candidates, statuses, solver, solver.attack_status, WIN)
    else:
        chosen = _widening_proven(game, candidates, statuses, solver, solver.attack_status, WIN,
                                  stages=ATTACK_STAGES[scheduler])
    diag.v8_attack_checked = tuple(statuses.items())
    diag.v8_attack_calls = solver.vcf_calls
    diag.v8_attack_nodes = solver.nodes_used
    diag.v8_attack_budget_exhausted = solver.exhausted
    diag.v8_attack_seconds = perf_counter() - started
    if chosen is not None:
        diag.v8_attack_status = WIN
        diag.v8_attack_move = chosen
        diag.v8_attack_rank = candidates.index(chosen) + 1
    return chosen, solver.timed_out


def _verify_root_choice(game, chosen, children, diag, *, node_limit, call_limit, node_budget,
                        max_children, mode, deadline):
    """``analysis.mcts_v8._verify_root_choice`` with a deadline solver."""
    started = perf_counter()
    solver = DeadlineSolver(node_limit=node_limit, call_limit=call_limit, node_budget=node_budget, deadline=deadline)
    order = _root_order(chosen, children)
    statuses = {chosen: solver.status_after(game, chosen, node_budget // 2, call_limit // 2)}
    if statuses[chosen] == SAFE or (mode == "veto" and statuses[chosen] != UNSAFE):
        return _record_root(diag, solver, statuses, order, chosen, chosen, started), solver.timed_out
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
    return _record_root(diag, solver, statuses, order, chosen, final, started), solver.timed_out


def _vct2_veto(game, chosen, children, diag, *, budget, max_children, deadline):
    """``analysis.mcts_v8._vct2_veto`` with deadline solvers (each check still a fresh solver)."""
    started = perf_counter()
    root_status = dict(diag.v8_root_checked)
    statuses, nodes, per_check = {}, 0, []
    timed_out = False

    def proven_lost(move) -> bool:
        nonlocal nodes, timed_out
        solver = DeadlineSelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                                         node_budget=budget['node_budget'], deadline=deadline)
        status = solver._bounded(game, move, None, None, lambda g: solver.after_move(g, 2)[0])
        timed_out = timed_out or solver.timed_out
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
    return final, timed_out


def check_config(config: dict) -> None:
    """TB-RT30 runs the TB-Engine-v1 flow only: PUCT tree, no H4 root-policy options."""
    if config['tree_mode'] != 'puct' or config['root_policy_order'] or config['root_policy_extra']:
        raise ValueError("rt30_search supports tree_mode='puct' without root_policy_order / root_policy_extra")
    _validate_v8_config(**{key: config[key] for key in _V8_KEYS})


def rt30_search(game, *, config: dict, root_policy=None, random=None, diagnostics=None, deadline=None,
                timing: dict | None = None, progress=None, info: RTInfo | None = None,
                attack_order: str = 'frozen', attack_scheduler: str = 'fair') -> Move:
    c = {**V8_DEFAULTS, **config}
    check_config(c)
    if c['puct_prior'] == 'policy' and root_policy is None:
        raise ValueError("puct_prior='policy' needs a root_policy callable")
    t = {**RT30_TIMING, **(timing or {})}
    info = info if info is not None else RTInfo()
    info.__dict__.update(vars(RTInfo()))
    info.deadline = deadline is not None
    info.remaining_at_start = None if deadline is None else round(deadline.remaining(), 3)
    move_started = perf_counter()

    def best(stage: str, move: Move) -> Move:
        info.best.append((stage, tuple(move)))
        if progress is not None:
            progress(stage, tuple(move))
        return move

    def done(move: Move) -> Move:
        info.seconds = perf_counter() - move_started
        return move

    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.__dict__.update(vars(SearchDiagnostics()))
    context = _RootContext(game.legal_moves(), diag)

    forced = _forced_v5_move(game, context=context)
    forced_stage = diag.forced_policy_stage
    if forced is not None and forced_stage in (1, 2, 3):
        diag.v8_route = f"stage{forced_stage}"
        diag.v8_v7_move = forced
        return done(best(f'stage{forced_stage}', forced))

    info.emergency = emergency_move(game, context, root_policy if c['puct_prior'] == 'policy' else None)
    best('emergency', info.emergency)

    module_started = perf_counter()
    own_vcf, own_nodes, _ = _find_vcf_with_stats(
        game, game.to_play, max_fours=c['own_vcf_max_fours'], node_limit=c['own_vcf_node_limit'])
    diag.v7_own_vcf_nodes = own_nodes
    if own_vcf is not None:
        diag.v7_own_vcf_found = True
        diag.v7_own_vcf_length = own_vcf.fours
        diag.forced_policy_stage = None
        diag.v7_module_seconds = perf_counter() - module_started
        diag.v8_route = "own_vcf"
        diag.v8_v7_move = own_vcf.first_move
        return done(best('own_vcf', own_vcf.first_move))

    if c['own_vct_attack']:
        attack, cut = _own_vct_attack(
            game, context, diag, node_limit=c['attack_vcf_node_limit'], call_limit=c['attack_call_limit'],
            node_budget=c['attack_node_budget'],
            deadline=limited(deadline, reserve=t['final_margin'], cap=t['attack_seconds']),
            order=attack_order, scheduler=attack_scheduler)
        if cut:
            info.timeouts.append('attack')
        if attack is not None:
            diag.forced_policy_stage = None
            diag.v7_module_seconds = perf_counter() - module_started
            diag.v8_route = "own_vct"
            diag.v8_changed = True
            return done(best('own_vct', attack))

    if forced is not None:
        if forced_stage == 4:
            forced = _stage4_v7_move(
                game, context, forced, diag, max_fours=c['safety_vcf_max_fours'],
                node_limit=c['safety_vcf_node_limit'], total_node_limit=c['safety_total_node_limit'])
        diag.v7_module_seconds = perf_counter() - module_started
        diag.v8_route = f"stage{forced_stage}"
        diag.v8_v7_move = forced
        best(f'stage{forced_stage}_v7', forced)
        if not c['stage_vct_safety']:
            return done(forced)
        started = perf_counter()
        solver = DeadlineSolver(node_limit=c['vct_vcf_node_limit'], call_limit=c['vct_call_limit'],
                                node_budget=c['vct_node_budget'], deadline=limited(deadline, reserve=t['final_margin']))
        order = _stage4_order(game, context, forced) if forced_stage == 4 else [forced]
        chosen = _stage_vct_move(game, context, forced, order, diag, solver,
                                 candidate_limit=c['candidate_limit'], neighborhood_radius=c['neighborhood_radius'])
        if solver.timed_out:
            info.timeouts.append('stage_vct')
        diag.v8_vct_calls = solver.vcf_calls
        diag.v8_vct_nodes = solver.nodes_used
        diag.v8_vct_budget_exhausted = solver.exhausted
        diag.v8_vct_seconds = perf_counter() - started
        diag.v8_changed = chosen != forced
        return done(best('stage_vct', chosen))

    safety = dict(candidate_limit=c['candidate_limit'], neighborhood_radius=c['neighborhood_radius'],
                  safety_vcf_max_fours=c['safety_vcf_max_fours'], safety_vcf_node_limit=c['safety_vcf_node_limit'],
                  safety_precheck_node_limit=c['safety_precheck_node_limit'],
                  safety_total_node_limit=c['safety_total_node_limit'],
                  self_forbidden_min_white=c['self_forbidden_min_white'])
    moves, score, reasons, _ = prepare_root_moves(game, context, diag, **safety, policy_scores=None,
                                                  order_by_policy=False, extra=0)
    diag.v7_module_seconds = perf_counter() - module_started

    tactical = c['tactical_score_threshold'] is not None and score >= c['tactical_score_threshold']
    budget = c['tactical_simulations'] if tactical else c['simulations']
    diag.best_root_tactical_score = score
    diag.selected_simulations = budget
    diag.simulation_mode = "tactical" if tactical else "normal"
    diag.root_candidates = tuple(moves)
    diag.v8_tree_mode = c['tree_mode']
    diag.v8_tree_simulations = budget
    if remaining(deadline) < t['puct_reserve'] + t['final_margin']:
        # No time for a full tree: the first root candidate (V7 order, safest tier first).
        info.skipped.append('puct')
        diag.v8_route = "tree"
        return done(best('root_order', moves[0]) if moves else info.emergency)

    tree_started = perf_counter()
    stats = PUCTStats()
    chosen, children = search_tree_puct(
        game, moves, budget, c_puct=c['puct_c'], prior=c['puct_prior'],
        policy=root_policy if c['puct_prior'] == "policy" else None,
        candidate_limit=c['candidate_limit'], neighborhood_radius=c['neighborhood_radius'],
        priority_top_k=c['priority_top_k'], random=random, stats=stats)
    top = max(moves, key=lambda m: (stats.root_prior[m], -moves.index(m)))
    diag.v8_puct_prior = c['puct_prior']
    diag.v8_puct_nn_calls = stats.nn_calls
    diag.v8_puct_prior_fallbacks = stats.prior_fallbacks
    diag.v8_puct_prior_entropy = prior_entropy([stats.root_prior[m] for m in moves])
    diag.v8_puct_prior_top = top
    diag.v8_puct_prior_top_prob = stats.root_prior[top]
    diag.v8_tree_seconds = perf_counter() - tree_started
    v7_move = chosen
    best('puct', chosen)

    if c['root_vct_safety']:
        chosen, cut = _verify_root_choice(
            game, chosen, children, diag, node_limit=c['root_vcf_node_limit'], call_limit=c['root_call_limit'],
            node_budget=c['root_node_budget'], max_children=c['root_max_children'], mode=c['root_vct_mode'],
            deadline=limited(deadline, reserve=t['vct2_reserve'] + t['final_margin']) if c['root_vct2_check']
            else limited(deadline, reserve=t['final_margin']))
        if cut:
            info.timeouts.append('root')
        best('root', chosen)
    if c['root_vct2_check']:
        if remaining(deadline) < t['vct2_min'] + t['final_margin']:
            info.skipped.append('vct2')
        else:
            chosen, cut = _vct2_veto(
                game, chosen, children, diag, budget={'node_limit': c['vct2_node_limit'],
                                                      'call_limit': c['vct2_call_limit'],
                                                      'node_budget': c['vct2_node_budget']},
                max_children=c['vct2_max_children'], deadline=limited(deadline, reserve=t['final_margin']))
            if cut:
                info.timeouts.append('vct2')
            best('vct2', chosen)
    selected = sorted(reasons.get(chosen, ()), key=lambda k: (-PRIORITY[k], k))
    diag.v6_selected_reasons = tuple(selected)
    diag.v6_selected_threat_type = selected[0] if selected else None
    diag.v8_route = "tree"
    diag.v8_v7_move = v7_move
    diag.v8_changed = chosen != v7_move
    diag.v8_root_visits = tuple((child.move, child.visits, child.mean_value) for child in children)
    return done(chosen)


class RT30Agent(MCTSV8Agent):
    """``MCTSV8Agent`` (same configuration handling and random stream) running ``rt30_search``."""

    def __init__(self, seed: int = 42, root_policy=None, timing: dict | None = None,
                 attack_order: str = 'frozen', attack_scheduler: str = 'fair', **overrides):
        super().__init__(seed=seed, root_policy=root_policy, **overrides)
        if attack_order not in ATTACK_ORDERS or attack_scheduler not in ATTACK_SCHEDULERS:
            raise ValueError(f'attack_order in {ATTACK_ORDERS}, attack_scheduler in {ATTACK_SCHEDULERS}')
        self.attack_order, self.attack_scheduler = attack_order, attack_scheduler
        check_config({key: getattr(self, key) for key in V8_DEFAULTS})
        self.name = 'TB-RT30'
        self.timing = {**RT30_TIMING, **(timing or {})}
        self.rt_info = RTInfo()

    def select_move(self, game, deadline=None, progress=None) -> Move:
        return rt30_search(game, config={key: getattr(self, key) for key in V8_DEFAULTS},
                           root_policy=self.root_policy, random=self._random, diagnostics=self.diagnostics,
                           deadline=deadline, timing=self.timing, progress=progress, info=self.rt_info,
                           attack_order=self.attack_order, attack_scheduler=self.attack_scheduler)
