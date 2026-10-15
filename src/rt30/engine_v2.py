"""TB-RT30-v2 move search: TB-RT30-v1 plus the V8-B slack retry (docs/web-ser5.md §9.5).

``rt30.engine`` (TB-RT30-v1, locked) is not changed: this module copies ``rt30_search`` and changes
one thing. V8-B still stops at its soft cap (``attack_seconds`` from its start, A1); when that cap,
not the move's deadline, cut it, the cut search is kept, the rest of the move runs exactly as in v1,
and if time is left before the move's own deadline (minus the final margin) V8-B resumes on the same
solver, statuses and VCF cache. A WIN proven then is played (``own_vct``), as TB-Engine-v1 would
have played the win its uncut V8-B found; otherwise v1's move stands.

- Only a time cut resumes. A V8-B that finished (WIN found or budget spent, ``exhausted`` by the
  node / call budget) is never retried, and the retry spends the same 200k-node / 10k-call budget.
- The resumed fair scheduler starts its rounds again from the remaining budget (``_first_proven``
  over the moves still UNKNOWN or unchecked; finished sub-results come from the cache). So a found
  win is proven, but it may be another win than the one TB-Engine-v1's single run finds first.
- ``RTInfoV2`` keeps the soft cap apart from real time shortage: ``attack_cap_hit`` (the soft cap
  stopped V8-B), ``attack_retried`` / ``attack_retry_seconds`` / ``attack_retry_win``; ``timeouts``
  lists 'attack' only when V8-B was still unfinished when the move was played.

With ``deadline=None`` nothing is cut, nothing is retried, and the move is TB-Engine-v1's.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from analysis.mcts_v8 import (
    WIN, SearchDiagnostics, _attack_candidates, _first_proven, _stage4_order, _stage_vct_move, V8_DEFAULTS,
    prepare_root_moves,
)
from analysis.puct_v8 import PUCTStats, prior_entropy, search_tree_puct
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import PRIORITY
from search.mcts_v7 import _find_vcf_with_stats, _stage4_v7_move

from .engine import (
    RT30_TIMING, RT30Agent, RTInfo, _verify_root_choice, _vct2_veto, attack_order as order_attack,
    check_config, emergency_move, limited, remaining,
)
from .deadline import Deadline
from .solver import DeadlineSolver

Move = tuple[int, int]
V2_TIMING = {**RT30_TIMING, 'attack_retry_min': 1.0}  # retry only with this much left before the final margin


@dataclass
class RTInfoV2(RTInfo):
    attack_cap_hit: bool = False         # the soft cap (not the move's deadline) stopped V8-B
    attack_retried: bool = False
    attack_retry_seconds: float = 0.0
    attack_retry_win: bool = False


@dataclass
class AttackState:
    """A V8-B search stopped by time: what a resume continues from."""
    solver: DeadlineSolver
    candidates: list
    statuses: dict
    started: float


def _own_vct_attack(game, context, diag, *, node_limit, call_limit, node_budget, deadline, order='frozen'):
    """``rt30.engine._own_vct_attack`` (fair scheduler) that also returns the state of a time-cut search."""
    started = perf_counter()
    candidates = order_attack(game, _attack_candidates(game, context), order)
    diag.v8_attack_candidates = len(candidates)
    solver = DeadlineSolver(node_limit=node_limit, call_limit=call_limit, node_budget=node_budget, deadline=deadline)
    statuses: dict[Move, str] = {}
    chosen = _first_proven(game, candidates, statuses, solver, solver.attack_status, WIN)
    _record_attack(diag, solver, statuses, candidates, chosen, perf_counter() - started)
    state = AttackState(solver, candidates, statuses, started) if solver.timed_out else None
    return chosen, state


def resume_attack(game, state: AttackState, diag, deadline):
    """Continue a time-cut V8-B on its own solver until ``deadline``; returns (win or None, cut again)."""
    solver = state.solver
    solver.deadline, solver.timed_out, solver.exhausted = deadline, False, False
    chosen = _first_proven(game, state.candidates, state.statuses, solver, solver.attack_status, WIN)
    _record_attack(diag, solver, state.statuses, state.candidates, chosen, perf_counter() - state.started)
    return chosen, solver.timed_out


def _record_attack(diag, solver, statuses, candidates, chosen, seconds):
    diag.v8_attack_checked = tuple(statuses.items())
    diag.v8_attack_calls = solver.vcf_calls
    diag.v8_attack_nodes = solver.nodes_used
    diag.v8_attack_budget_exhausted = solver.exhausted
    diag.v8_attack_seconds = seconds
    if chosen is not None:
        diag.v8_attack_status = WIN
        diag.v8_attack_move = chosen
        diag.v8_attack_rank = candidates.index(chosen) + 1


def rt30_search_v2(game, *, config: dict, root_policy=None, random=None, diagnostics=None, deadline=None,
                   timing: dict | None = None, progress=None, info: RTInfoV2 | None = None,
                   attack_order: str = 'frozen', root_rescue: bool = True, attack_retry: bool = True) -> Move:
    c = {**V8_DEFAULTS, **config}
    check_config(c)
    if c['puct_prior'] == 'policy' and root_policy is None:
        raise ValueError("puct_prior='policy' needs a root_policy callable")
    t = {**V2_TIMING, **(timing or {})}
    info = info if info is not None else RTInfoV2()
    info.__dict__.update(vars(RTInfoV2()))
    info.deadline = deadline is not None
    info.remaining_at_start = None if deadline is None else round(deadline.remaining(), 3)
    move_started = perf_counter()
    attack_state: AttackState | None = None

    def best(stage: str, move: Move) -> Move:
        info.best.append((stage, tuple(move)))
        if progress is not None:
            progress(stage, tuple(move))
        return move

    def done(move: Move) -> Move:
        info.seconds = perf_counter() - move_started
        return move

    def finish(move: Move) -> Move:
        """The v1 move, unless a soft-capped V8-B proves a win in the time left."""
        if attack_state is not None and attack_retry and info.attack_cap_hit:
            if remaining(deadline) >= t['attack_retry_min'] + t['final_margin']:
                retry_started = perf_counter()
                info.attack_retried = True
                win, cut = resume_attack(game, attack_state, diag, limited(deadline, reserve=t['final_margin']))
                info.attack_retry_seconds = round(perf_counter() - retry_started, 4)
                if not cut:
                    info.timeouts.remove('attack')  # V8-B finished: a win, or none within its budget
                if win is not None:
                    info.attack_retry_win = True
                    diag.v8_route = "own_vct"
                    diag.v8_changed = True
                    return done(best('own_vct_retry', win))
        return done(move)

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
        hard = limited(deadline, reserve=t['final_margin'])
        capped = limited(deadline, reserve=t['final_margin'], cap=t['attack_seconds'])
        attack, attack_state = _own_vct_attack(
            game, context, diag, node_limit=c['attack_vcf_node_limit'], call_limit=c['attack_call_limit'],
            node_budget=c['attack_node_budget'], deadline=capped, order=attack_order)
        if attack_state is not None:
            info.timeouts.append('attack')
            info.attack_cap_hit = isinstance(capped, Deadline) and capped.at < hard.at
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
            return finish(forced)
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
        return finish(best('stage_vct', chosen))

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
        info.skipped.append('puct')
        diag.v8_route = "tree"
        return finish(best('root_order', moves[0]) if moves else info.emergency)

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

    rescued = False
    if c['root_vct_safety']:
        rescue_deadline = (limited(deadline, reserve=t['final_margin'])
                           if root_rescue and deadline is not None and c['root_vct2_check'] else None)
        chosen, cut, rescued = _verify_root_choice(
            game, chosen, children, diag, node_limit=c['root_vcf_node_limit'], call_limit=c['root_call_limit'],
            node_budget=c['root_node_budget'], max_children=c['root_max_children'], mode=c['root_vct_mode'],
            deadline=limited(deadline, reserve=t['vct2_reserve'] + t['final_margin']) if c['root_vct2_check']
            else limited(deadline, reserve=t['final_margin']), rescue_deadline=rescue_deadline)
        if cut:
            info.timeouts.append('root')
        if rescued:
            info.rescued = True
        best('root', chosen)
    if c['root_vct2_check']:
        if rescued:
            info.skipped.append('vct2_root_rescue')
        elif remaining(deadline) < t['vct2_min'] + t['final_margin']:
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
    return finish(chosen)


class RT30AgentV2(RT30Agent):
    """``RT30Agent`` running ``rt30_search_v2`` (fair V8-B scheduler only: the one that resumes)."""

    def __init__(self, seed: int = 42, root_policy=None, timing: dict | None = None,
                 attack_order: str = 'frozen', root_rescue: bool = True, attack_retry: bool = True, **overrides):
        super().__init__(seed=seed, root_policy=root_policy, timing=None, attack_order=attack_order,
                         attack_scheduler='fair', root_rescue=root_rescue, **overrides)
        self.name = 'TB-RT30-v2'
        self.timing = {**V2_TIMING, **(timing or {})}
        self.attack_retry = attack_retry
        self.rt_info = RTInfoV2()

    def select_move(self, game, deadline=None, progress=None) -> Move:
        return rt30_search_v2(game, config={key: getattr(self, key) for key in V8_DEFAULTS},
                              root_policy=self.root_policy, random=self._random, diagnostics=self.diagnostics,
                              deadline=deadline, timing=self.timing, progress=progress, info=self.rt_info,
                              attack_order=self.attack_order, root_rescue=self.root_rescue,
                              attack_retry=self.attack_retry)
