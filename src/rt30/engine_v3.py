"""TB-RT30-v3 move search: TB-RT30-v1 with V8-B in a parallel helper process (X2, docs/web-ser5.md §9.8).

``rt30.engine`` (TB-RT30-v1, locked) is not changed: this module copies ``rt30_search`` and changes
one thing. TB-Engine-v1 runs V8-B first and, when it proves a win, plays it without running the
rest; V8-B is independent of everything after it. So v3 sends V8-B (with the move's whole time:
the deadline minus the final margin, no 8 s cap) to the resident helper (``rt30.attack_helper``)
and runs the rest of the flow meanwhile, exactly as v1 does after a V8-B that found nothing. Then:

- the helper proved a WIN -> that move, with the diagnostics and the random stream TB-Engine-v1
  would have: the diagnostics are reset to their state before V8-B plus V8-B's own, and the agent's
  random state (the PUCT rollouts consumed it) is put back to where V8-B started;
- the helper finished without a win -> the main move (TB-Engine-v1's, the rest being the same);
- the helper was still searching at the deadline -> the main move, 'attack' in ``timeouts``.

So whenever V8-B finishes in time the move is TB-Engine-v1's. A WIN that arrives while the main
search is still running ends the move at the next phase boundary. Without a usable helper (it
cannot start in a daemonic process, it died, or the deadline is a test clock) V8-B runs inline with
v1's cap. ``RTInfoV3`` logs ``attack_parallel``, ``attack_helper_seconds``, ``attack_wait_seconds``,
``attack_win_early`` and ``attack_helper_failed``.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, perf_counter

from analysis.mcts_v8 import (
    SearchDiagnostics, _stage4_order, _stage_vct_move, V8_DEFAULTS, prepare_root_moves,
)
from analysis.puct_v8 import PUCTStats, prior_entropy, search_tree_puct
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import PRIORITY
from search.mcts_v7 import _find_vcf_with_stats, _stage4_v7_move

from . import attack_helper
from .engine import (
    RT30_TIMING, RT30Agent, RTInfo, _own_vct_attack, _verify_root_choice, _vct2_veto, check_config,
    emergency_move, limited, remaining,
)
from .solver import DeadlineSolver

Move = tuple[int, int]


@dataclass
class RTInfoV3(RTInfo):
    attack_parallel: bool = False        # V8-B ran in the helper
    attack_helper_seconds: float = 0.0   # the helper's V8-B time
    attack_wait_seconds: float = 0.0     # time the main search waited for the helper at the end
    attack_win_early: bool = False       # the helper's WIN ended the move before the main search did
    attack_helper_failed: str = ''       # why the helper could not be used ('' = it was, or not needed)


class _Parallel:
    """One V8-B request to the helper and what the move needs to undo when it wins."""

    def __init__(self, helper, request_id, diag_before: dict, random_state, until: float | None):
        self.helper, self.request_id = helper, request_id
        self.diag_before, self.random_state, self.until = diag_before, random_state, until
        self.answer = None
        self.done = False

    def poll(self) -> dict | None:
        if not self.done:
            answer = self.helper.result(self.request_id, until=monotonic())
            if answer is not None:
                self.answer, self.done = answer, True
        return self.answer

    def wait(self) -> dict | None:
        if not self.done:
            self.answer = self.helper.result(self.request_id, until=self.until)
            self.done = True
        return self.answer


def rt30_search_v3(game, *, config: dict, root_policy=None, random=None, diagnostics=None, deadline=None,
                   timing: dict | None = None, progress=None, info: RTInfoV3 | None = None,
                   attack_order: str = 'frozen', root_rescue: bool = True, parallel_attack: bool = True) -> Move:
    c = {**V8_DEFAULTS, **config}
    check_config(c)
    if c['puct_prior'] == 'policy' and root_policy is None:
        raise ValueError("puct_prior='policy' needs a root_policy callable")
    t = {**RT30_TIMING, **(timing or {})}
    info = info if info is not None else RTInfoV3()
    info.__dict__.update(vars(RTInfoV3()))
    info.deadline = deadline is not None
    info.remaining_at_start = None if deadline is None else round(deadline.remaining(), 3)
    move_started = perf_counter()
    pending: _Parallel | None = None

    def best(stage: str, move: Move) -> Move:
        info.best.append((stage, tuple(move)))
        if progress is not None:
            progress(stage, tuple(move))
        return move

    def done(move: Move) -> Move:
        info.seconds = perf_counter() - move_started
        return move

    def attack_win(answer: dict) -> Move:
        """TB-Engine-v1's state after a V8-B win: diagnostics and random stream as if nothing else ran."""
        info.attack_helper_seconds = round(answer['seconds'], 4)
        diag.__dict__.clear()
        diag.__dict__.update(pending.diag_before)
        diag.__dict__.update(answer['diag'])
        if random is not None and pending.random_state is not None:
            random.setstate(pending.random_state)
        diag.forced_policy_stage = None
        diag.v8_route = "own_vct"
        diag.v8_changed = True
        return done(best('own_vct', tuple(answer['move'])))

    def early() -> Move | None:
        """The helper's WIN, if it has arrived (checked between phases)."""
        if pending is not None:
            answer = pending.poll()
            if answer is not None and answer['move'] is not None:
                info.attack_win_early = True
                return attack_win(answer)
        return None

    def finish(move: Move) -> Move:
        if pending is None:
            return done(move)
        waited = perf_counter()
        answer = pending.wait()
        info.attack_wait_seconds = round(perf_counter() - waited, 4)
        if answer is None:
            info.timeouts.append('attack')
            info.attack_helper_failed = pending.helper.last_error or ('' if pending.helper.alive() else 'helper died')
            return done(move)
        info.attack_helper_seconds = round(answer['seconds'], 4)
        if answer['move'] is not None:
            return attack_win(answer)
        diag.__dict__.update(answer['diag'])  # TB-Engine-v1 records V8-B's search before the rest
        if answer['timed_out']:
            info.timeouts.append('attack')
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
        helper = None
        if not parallel_attack:
            info.attack_helper_failed = 'off'
        elif deadline is not None and getattr(deadline, 'clock', None) is not monotonic:
            info.attack_helper_failed = 'test clock'  # the helper shares only the machine's monotonic clock
        else:
            helper = attack_helper.get_helper()
            if helper is None:
                info.attack_helper_failed = attack_helper.unavailable_reason() or 'unavailable'
        if helper is not None:
            params = {'node_limit': c['attack_vcf_node_limit'], 'call_limit': c['attack_call_limit'],
                      'node_budget': c['attack_node_budget'], 'order': attack_order}
            diag.forced_policy_stage = None  # as TB-Engine-v1 records it after a V8-B win
            before = dict(vars(diag))
            diag.forced_policy_stage = forced_stage
            request = helper.submit(game.history, params, None if hard is None else hard.at)
            pending = _Parallel(helper, request, before, random.getstate() if random is not None else None,
                                None if hard is None else hard.at)
            info.attack_parallel = True
        else:
            attack, cut = _own_vct_attack(
                game, context, diag, node_limit=c['attack_vcf_node_limit'], call_limit=c['attack_call_limit'],
                node_budget=c['attack_node_budget'],
                deadline=limited(deadline, reserve=t['final_margin'], cap=t['attack_seconds']),
                order=attack_order, scheduler='fair')
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
            return finish(forced)
        won = early()
        if won is not None:
            return won
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
    won = early()
    if won is not None:
        return won

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
        won = early()
        if won is not None:
            return won
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
        won = early()
        if won is not None:
            return won
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


class RT30AgentV3(RT30Agent):
    """``RT30Agent`` running ``rt30_search_v3``."""

    def __init__(self, seed: int = 42, root_policy=None, timing: dict | None = None,
                 attack_order: str = 'frozen', root_rescue: bool = True, parallel_attack: bool = True, **overrides):
        super().__init__(seed=seed, root_policy=root_policy, timing=timing, attack_order=attack_order,
                         attack_scheduler='fair', root_rescue=root_rescue, **overrides)
        self.name = 'TB-RT30-v3'
        self.parallel_attack = parallel_attack
        self.rt_info = RTInfoV3()

    def select_move(self, game, deadline=None, progress=None) -> Move:
        return rt30_search_v3(game, config={key: getattr(self, key) for key in V8_DEFAULTS},
                              root_policy=self.root_policy, random=self._random, diagnostics=self.diagnostics,
                              deadline=deadline, timing=self.timing, progress=progress, info=self.rt_info,
                              attack_order=self.attack_order, root_rescue=self.root_rescue,
                              parallel_attack=self.parallel_attack)
