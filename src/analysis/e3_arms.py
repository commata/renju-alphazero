"""E3 arms on top of the frozen S3-VCT2-v1 (docs/mcts-v8-teacher.md §12.31, §12.32).

``S3_VCT2_V1`` is never edited: each arm is a separate configuration or a wrapper.

- **E3-A** (``E3_A``): the tree route's selective depth-2 check with a 50k node budget per
  check instead of 10k. Everything else, K = 4 and the replacement rule included, is S3-VCT2-v1.
- **E3-S** (``StageVCT2Agent``): S3-VCT2-v1 unchanged, plus the same selective 10k check on the
  final move of the Stage 4/5 routes (V8-A), which S3 never checks. The rule is fixed before the
  runs (§12.32):

  1. check V8-A's final move; if it is not proven lost, keep it;
  2. if it is, try the rest of the forced-defense order (Stage 4: ``_stage4_order``, V7's move
     first; Stage 5: the single point), skipping moves V8-A proved VCT1-UNSAFE and moves that
     lose at once;
  3. when no forced move qualifies, widen to the V6 root candidates (``_root_candidates_v6``)
     with the same exclusions;
  4. play the first move that the check does not prove lost (UNKNOWN allowed, as in S3);
  5. with none, keep V8-A's move.

  There is no cap on the number of checks (each is one fresh 10k ``SelectiveSolver``); the
  cost is recorded per move.
"""
from __future__ import annotations

from time import perf_counter

from renju import Game
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import _root_candidates_v6

from .mcts_v8 import SearchDiagnostics, _not_immediately_lost, _stage4_order
from .s3_vct2_v1 import S3_VCT2_V1
from .selective_vct import SelectiveSolver
from .threats import UNSAFE

Move = tuple[int, int]
E3_A = {**S3_VCT2_V1, 'vct2_node_budget': 50_000}
STAGE_ROUTES = ('stage4', 'stage5')
STAGE_VCT2_BUDGET = {'node_limit': S3_VCT2_V1['vct2_node_limit'], 'call_limit': S3_VCT2_V1['vct2_call_limit'],
                     'node_budget': S3_VCT2_V1['vct2_node_budget']}


def selective_status(game: Game, move: Move, budget: dict) -> tuple[str, int]:
    """S3's check of ``move`` for ``game.to_play``: UNSAFE (proven lost), SAFE (none found) or UNKNOWN."""
    solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    status = solver._bounded(game, tuple(move), None, None, lambda g: solver.after_move(g, 2)[0])
    return status, solver.nodes_used


def stage_defense_order(game: Game, route: str, v7_move: Move, candidate_limit: int,
                        neighborhood_radius: int) -> list[tuple[Move, str]]:
    """[(move, tier)] V8-A could play: the forced defenses in V8-A's order, then the widened root."""
    diag = SearchDiagnostics()
    context = _RootContext(game.legal_moves(), diag)
    _forced_v5_move(game, context=context)
    if f'stage{diag.forced_policy_stage}' != route:
        raise ValueError(f'position gives stage {diag.forced_policy_stage}, the route is {route}')
    forced = _stage4_order(game, context, tuple(v7_move)) if route == 'stage4' else [tuple(v7_move)]
    root, _, _ = _root_candidates_v6(game, context, candidate_limit, neighborhood_radius)
    seen = set(forced)
    return [(m, 'forced') for m in forced] + [(m, 'widened') for m in root if m not in seen]


def stage_vct2_veto(game: Game, route: str, played: Move, v7_move: Move, v8a_checked, *,
                    budget: dict = STAGE_VCT2_BUDGET, candidate_limit: int = S3_VCT2_V1['candidate_limit'],
                    neighborhood_radius: int = S3_VCT2_V1['neighborhood_radius']) -> tuple[Move, dict]:
    """The E3-S rule (module docstring): the move to play and the check record."""
    started = perf_counter()
    played = tuple(played)
    v8a = {tuple(m): s for m, s in v8a_checked}
    status, nodes = selective_status(game, played, budget)
    checked = [[list(played), status, nodes, 'played']]
    final, tier = played, 'played'
    if status == UNSAFE:
        for move, move_tier in stage_defense_order(game, route, v7_move, candidate_limit, neighborhood_radius):
            if move == played or v8a.get(move) == UNSAFE or not _not_immediately_lost(game, move):
                continue
            status, nodes = selective_status(game, move, budget)
            checked.append([list(move), status, nodes, move_tier])
            if status != UNSAFE:
                final, tier = move, move_tier
                break
    info = {'checked': checked, 'switched': final != played, 'final_tier': tier,
            'nodes': sum(c[2] for c in checked), 'seconds': round(perf_counter() - started, 4)}
    return final, info


class StageVCT2Agent:
    """S3-VCT2-v1 (or any V8 agent) plus the E3-S check on its Stage 4/5 moves."""

    def __init__(self, base, budget: dict | None = None):
        self.base = base
        self.budget = dict(budget or STAGE_VCT2_BUDGET)
        self.name = f'{base.name}+stagevct2'
        self.stage_info: dict = {}

    @property
    def diagnostics(self):
        return self.base.diagnostics

    def select_move(self, game: Game) -> Move:
        move = self.base.select_move(game)
        self.stage_info = {}
        diag = self.base.diagnostics
        if diag.v8_route not in STAGE_ROUTES:
            return move
        final, info = stage_vct2_veto(game, diag.v8_route, move, diag.v8_v7_move, diag.v8_vct_checked,
                                      budget=self.budget, candidate_limit=self.base.candidate_limit,
                                      neighborhood_radius=self.base.neighborhood_radius)
        self.stage_info = info
        if info['switched']:
            diag.v8_changed = True
        return final


def make_agent(arm: str, policy, seed: int = 42):
    """'s3' (S3-VCT2-v1), 'e3a' or 'e3s'; ``policy`` is the H3 ``RootPolicy``."""
    from .mcts_v8_agent import MCTSV8Agent
    from .s3_vct2_v1 import make_agent as make_s3

    if arm == 's3':
        return make_s3(policy, seed=seed)
    if arm == 'e3a':
        agent = MCTSV8Agent(seed=seed, root_policy=policy, **E3_A)
        agent.name = 'E3-A'
        return agent
    if arm == 'e3s':
        agent = StageVCT2Agent(make_s3(policy, seed=seed))
        agent.name = 'E3-S'
        return agent
    raise ValueError(f'unknown E3 arm {arm!r}')
