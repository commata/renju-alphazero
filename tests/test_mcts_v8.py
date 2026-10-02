import json
from pathlib import Path
from random import Random
import unittest

from agents import MCTSV7Agent
from analysis.mcts_v8 import (
    REFUTED, V8_DEFAULTS, WIN, SearchDiagnostics, _attack_candidates, _BudgetedSolver, _board_key,
    _first_proven, _first_safe, _not_immediately_lost, _own_vct_attack, _proven, _stage_vct_move,
    _verify_root_choice, mcts_search_v8,
)
from analysis.tactical_labels import proves_threat
from search.mcts_v5 import _RootContext
from search.mcts_v6 import _root_candidates_v6
from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.threats import SAFE, UNKNOWN, UNSAFE, ThreatSolver
from renju import Game
from search.mcts_v7 import V7_FINAL, mcts_search_v7

ROOT = Path(__file__).resolve().parents[1]
GAMES = json.loads((ROOT / 'tests/fixtures/web_play_v7_human_games_v1.json').read_text(encoding='utf-8'))['games']
PROBES = json.loads((ROOT / 'tests/fixtures/vct_probes_v1.json').read_text(encoding='utf-8'))['probes']
OFF = {**V8_DEFAULTS, 'stage_vct_safety': False, 'own_vct_attack': False, 'root_vct_safety': False}  # V7
A_ONLY = {**V8_DEFAULTS, 'own_vct_attack': False, 'root_vct_safety': False}  # V8-A regression config


def _game(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _probe(game_prefix, kind='must_defend_vct'):
    # 163903 is the cheapest game: Stage 4 branch at ply 13, VCT1 attack (8, 5) at ply 14.
    return next(p for p in PROBES if p['kind'] == kind and p['symmetry'] == 0
                and p['source']['game'].startswith(game_prefix))


# One human-game position per V7 route: (game index, ply) -> route/simulation mode with modules off.
ROUTE_POSITIONS = {
    (0, 2): 'tree/normal', (2, 17): 'tree/tactical', (0, 19): 'stage1', (0, 11): 'stage2',
    (0, 17): 'stage3', (1, 15): 'own_vcf', (0, 9): 'stage4', (0, 12): 'stage5',
}


def _route(diag):
    return diag.v8_route + ('/' + diag.simulation_mode if diag.v8_route == 'tree' else '')


class ModulesOffMatchV7Test(unittest.TestCase):
    def test_every_route_matches_v7(self):
        for (index, ply), route in ROUTE_POSITIONS.items():
            game = _game(GAMES[index]['moves'][:ply])
            with self.subTest(route=route):
                expected = mcts_search_v7(game, **V7_FINAL, random=Random(ply))
                diag = SearchDiagnostics()
                self.assertEqual(mcts_search_v8(game, **OFF, random=Random(ply), diagnostics=diag), expected)
                self.assertEqual(_route(diag), route)

    def test_same_moves_as_v7(self):
        moves = GAMES[1]['moves']
        for ply in range(2, 14):
            game = _game(moves[:ply])
            with self.subTest(ply=ply):
                expected = mcts_search_v7(game, **V7_FINAL, random=Random(ply))
                diag = SearchDiagnostics()
                self.assertEqual(mcts_search_v8(game, **OFF, random=Random(ply), diagnostics=diag), expected)
                self.assertEqual(diag.v8_v7_move, expected)
                self.assertFalse(diag.v8_changed)
                self.assertEqual(game.history, [tuple(m) for m in moves[:ply]])

    def test_tree_move_records_root_visits(self):
        diag = SearchDiagnostics()
        mcts_search_v8(_game(GAMES[1]['moves'][:4]), **OFF, random=Random(3), diagnostics=diag)
        self.assertEqual(diag.v8_route, 'tree')
        self.assertEqual(sum(visits for _, visits, _ in diag.v8_root_visits), diag.selected_simulations)


class StageVCTSafetyTest(unittest.TestCase):
    def test_stage4_branch_point_picks_proven_safe_defense(self):
        probe = _probe('20260929-163903')
        game = _game(probe['moves'])
        diag = SearchDiagnostics()
        move = mcts_search_v8(game, **A_ONLY, random=Random(1), diagnostics=diag)
        self.assertEqual(diag.v8_route, 'stage4')
        self.assertEqual(list(diag.v8_v7_move), probe['avoid_moves'][0])  # v7's losing (6, 8)
        self.assertIn(list(move), probe['correct_moves'])
        self.assertTrue(diag.v8_changed)
        self.assertEqual(dict(diag.v8_vct_checked)[move], SAFE)
        self.assertFalse(diag.v8_vct_budget_exhausted)
        self.assertEqual(game.history, [tuple(m) for m in probe['moves']])

    def test_exhausted_budget_keeps_unrefuted_v7_move(self):
        probe = _probe('20260929-163903')
        game = _game(probe['moves'])
        diag = SearchDiagnostics()
        move = mcts_search_v8(game, **{**A_ONLY, 'vct_call_limit': 1},
                              random=Random(1), diagnostics=diag)
        self.assertTrue(diag.v8_vct_budget_exhausted)
        self.assertEqual(move, diag.v8_v7_move)
        self.assertEqual(dict(diag.v8_vct_checked)[move], UNKNOWN)
        self.assertEqual(game.history, [tuple(m) for m in probe['moves']])

    def test_budgeted_solver_restores_board(self):
        game = _game(_probe('20260929-163903')['moves'])
        before = _board_key(game)
        solver = _BudgetedSolver(node_limit=20_000, call_limit=3, node_budget=10**6)
        self.assertEqual(solver.status_after(game, (6, 8)), UNKNOWN)
        self.assertTrue(solver.exhausted)
        self.assertEqual(_board_key(game), before)


    def test_node_budget_cut_is_unknown_not_cached(self):
        game = _game(_probe('20260929-163903')['moves'])
        before = _board_key(game)
        solver = _BudgetedSolver(node_limit=20_000, call_limit=10**6, node_budget=50)
        self.assertEqual(solver.status_after(game, (6, 8)), UNKNOWN)
        self.assertTrue(solver.exhausted)
        self.assertLessEqual(solver.nodes_used, 50)
        self.assertEqual(_board_key(game), before)


class _ScriptedSolver:
    """Stand-in for _BudgetedSolver: move -> (VCF nodes, final status[, VCF calls]) to finish.

    Work on a move accumulates across calls like the real solver's caches, so a
    cut move resumes where it stopped. A move finishes once both its node and
    call needs are met.
    """

    def __init__(self, script, node_budget, default=(10**9, UNKNOWN), call_limit=10**9):
        self.script, self.default = script, default
        self.node_budget, self.nodes_used = node_budget, 0
        self.call_limit, self.vcf_calls = call_limit, 0
        self.exhausted = self.last_cut = False
        self.spent = {}
        self.calls = []

    def attack_status(self, game, move, share=None, call_share=None):
        return self.status_after(game, move, share, call_share)

    def status_after(self, game, move, share=None, call_share=None):
        self.calls.append((move, share, call_share))
        self.last_cut = True
        if self.exhausted:
            return UNKNOWN
        entry = self.script.get(move, self.default)
        cost, status, calls = entry if len(entry) == 3 else (*entry, 0)
        spent_nodes, spent_calls = self.spent.get(move, (0, 0))
        node_room = self.node_budget - self.nodes_used
        call_room = self.call_limit - self.vcf_calls
        node_room = node_room if share is None else min(share, node_room)
        call_room = call_room if call_share is None else min(call_share, call_room)
        add_nodes = min(cost - spent_nodes, node_room)
        add_calls = min(calls - spent_calls, call_room)
        self.nodes_used += add_nodes
        self.vcf_calls += add_calls
        self.spent[move] = (spent_nodes + add_nodes, spent_calls + add_calls)
        if self.spent[move] == (cost, calls):
            self.last_cut = False
            return status
        self.exhausted = self.nodes_used >= self.node_budget or self.vcf_calls >= self.call_limit
        return UNKNOWN


def _stage_position(prefix_index, ply):
    game = _game(GAMES[prefix_index]['moves'][:ply])
    context = _RootContext(game.legal_moves(), SearchDiagnostics())
    return game, context


class FirstSafeBudgetTest(unittest.TestCase):
    def test_expensive_first_move_cannot_starve_the_next(self):
        # A needs 150k nodes to be proven UNSAFE, B 60k to be proven SAFE, budget 200k.
        # A single "rest of the budget" pass after one 25k share would give A everything.
        solver = _ScriptedSolver({'A': (150_000, UNSAFE), 'B': (60_000, SAFE)}, 200_000)
        statuses = {}
        self.assertEqual(_first_safe(None, ['A', 'B'], statuses, solver), 'B')
        self.assertEqual(statuses, {'A': UNKNOWN, 'B': SAFE})
        self.assertFalse(solver.exhausted)

    def test_final_round_uses_the_whole_remainder(self):
        solver = _ScriptedSolver({'A': (190_000, SAFE)}, 200_000)
        self.assertEqual(_first_safe(None, ['A'], {}, solver), 'A')
        solver = _ScriptedSolver({'A': (150_000, UNSAFE), 'B': (45_000, SAFE)}, 200_000)
        self.assertEqual(_first_safe(None, ['A', 'B'], {}, solver), 'B')

    def test_expensive_first_move_cannot_starve_the_next_of_calls(self):
        # Nodes are plentiful; A needs 2,900 of the 3,000 VCF calls to be proven UNSAFE,
        # B needs 400 calls to be proven SAFE. A global call counter would let A take them all.
        solver = _ScriptedSolver({'A': (50, UNSAFE, 2_900), 'B': (50, SAFE, 400)}, 200_000, call_limit=3_000)
        self.assertEqual(_first_safe(None, ['A', 'B'], {}, solver), 'B')
        self.assertFalse(solver.exhausted)
        self.assertTrue(all(call_share is not None for _, _, call_share in solver.calls))

    def test_half_share_rounding_still_gives_every_candidate_a_call(self):
        # 6 calls / 4 candidates: half-share rounds to 0, but a fair share of 1 still exists.
        # The first candidate must not receive all 6 calls before B gets its one-call SAFE proof.
        solver = _ScriptedSolver(
            {'A': (1, UNSAFE, 6), 'B': (1, SAFE, 1)},
            node_budget=100, call_limit=6,
        )
        self.assertEqual(_first_safe(None, ['A', 'B', 'C', 'D'], {}, solver), 'B')
        self.assertEqual(solver.calls[0][2], 1)
        self.assertEqual(solver.calls[1][2], 1)
        self.assertFalse(solver.exhausted)

    def test_half_share_rounding_still_gives_every_candidate_a_node(self):
        # Same boundary for nodes: 6 nodes / 4 candidates can still give each one node.
        solver = _ScriptedSolver(
            {'A': (6, UNSAFE), 'B': (1, SAFE)},
            node_budget=6, call_limit=100,
        )
        self.assertEqual(_first_safe(None, ['A', 'B', 'C', 'D'], {}, solver), 'B')
        self.assertEqual(solver.calls[0][1], 1)
        self.assertEqual(solver.calls[1][1], 1)
        self.assertFalse(solver.exhausted)

    def test_unshareable_remainder_is_not_given_to_one_move(self):
        # 3 calls / 4 candidates: no fair share of one call exists. A must not take all 3;
        # every move gets one zero-budget pass and the budget is reported exhausted.
        solver = _ScriptedSolver({'A': (1, UNSAFE, 3), 'B': (1, SAFE, 1)}, node_budget=100, call_limit=3)
        self.assertIsNone(_first_safe(None, ['A', 'B', 'C', 'D'], {}, solver))
        self.assertEqual([call for _, _, call in solver.calls], [0, 0, 0, 0])
        self.assertEqual(solver.vcf_calls, 0)
        self.assertTrue(solver.exhausted)

    def test_finished_unknown_is_not_rechecked_each_round(self):
        # UNKNOWN from a per-VCF node limit is final; only moves cut by their share go on.
        solver = _ScriptedSolver({'A': (10, UNKNOWN), 'B': (150_000, SAFE)}, 200_000)
        self.assertEqual(_first_safe(None, ['A', 'B'], {}, solver), 'B')
        self.assertEqual(sum(move == 'A' for move, _, _ in solver.calls), 1)


class StageFallbackPolicyTest(unittest.TestCase):
    def _run(self, game, context, v7_move, defenses, solver):
        diag = SearchDiagnostics()
        before = list(game.history)
        chosen = _stage_vct_move(game, context, v7_move, defenses, diag, solver,
                                 candidate_limit=20, neighborhood_radius=2)
        self.assertEqual(game.history, before)
        return chosen, diag

    def test_stage5_safe_move_is_kept(self):
        game, context = _stage_position(0, 12)  # real Stage 5: v7 blocks at (6, 9)
        solver = _ScriptedSolver({(6, 9): (10, SAFE)}, 200_000)
        chosen, diag = self._run(game, context, (6, 9), [(6, 9)], solver)
        self.assertEqual(chosen, (6, 9))
        self.assertFalse(diag.v8_vct_widened)

    def test_stage5_unsafe_move_widens_to_a_safe_root_move(self):
        game, context = _stage_position(0, 12)
        root, _, _ = _root_candidates_v6(game, _RootContext(game.legal_moves(), SearchDiagnostics()), 20, 2)
        root = [m for m in root if m != (6, 9)]
        script = {(6, 9): (10, UNSAFE), root[0]: (10, UNSAFE), root[1]: (10, UNSAFE), root[2]: (10, SAFE)}
        chosen, diag = self._run(game, context, (6, 9), [(6, 9)], _ScriptedSolver(script, 200_000))
        self.assertTrue(diag.v8_vct_widened)
        self.assertEqual(chosen, root[2])
        self.assertEqual(dict(diag.v8_vct_checked)[chosen], SAFE)

    def test_widened_unknown_beats_a_proven_loss(self):
        # Stage 4 (163903 ply 13): every forced defense proven UNSAFE, the root checks run out of
        # budget. The fallback must be an unrefuted root move, never V7's proven-UNSAFE move.
        probe = _probe('20260929-163903')
        game = _game(probe['moves'])
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        defenses = [(6, 8), (6, 3), (6, 4), (6, 9)]
        script = {move: (10, UNSAFE) for move in defenses}
        chosen, diag = self._run(game, context, (6, 8), defenses, _ScriptedSolver(script, 50_000))
        statuses = dict(diag.v8_vct_checked)
        self.assertTrue(diag.v8_vct_widened)
        self.assertNotIn(chosen, defenses)
        self.assertEqual(statuses[chosen], UNKNOWN)
        self.assertTrue(_not_immediately_lost(game, chosen))

    def _unchecked_root_case(self, budget):
        probe = _probe('20260929-163903')
        game = _game(probe['moves'])
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        defenses = [(6, 8), (6, 3), (6, 4), (6, 9)]
        script = {move: (10, UNSAFE) for move in defenses}
        chosen, diag = self._run(game, context, (6, 8), defenses, _ScriptedSolver(script, budget))
        statuses = dict(diag.v8_vct_checked)
        self.assertTrue(diag.v8_vct_widened)
        self.assertNotIn(chosen, defenses)
        self.assertNotEqual(statuses.get(chosen, UNKNOWN), UNSAFE)
        self.assertTrue(_not_immediately_lost(game, chosen))
        return statuses

    def test_unrefuted_root_beats_a_proven_loss_when_fair_share_is_zero(self):
        # 15 nodes left for ~20 root moves: an equal split cannot give every move one node.
        # V8 makes only a zero-budget structural pass, then falls back to an unrefuted root move.
        statuses = self._unchecked_root_case(40 + 15)
        self.assertGreater(sum(status == UNKNOWN for status in statuses.values()), 1)

    def test_unchecked_root_beats_a_proven_loss_when_budget_ran_out(self):
        # The budget ends exactly with the last defense proven UNSAFE; the root is still built
        # and a never-checked root move is preferred over V7's proven loss.
        statuses = self._unchecked_root_case(40)
        defenses = [(6, 8), (6, 3), (6, 4), (6, 9)]
        self.assertTrue(all(statuses[m] == UNSAFE for m in defenses))
        self.assertTrue(all(statuses[m] == UNKNOWN for m in statuses if m not in defenses))

    def test_unrefuted_defense_beats_v7_proven_loss(self):
        game, context = _stage_position(0, 12)
        script = {(6, 9): (10, UNSAFE), (5, 5): (10**9, UNKNOWN)}
        chosen, _ = self._run(game, context, (6, 9), [(6, 9), (5, 5)], _ScriptedSolver(script, 1_000))
        self.assertEqual(chosen, (5, 5) if _not_immediately_lost(game, (5, 5)) else (6, 9))

    def test_all_proven_unsafe_keeps_v7_move(self):
        game, context = _stage_position(0, 12)
        solver = _ScriptedSolver({}, 10**7, default=(10, UNSAFE))
        chosen, diag = self._run(game, context, (6, 9), [(6, 9)], solver)
        self.assertTrue(diag.v8_vct_widened)
        self.assertEqual(chosen, (6, 9))
        self.assertTrue(all(status == UNSAFE for _, status in diag.v8_vct_checked))


class Stage5EngineTest(unittest.TestCase):
    def test_real_stage5_block_is_proven_safe(self):
        game = _game(GAMES[0]['moves'][:12])
        diag = SearchDiagnostics()
        move = mcts_search_v8(game, **A_ONLY, random=Random(12), diagnostics=diag)
        self.assertEqual(diag.v8_route, 'stage5')
        self.assertEqual(move, (6, 9))
        self.assertEqual(dict(diag.v8_vct_checked), {(6, 9): SAFE})
        self.assertFalse(diag.v8_changed)
        self.assertEqual(game.history, [tuple(m) for m in GAMES[0]['moves'][:12]])

    def test_attack_without_win_leaves_v8a_unchanged(self):
        # A on + B on with no proven WIN must play exactly what A on + B off plays.
        moves = GAMES[0]['moves'][:12]
        a_only, both = SearchDiagnostics(), SearchDiagnostics()
        expected = mcts_search_v8(_game(moves), **A_ONLY, random=Random(12), diagnostics=a_only)
        game = _game(moves)
        self.assertEqual(mcts_search_v8(game, **V8_DEFAULTS, random=Random(12), diagnostics=both), expected)
        self.assertEqual(both.v8_route, a_only.v8_route)
        self.assertEqual(both.v8_attack_status, '')
        self.assertNotIn(WIN, dict(both.v8_attack_checked).values())
        self.assertEqual(game.history, [tuple(m) for m in moves])


class OwnVCTAttackTest(unittest.TestCase):
    # 163903 ply 14: white's proven VCT1 threat is (8, 5), 6th of 8 candidates (design §4.2.2).
    def _attack_game(self):
        probe = _probe('20260929-163903', 'vct_attack')
        return _game(probe['moves']), probe

    def test_real_attack_is_proven_and_independently_verified(self):
        game, probe = self._attack_game()
        diag = SearchDiagnostics()
        move = mcts_search_v8(game, **V8_DEFAULTS, random=Random(1), diagnostics=diag)
        self.assertEqual(diag.v8_route, 'own_vct')
        self.assertIn(list(move), probe['correct_moves'])
        self.assertEqual(dict(diag.v8_attack_checked)[move], WIN)
        self.assertEqual(diag.v8_attack_status, WIN)
        self.assertEqual(diag.v8_attack_move, move)
        self.assertEqual(diag.v8_attack_rank, 6)
        self.assertTrue(diag.v8_changed)
        self.assertIsNone(diag.v8_v7_move)
        self.assertEqual(game.history, [tuple(m) for m in probe['moves']])
        self.assertTrue(proves_threat(game, move, ThreatSolver(node_limit=100_000)))

    def test_unproven_attack_is_never_played(self):
        # A 50-node budget cannot prove (8, 5): V8-B plays nothing and V8 continues as with B off.
        game, probe = self._attack_game()
        diag, a_only = SearchDiagnostics(), SearchDiagnostics()
        move = mcts_search_v8(game, **{**V8_DEFAULTS, 'attack_node_budget': 50},
                              random=Random(1), diagnostics=diag)
        expected = mcts_search_v8(_game(probe['moves']), **{**V8_DEFAULTS, 'own_vct_attack': False},
                                  random=Random(1), diagnostics=a_only)
        self.assertEqual(move, expected)
        self.assertNotEqual(diag.v8_route, 'own_vct')
        self.assertEqual(diag.v8_attack_status, '')
        self.assertTrue(diag.v8_attack_budget_exhausted)
        self.assertNotIn(WIN, dict(diag.v8_attack_checked).values())
        self.assertEqual(game.history, [tuple(m) for m in probe['moves']])

    def test_budget_cut_then_resume_still_needs_a_full_proof(self):
        game, _ = self._attack_game()
        before = _board_key(game)
        solver = _BudgetedSolver(node_limit=20_000, call_limit=10**6, node_budget=10**6)
        self.assertEqual(solver.attack_status(game, (8, 5), 100, 10**6), UNKNOWN)
        self.assertTrue(solver.last_cut)
        self.assertEqual(_board_key(game), before)
        self.assertEqual(solver.attack_status(game, (8, 5)), WIN)
        self.assertFalse(solver.last_cut)
        self.assertEqual(solver.attack_status(game, (9, 6)), REFUTED)  # refuted in one VCF call
        self.assertEqual(_board_key(game), before)

    def test_attack_that_makes_five_is_a_win(self):
        # Stage 1 normally plays a five first; the check itself must still read it as WIN.
        game = _game([(7, 7), (0, 0), (7, 8), (0, 2), (7, 9), (0, 4), (7, 10), (0, 6)])
        solver = _BudgetedSolver(node_limit=1_000, call_limit=10, node_budget=1_000)
        self.assertEqual(solver.attack_status(game, (7, 11)), WIN)
        self.assertEqual(len(game.history), 8)

    def test_candidates_are_legal_fours_or_open_threes(self):
        game, probe = self._attack_game()
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        candidates = _attack_candidates(game, context)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates.index((8, 5)), 5)
        self.assertTrue(set(candidates) <= set(context.legal))

    def test_no_candidates_costs_nothing(self):
        game = _game(GAMES[0]['moves'][:4])  # no four or open three to make yet
        diag = SearchDiagnostics()
        context = _RootContext(game.legal_moves(), diag)
        self.assertIsNone(_own_vct_attack(game, context, diag, node_limit=20_000,
                                          call_limit=10_000, node_budget=200_000))
        self.assertEqual((diag.v8_attack_candidates, diag.v8_attack_calls, diag.v8_attack_nodes), (0, 0, 0))
        self.assertFalse(diag.v8_attack_budget_exhausted)

    def test_attack_uses_the_same_fair_scheduler(self):
        # A needs 150k nodes to be REFUTED, B 60k to be a WIN: A cannot take B's budget.
        solver = _ScriptedSolver({'A': (150_000, REFUTED), 'B': (60_000, WIN)}, 200_000)
        statuses = {}
        self.assertEqual(_first_proven(None, ['A', 'B'], statuses, solver, solver.attack_status, WIN), 'B')
        self.assertEqual(statuses, {'A': UNKNOWN, 'B': WIN})


class _Child:
    def __init__(self, move, visits, mean_value=0.0):
        self.move, self.visits, self.mean_value = move, visits, mean_value


def _scripted_status(script):
    """Replace _BudgetedSolver.status_after: fixed status per move, no budget use."""
    def status_after(self, game, move, share=None, call_share=None):
        self.last_cut = False
        return script.get(move, UNKNOWN)
    return status_after


class RootVerifyPolicyTest(unittest.TestCase):
    # A quiet early position: no move loses at once, so the fallback guard never interferes.
    def _run(self, script, chosen, children):
        from unittest import mock
        game = _game(GAMES[1]['moves'][:4])
        diag = SearchDiagnostics()
        with mock.patch.object(_BudgetedSolver, 'status_after', _scripted_status(script)):
            final = _verify_root_choice(game, chosen, children, diag, node_limit=20_000,
                                        call_limit=10_000, node_budget=200_000)
        self.assertEqual(len(game.history), 4)
        return final, diag

    CHILDREN = [_Child((7, 9), 30), _Child((8, 9), 20), _Child((6, 6), 20, 0.5), _Child((9, 9), 5)]

    def test_safe_v7_move_is_kept_and_nothing_else_is_checked(self):
        final, diag = self._run({(7, 9): SAFE}, (7, 9), self.CHILDREN)
        self.assertEqual((final, diag.v8_root_rank), ((7, 9), 1))
        self.assertEqual(dict(diag.v8_root_checked), {(7, 9): SAFE})

    def test_unsafe_v7_move_gives_way_to_the_next_proven_safe_child(self):
        # Tree order after V7's move: (6, 6) (20 visits, higher mean) before (8, 9).
        final, diag = self._run({(7, 9): UNSAFE, (6, 6): UNSAFE, (8, 9): SAFE}, (7, 9), self.CHILDREN)
        self.assertEqual((final, diag.v8_root_rank), ((8, 9), 3))

    def test_unrefuted_v7_move_stands_without_a_proven_safe_child(self):
        final, _ = self._run({(7, 9): UNKNOWN, (6, 6): UNSAFE}, (7, 9), self.CHILDREN)
        self.assertEqual(final, (7, 9))

    def test_proven_loss_gives_way_to_an_unrefuted_child(self):
        final, diag = self._run({(7, 9): UNSAFE, (6, 6): UNSAFE}, (7, 9), self.CHILDREN)
        self.assertEqual((final, diag.v8_root_rank), ((8, 9), 3))

    def test_all_proven_unsafe_keeps_v7_move(self):
        script = {c.move: UNSAFE for c in self.CHILDREN}
        final, _ = self._run(script, (7, 9), self.CHILDREN)
        self.assertEqual(final, (7, 9))

    def test_capped_children_widen_when_all_are_proven_unsafe(self):
        # max_children=2: (7, 9) and (6, 6) are both proven UNSAFE, so the check must go on to
        # the children beyond the cap instead of keeping V7's proven loss.
        from unittest import mock
        game = _game(GAMES[1]['moves'][:4])
        diag = SearchDiagnostics()
        script = {(7, 9): UNSAFE, (6, 6): UNSAFE, (8, 9): SAFE}
        with mock.patch.object(_BudgetedSolver, 'status_after', _scripted_status(script)):
            final = _verify_root_choice(game, (7, 9), self.CHILDREN, diag, node_limit=20_000,
                                        call_limit=10_000, node_budget=200_000, max_children=2)
        self.assertEqual((final, diag.v8_root_rank), ((8, 9), 3))

    def test_capped_fallback_considers_unchecked_children(self):
        from unittest import mock
        game = _game(GAMES[1]['moves'][:4])
        diag = SearchDiagnostics()
        script = {(7, 9): UNSAFE, (6, 6): UNSAFE, (8, 9): UNSAFE, (9, 9): UNSAFE}
        def status_after(self, game, move, share=None, call_share=None):
            self.last_cut = False
            return script[move] if move in ((7, 9), (6, 6)) else UNKNOWN  # beyond the cap: unresolved
        with mock.patch.object(_BudgetedSolver, 'status_after', status_after):
            final = _verify_root_choice(game, (7, 9), self.CHILDREN, diag, node_limit=20_000,
                                        call_limit=10_000, node_budget=200_000, max_children=2)
        self.assertNotIn(final, ((7, 9), (6, 6)))

    def test_proven_collects_several_in_proof_order(self):
        solver = _ScriptedSolver({'A': (10, SAFE), 'B': (3_000, SAFE), 'C': (20, SAFE)}, 10_000)  # round-1 share 1,666
        self.assertEqual(_proven(None, ['A', 'B', 'C'], {}, solver, solver.status_after, SAFE, want=2),
                         ['A', 'C'])


class RootVetoPolicyTest(unittest.TestCase):
    # root_vct_mode="veto": only a proven loss of V7's move switches (design §12.3).
    # (6, 6) is occupied in this position, so the veto tests (which reach the at-once-loss guard) use (6, 8).
    CHILDREN = [_Child((7, 9), 30), _Child((8, 9), 20), _Child((6, 8), 20, 0.5), _Child((9, 9), 5)]

    def _run(self, script, chosen=(7, 9), mode='veto'):
        from unittest import mock
        game = _game(GAMES[1]['moves'][:4])
        diag = SearchDiagnostics()
        with mock.patch.object(_BudgetedSolver, 'status_after', _scripted_status(script)):
            final = _verify_root_choice(game, chosen, self.CHILDREN, diag, node_limit=20_000,
                                        call_limit=10_000, node_budget=200_000, max_children=4, mode=mode)
        return final, diag

    def test_unknown_v7_move_is_kept_in_veto_mode_only(self):
        script = {(7, 9): UNKNOWN, (6, 8): SAFE}
        final, diag = self._run(script)
        self.assertEqual((final, diag.v8_root_switch), ((7, 9), ''))
        self.assertEqual(dict(diag.v8_root_checked), {(7, 9): UNKNOWN})  # nothing else is checked
        final, diag = self._run(script, mode='aggressive')
        self.assertEqual((final, diag.v8_root_switch), ((6, 8), 'unknown'))

    def test_proven_loss_switches_to_best_ranked_unrefuted_child(self):
        # (6, 8) is UNKNOWN: in veto mode it is not vetoed, so it is played (rank 2),
        # even though (8, 9) further down is proven SAFE. The children are checked as in
        # aggressive mode (which plays (8, 9)), so both modes see the same statuses.
        script = {(7, 9): UNSAFE, (6, 8): UNKNOWN, (8, 9): SAFE}
        final, diag = self._run(script)
        self.assertEqual((final, diag.v8_root_rank, diag.v8_root_switch), ((6, 8), 2, 'proven_loss'))
        aggressive, other = self._run(script, mode='aggressive')
        self.assertEqual(aggressive, (8, 9))
        self.assertEqual(dict(diag.v8_root_checked), dict(other.v8_root_checked))

    def test_vetoed_children_are_skipped(self):
        final, diag = self._run({(7, 9): UNSAFE, (6, 8): UNSAFE, (8, 9): SAFE})
        self.assertEqual((final, diag.v8_root_rank), ((8, 9), 3))

    def test_all_proven_losses_keep_v7_move(self):
        final, diag = self._run({c.move: UNSAFE for c in self.CHILDREN})
        self.assertEqual((final, diag.v8_root_switch), ((7, 9), ''))

    def test_proven_safe_v7_move_is_kept(self):
        final, diag = self._run({(7, 9): SAFE})
        self.assertEqual((final, diag.v8_root_rank), ((7, 9), 1))

    def test_engine_veto_mode_runs_and_is_validated(self):
        game = _game(GAMES[1]['moves'][:4])
        expected = mcts_search_v7(game, **V7_FINAL, random=Random(4))
        diag = SearchDiagnostics()
        config = {**V8_DEFAULTS, 'own_vct_attack': False, 'root_vct_mode': 'veto'}
        self.assertEqual(mcts_search_v8(game, **config, random=Random(4), diagnostics=diag), expected)
        with self.assertRaises(ValueError):
            MCTSV8Agent(root_vct_mode='bogus')


class RootVerifyEngineTest(unittest.TestCase):
    def test_safe_tree_move_is_v7s_move(self):
        # Quiet position: V7's tree move is proven VCT1-SAFE, so V8-C changes nothing.
        game = _game(GAMES[1]['moves'][:4])
        expected = mcts_search_v7(game, **V7_FINAL, random=Random(4))
        diag = SearchDiagnostics()
        move = mcts_search_v8(game, **{**V8_DEFAULTS, 'own_vct_attack': False}, random=Random(4), diagnostics=diag)
        self.assertEqual((move, diag.v8_v7_move, diag.v8_changed, diag.v8_root_rank), (expected, expected, False, 1))
        self.assertEqual(dict(diag.v8_root_checked), {expected: SAFE})


class AgentTest(unittest.TestCase):
    def test_agent_matches_v7_agent_with_modules_off(self):
        game = _game(GAMES[0]['moves'][:6])
        v8 = MCTSV8Agent(seed=5, stage_vct_safety=False, own_vct_attack=False, root_vct_safety=False)
        v7 = MCTSV7Agent(seed=5)
        self.assertEqual(v8.select_move(game), v7.select_move(game))

    def test_rejects_bad_options(self):
        with self.assertRaises(TypeError):
            MCTSV8Agent(unknown_option=1)
        with self.assertRaises(ValueError):
            MCTSV8Agent(vct_call_limit=0)
        with self.assertRaises(ValueError):
            MCTSV8Agent(vct_node_budget=0)
        with self.assertRaises(ValueError):
            MCTSV8Agent(stage_vct_safety=1)
        with self.assertRaises(ValueError):
            MCTSV8Agent(own_vct_attack=1)
        with self.assertRaises(ValueError):
            MCTSV8Agent(attack_node_budget=0)
        with self.assertRaises(ValueError):
            MCTSV8Agent(root_vct_safety=1)
        with self.assertRaises(ValueError):
            MCTSV8Agent(root_node_budget=0)


if __name__ == '__main__':
    unittest.main()
