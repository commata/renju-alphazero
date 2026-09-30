import json
from pathlib import Path
from random import Random
import unittest

from agents import MCTSV7Agent
from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics, _BudgetedSolver, _board_key, mcts_search_v8
from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.threats import SAFE, UNKNOWN
from renju import Game
from search.mcts_v7 import V7_FINAL, mcts_search_v7

ROOT = Path(__file__).resolve().parents[1]
GAMES = json.loads((ROOT / 'tests/fixtures/web_play_v7_human_games_v1.json').read_text(encoding='utf-8'))['games']
PROBES = json.loads((ROOT / 'tests/fixtures/vct_probes_v1.json').read_text(encoding='utf-8'))['probes']
OFF = {**V8_DEFAULTS, 'stage_vct_safety': False}


def _game(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _probe(game_prefix):
    # 163903 ply 13 (v7 black, Stage 4): the cheapest of the three Stage 4 branch points.
    return next(p for p in PROBES if p['kind'] == 'must_defend_vct' and p['symmetry'] == 0
                and p['source']['game'].startswith(game_prefix))


class ModulesOffMatchV7Test(unittest.TestCase):
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
        move = mcts_search_v8(game, **V8_DEFAULTS, random=Random(1), diagnostics=diag)
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
        move = mcts_search_v8(game, **{**V8_DEFAULTS, 'vct_call_limit': 1},
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


class AgentTest(unittest.TestCase):
    def test_agent_matches_v7_agent_with_modules_off(self):
        game = _game(GAMES[0]['moves'][:6])
        v8 = MCTSV8Agent(seed=5, stage_vct_safety=False)
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


if __name__ == '__main__':
    unittest.main()
