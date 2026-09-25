from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from renju import BLACK, WHITE, Game
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import (SearchDiagnostics, V5_FINAL, _black_43_defense_risk,
                            _root_candidates_v6, mcts_search_v6)
from test_threat_patterns import position, cross


class V6SearchTest(unittest.TestCase):
    def test_injection_preserved_above_limit(self):
        game = position(cross(kind='33') + cross(kind='33', center=(3,3)), WHITE)
        diag = SearchDiagnostics()
        context = _RootContext(game.legal_moves(), diag)
        moves, _, reasons = _root_candidates_v6(game, context, 1, 2)
        self.assertIn((7,7), moves)
        self.assertIn('white_33', reasons[(7,7)])
        self.assertGreater(len(moves), 1)

    def test_forced_stages_match_v5_without_planner(self):
        cases = [position([(7,c) for c in range(4)]),
                 position(opponents=[(7,c) for c in range(4)]),
                 position([(7,6),(7,7),(7,8)]),
                 position(opponents=[(7,6),(7,7),(7,8)]),
                 position(opponents=cross(kind='33'))]
        for expected_stage, game in enumerate(cases, 1):
            diag = SearchDiagnostics()
            expected = _forced_v5_move(game)
            with patch('search.mcts_v6.plan_root', side_effect=AssertionError('forced policy first')):
                move = mcts_search_v6(game, diagnostics=diag)
            self.assertEqual(move, expected)
            self.assertEqual(diag.forced_policy_stage, expected_stage)
            self.assertEqual(diag.selected_simulations, 0)

    def test_determinism_and_state(self):
        game = Game()
        game.play(7,7)
        before, rng = deepcopy(vars(game)), random.getstate()
        moves = [mcts_search_v6(game, simulations=1, tactical_simulations=1,
                                random=random.Random(123)) for _ in range(2)]
        self.assertEqual(moves[0], moves[1])
        self.assertIn(moves[0], game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), rng)

    def test_final_parameters(self):
        self.assertEqual(V5_FINAL, dict(simulations=50, tactical_simulations=100,
                         tactical_score_threshold=1800, exploration=2**0.5,
                         candidate_limit=20, initial_width=8, neighborhood_radius=2, priority_top_k=8))

    def test_white_black_43_defense_injection(self):
        game = position(opponents=cross(), player=WHITE)
        diag = SearchDiagnostics()
        context = _RootContext(game.legal_moves(), diag)
        moves, _, reasons = _root_candidates_v6(game, context, 1, 2)
        self.assertIn((7,7), moves)
        self.assertIn('black_43_defense', reasons[(7,7)])
        self.assertGreater(diag.black_43_defense_candidates, 1)
        self.assertEqual(diag.black_43_defense_candidates, diag.black_43_defense_injections)
        self.assertTrue(set(reasons).issubset(context.legal))
        self.assertIsNone(diag.forced_policy_stage)

    def test_black_43_defense_risk_counts_uncovered_creator(self):
        threats = cross(center=(4,4)) + cross(center=(10,10))
        game = position(opponents=threats, player=WHITE)
        immediate, _ = _black_43_defense_risk(game, (4,4))
        self.assertGreaterEqual(immediate, 1)

    def test_complete_black_43_defense_ranks_before_partial_defense(self):
        game = position(opponents=cross(), player=WHITE)
        diag = SearchDiagnostics()
        context = _RootContext(game.legal_moves(), diag)

        def risk(_game, move):
            return (0, 0) if move == (7,7) else (1, 0)

        with patch('search.mcts_v6._black_43_defense_risk', side_effect=risk):
            moves, _, reasons = _root_candidates_v6(game, context, 1, 2)
        defenses = [m for m in moves if 'black_43_defense' in reasons.get(m, ())]
        self.assertTrue(defenses)
        self.assertEqual(defenses[0], (7,7))
        self.assertGreater(diag.black_43_defense_complete_candidates, 0)
        self.assertEqual(diag.black_43_defense_min_immediate_remaining, 0)

    def test_white_win_precedes_black_43_defense(self):
        game = position([(2,c) for c in range(4)], WHITE, cross())
        diag = SearchDiagnostics()
        self.assertEqual(mcts_search_v6(game, diagnostics=diag), (2,4))
        self.assertEqual(diag.forced_policy_stage, 1)
        self.assertEqual(diag.v6_root_injection_count, 0)

    def test_agent_defaults_and_invalid_config(self):
        from agents import MCTSV6Agent
        agent = MCTSV6Agent()
        for key, value in V5_FINAL.items():
            self.assertEqual(getattr(agent, key), value)
        for options in (dict(simulations=0), dict(tactical_simulations=True),
                        dict(exploration=float('nan')), dict(initial_width=21)):
            with self.assertRaises(ValueError):
                MCTSV6Agent(**options)

    def test_nonempty_planner_determinism_and_diagnostics_reset(self):
        from agents import MCTSV6Agent
        game = position(cross(kind='33'), WHITE)
        before, rng = deepcopy(vars(game)), random.getstate()
        first = MCTSV6Agent(seed=22, simulations=2, tactical_simulations=2)
        second = MCTSV6Agent(seed=22, simulations=2, tactical_simulations=2)
        self.assertEqual(first.select_move(game), second.select_move(game))
        self.assertGreater(first.diagnostics.v6_root_injection_count, 0)
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), rng)
        forced = position([(2,c) for c in range(4)], WHITE)
        first.select_move(forced)
        self.assertEqual(first.diagnostics.v6_root_injection_count, 0)
        self.assertIsNone(first.diagnostics.v6_selected_threat_type)

    def test_no_planner_preserves_v5_root_order(self):
        from search.mcts_v5 import _root_candidates_v5
        game = Game()
        context = _RootContext(game.legal_moves(), SearchDiagnostics(), injected=[(0,0),(14,14)])
        expected, score = _root_candidates_v5(game, context, 20, 2)
        moves, observed, _ = _root_candidates_v6(game, context, 20, 2)
        self.assertEqual((moves, observed), (expected, score))

    def test_terminal_raises(self):
        from renju import IllegalMove
        game = Game()
        game.done = True
        with self.assertRaises(IllegalMove):
            mcts_search_v6(game)

    def test_real_multi_danger_reaches_search_with_defense_injection(self):
        from agents import MCTSV6Agent
        stones = [(5,8),(7,8),(8,6),(3,3),(5,3),(10,8),(7,5),(11,10),
                  (7,4),(9,8),(5,9),(10,5),(9,4),(11,5),(3,9)]
        game = position(player=WHITE, opponents=stones)
        before = deepcopy(vars(game))
        self.assertIsNone(_forced_v5_move(game))
        agent = MCTSV6Agent(seed=19, simulations=1, tactical_simulations=1)
        move = agent.select_move(game)
        self.assertIn(move, game.legal_moves())
        self.assertTrue({(6,4),(7,7)}.issubset(agent.diagnostics.root_candidates))
        self.assertGreater(agent.diagnostics.black_43_defense_injections, 0)
        self.assertIsNone(agent.diagnostics.forced_policy_stage)
        self.assertEqual(vars(game), before)


if __name__ == '__main__':
    unittest.main()
