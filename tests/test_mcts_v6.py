from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from renju import BLACK, WHITE, Game
from search.mcts_v5 import _RootContext, _forced_v5_move
from search.mcts_v6 import SearchDiagnostics, V5_FINAL, _root_candidates_v6, mcts_search_v6
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

    def test_white_win_precedes_black_43_defense(self):
        game = position([(2,c) for c in range(4)], WHITE, cross())
        diag = SearchDiagnostics()
        self.assertEqual(mcts_search_v6(game, diagnostics=diag), (2,4))
        self.assertEqual(diag.forced_policy_stage, 1)
        self.assertEqual(diag.v6_root_injection_count, 0)


if __name__ == '__main__':
    unittest.main()
