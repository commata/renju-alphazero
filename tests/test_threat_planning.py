from copy import deepcopy
import unittest

from renju import BLACK, WHITE
from search.threat_patterns import compound_at, placed, compound_moves
from search.threat_planning import (future_setups, PlannerLimits, PlanningStats,
                                    forbidden_defense_attacks, white_defense_profile)
from search.mcts_v5 import _RootContext
from search.mcts_v6 import SearchDiagnostics, plan_root
from test_threat_patterns import position


BLACK_SETUP = [(5,10),(9,5),(10,6),(4,7),(8,4),(6,9),(10,9),
               (4,10),(7,10),(4,4),(8,9),(5,5),(6,4)]
WHITE_33_SETUP = [(5,9),(5,10),(6,9),(7,5),(4,7),(7,8),(8,5),(9,4),(8,4)]
WHITE_COMPOUND_SETUP = [(8,5),(9,8),(7,9),(4,4),(10,8),(7,4),(9,5),
                        (5,5),(8,4),(10,9),(7,7),(10,6),(5,8)]


class PlanningTest(unittest.TestCase):
    def test_actual_two_ply_compounds(self):
        cases = [(BLACK, BLACK_SETUP, (7,9), '43'),
                 (WHITE, WHITE_33_SETUP, (9,6), '33'),
                 (WHITE, WHITE_COMPOUND_SETUP, (7,3), '43'),
                 (WHITE, WHITE_COMPOUND_SETUP, (7,3), '44')]
        for player, stones, move, kind in cases:
            with self.subTest(player=player, kind=kind):
                game = position(stones, player)
                before = deepcopy(vars(game))
                self.assertIsNone(compound_at(game, player, move))
                setup = future_setups(game, player)[move]
                self.assertIn(kind, setup.kinds)
                self.assertEqual(setup.legal_defense_count, setup.surviving_response_count)
                self.assertGreater(setup.continuation_count, 0)
                self.assertEqual(vars(game), before)

    def test_black_setup_creator_is_legal_in_engine(self):
        game = position(BLACK_SETUP)
        game.play(7,9)
        # The four has a single forced block, followed by a legal compound.
        game.play(9,9)
        moves = compound_moves(game, BLACK)
        self.assertIn((4,9), moves)
        game.play(4,9)

    def test_white_future_black_prevention_is_non_forcing(self):
        from search.threat_patterns import black_legal_43_moves
        stones = [(6,7),(8,10),(7,8),(7,7),(6,4),(5,6),(10,4),(5,10)]
        game = position(player=WHITE, opponents=stones)
        self.assertEqual(black_legal_43_moves(game), [])
        diag = SearchDiagnostics()
        reasons = plan_root(game, _RootContext(game.legal_moves(), diag))
        self.assertIn('future_black_43_defense', reasons[(9,10)])
        self.assertGreater(diag.future_black_43_setups, 0)
        self.assertIsNone(diag.forced_policy_stage)

    def test_immediate_counter_win_rejects_setup(self):
        game = position(BLACK_SETUP, opponents=[(1,c) for c in range(4)])
        self.assertEqual(future_setups(game, BLACK), {})

    def test_caps_are_observable(self):
        game = position(WHITE_COMPOUND_SETUP, WHITE)
        stats = PlanningStats()
        self.assertEqual(future_setups(game, WHITE, limits=PlannerLimits(defenses=0), stats=stats), {})
        self.assertGreater(stats.defense_cap_skips, 0)
        self.assertGreater(stats.setup_cap_hits, 0)

    def test_forbidden_defense_induction(self):
        game = position([(4,6),(5,6),(6,6)], WHITE,
                        [(7,3),(7,4),(7,5),(7,7),(7,8),(3,6)])
        before = deepcopy(vars(game))
        profiles = forbidden_defense_attacks(game)
        self.assertIn((8,6), profiles)
        profile = profiles[(8,6)]
        self.assertEqual(profile.legal_defense_count, 0)
        self.assertEqual(profile.forbidden_defense_count, 1)
        self.assertEqual(profile.remaining_winning_continuations, 1)
        self.assertEqual(vars(game), before)

    def test_ordinary_closed_four_has_legal_defense(self):
        game = position([(7,4),(7,5),(7,6)], WHITE, [(7,3)])
        profile = white_defense_profile(game, (7,7))
        self.assertEqual(profile.legal_defense_count, 1)
        self.assertEqual(profile.forbidden_defense_count, 0)
        self.assertEqual(profile.remaining_winning_continuations, 0)

    def test_planner_restores_state_on_nested_exception(self):
        from unittest.mock import patch
        game = position(BLACK_SETUP)
        before = deepcopy(vars(game))
        with patch('search.threat_planning._continuations', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                future_setups(game, BLACK)
        self.assertEqual(vars(game), before)

    def test_required_setup_stone_must_participate_in_compound(self):
        from test_threat_patterns import cross
        game = position(cross(), WHITE)
        self.assertIsNotNone(compound_at(game, WHITE, (7,7)))
        self.assertIsNone(compound_at(game, WHITE, (7,7), required_stone=(0,0)))


if __name__ == '__main__':
    unittest.main()
