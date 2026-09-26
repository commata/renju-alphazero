import json
from copy import deepcopy
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from agents import MCTSV7Agent
from renju import BLACK, WHITE, Game
from search.mcts_v6 import V5_FINAL
from search.mcts_v5 import _RootContext
from search.mcts_v7 import (V7_FINAL, SearchDiagnostics, _VCFBudget,
                            _apply_self_forbidden_penalty, _apply_vcf_safety,
                            _candidate_allows_vcf, _stage4_v7_move,
                            find_vcf, mcts_search_v7)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mcts_v7_positions.json"
REGRESSION_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mcts_v7_vcf_regression.json"


def _coord(value):
    return value[0] - 1, value[1] - 1


def _load_fixtures():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert data["format"] == "mcts-v7-fixtures-v1"
    return {item["id"]: item for item in data["fixtures"]}


FIXTURES = _load_fixtures()


def _load_regression_fixtures():
    data = json.loads(REGRESSION_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert data["format"] == "mcts-v7-vcf-regression-v1"
    return data["fixtures"]


REGRESSION_FIXTURES = _load_regression_fixtures()


def _game(item):
    game = Game()
    for move in item["moves"]:
        game.play(*_coord(move))
    expected = BLACK if item["to_play"] == "BLACK" else WHITE
    if game.to_play != expected:
        raise AssertionError((item["id"], game.to_play, expected))
    return game


class V7ConfigTest(unittest.TestCase):
    def test_final_config_extends_v6_without_changing_v6_values(self):
        for key, value in V5_FINAL.items():
            self.assertEqual(V7_FINAL[key], value)
        self.assertEqual(
            {key: V7_FINAL[key] for key in V7_FINAL if key not in V5_FINAL},
            dict(
                own_vcf_max_fours=10,
                own_vcf_node_limit=5000,
                safety_vcf_max_fours=10,
                safety_vcf_node_limit=4000,
                safety_precheck_node_limit=8000,
                safety_total_node_limit=8000,
                self_forbidden_min_white=3,
            ),
        )

    def test_agent_defaults_and_invalid_v7_config(self):
        agent = MCTSV7Agent()
        self.assertEqual(agent.name, "MCTS-v7")
        for key, value in V7_FINAL.items():
            self.assertEqual(getattr(agent, key), value)
        for options in (
            dict(own_vcf_max_fours=0),
            dict(own_vcf_node_limit=True),
            dict(safety_vcf_node_limit=0),
            dict(safety_precheck_node_limit=0),
            dict(safety_total_node_limit=0),
            dict(self_forbidden_min_white=5),
        ):
            with self.assertRaises(ValueError):
                MCTSV7Agent(**options)


class VCFTest(unittest.TestCase):
    def test_own_vcf_fixtures_detected_and_state_restored(self):
        for item in FIXTURES.values():
            if item["kind"] != "own_vcf":
                continue
            game = _game(item)
            before = deepcopy(vars(game))
            result = find_vcf(
                game,
                game.to_play,
                max_fours=V7_FINAL["own_vcf_max_fours"],
                node_limit=V7_FINAL["own_vcf_node_limit"],
            )
            self.assertIsNotNone(result, item["id"])
            expected = {_coord(move) for move in item["expected"]["vcf_first_moves"]}
            self.assertIn(result.first_move, expected, item["id"])
            self.assertEqual(vars(game), before, item["id"])

    def test_regression_124_positions_match_reference(self):
        counts = {}
        for item in REGRESSION_FIXTURES:
            counts[item["kind"]] = counts.get(item["kind"], 0) + 1
            game = _game(item)
            if item["kind"] == "losing_move_allows_vcf":
                game.play(*_coord(item["losing_move"]))
                attacker = game.to_play
                node_limit = V7_FINAL["safety_vcf_node_limit"]
            else:
                attacker = game.to_play
                node_limit = V7_FINAL["own_vcf_node_limit"]

            before = deepcopy(vars(game))
            result = find_vcf(
                game, attacker, max_fours=10, node_limit=node_limit,
            )
            self.assertIsNotNone(result, item["id"])
            self.assertEqual(
                result.fours, item["expected"]["reference_fours"], item["id"],
            )
            if "reference_first_move" in item["expected"]:
                self.assertEqual(
                    result.first_move,
                    _coord(item["expected"]["reference_first_move"]),
                    item["id"],
                )
            self.assertEqual(
                result.nodes, item["expected"]["reference_nodes"], item["id"],
            )
            self.assertEqual(vars(game), before, item["id"])

        self.assertEqual(
            counts,
            dict(
                vcf_streak_start=27,
                missed_own_vcf=79,
                losing_move_allows_vcf=18,
            ),
        )

    def test_node_limit_and_argument_validation(self):
        game = _game(FIXTURES["seed44-g005-p025"])
        self.assertIsNone(find_vcf(game, game.to_play, max_fours=10, node_limit=1))
        for kwargs in (
            dict(attacker=0, max_fours=10, node_limit=100),
            dict(attacker=game.to_play, max_fours=0, node_limit=100),
            dict(attacker=game.to_play, max_fours=10, node_limit=0),
        ):
            with self.assertRaises(ValueError):
                find_vcf(game, **kwargs)


class V7FixtureTest(unittest.TestCase):
    def test_stage3v_selects_vcf_first_move(self):
        for item in FIXTURES.values():
            if item["kind"] != "own_vcf":
                continue
            game = _game(item)
            expected = {_coord(move) for move in item["expected"]["vcf_first_moves"]}
            diag = SearchDiagnostics()
            before = deepcopy(vars(game))
            move = mcts_search_v7(
                game, simulations=1, tactical_simulations=1,
                random=random.Random(17), diagnostics=diag,
            )
            self.assertIn(move, expected, item["id"])
            self.assertTrue(diag.v7_own_vcf_found, item["id"])
            self.assertGreater(diag.v7_own_vcf_nodes, 0, item["id"])
            self.assertEqual(vars(game), before, item["id"])

    def test_safety_fixtures_do_not_leave_opponent_vcf(self):
        for item in FIXTURES.values():
            if item["kind"] != "opponent_vcf_safety":
                continue
            game = _game(item)
            player = game.to_play
            opponent = -player
            diag = SearchDiagnostics()
            move = mcts_search_v7(
                game, simulations=1, tactical_simulations=1,
                random=random.Random(23), diagnostics=diag,
            )
            self.assertGreater(
                diag.v7_safety_checked + diag.v7_safety_precheck_skipped,
                0, item["id"],
            )
            budget = _VCFBudget(V7_FINAL["safety_total_node_limit"])
            unsafe, _, inconclusive = _candidate_allows_vcf(
                game,
                move,
                player,
                opponent,
                max_fours=V7_FINAL["safety_vcf_max_fours"],
                node_limit=V7_FINAL["safety_vcf_node_limit"],
                budget=budget,
            )
            self.assertFalse(inconclusive, item["id"])
            self.assertFalse(unsafe, item["id"])

    def test_safety_probe_evaluates_unique_forced_four_reply(self):
        item = next(
            row for row in REGRESSION_FIXTURES
            if row["id"] == "safety-seed44-g020-p055"
        )
        game = _game(item)
        player = game.to_play
        opponent = -player
        losing = _coord(item["losing_move"])
        budget = _VCFBudget(V7_FINAL["safety_total_node_limit"])
        unsafe, result, inconclusive = _candidate_allows_vcf(
            game,
            losing,
            player,
            opponent,
            max_fours=V7_FINAL["safety_vcf_max_fours"],
            node_limit=V7_FINAL["safety_vcf_node_limit"],
            budget=budget,
        )
        self.assertFalse(inconclusive)
        self.assertTrue(unsafe)
        self.assertIsNotNone(result)

    def test_safety_inconclusive_is_demoted_not_removed(self):
        game = Game()
        game.play(7, 7)  # WHITE
        moves = [(6, 7), (7, 6), (8, 7)]
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        diag = SearchDiagnostics()

        with patch("search.mcts_v7._has_four_material", return_value=False), \
             patch("search.mcts_v7._candidate_changes_black_legality",
                   return_value=True), \
             patch("search.mcts_v7._candidate_allows_vcf",
                   side_effect=[
                       (False, None, True),   # inconclusive
                       (False, None, False),  # verified safe
                       (True, object(), False),  # confirmed VCF
                   ]):
            ranked = _apply_vcf_safety(
                game,
                context,
                moves,
                diag,
                max_fours=10,
                node_limit=4000,
                precheck_node_limit=8000,
                total_node_limit=8000,
            )

        self.assertEqual(ranked, [moves[1], moves[0]])
        self.assertEqual(diag.v7_safety_inconclusive, 1)
        self.assertEqual(diag.v7_safety_removed, 1)
        self.assertFalse(diag.v7_safety_fallback)

    def test_safety_precheck_uses_higher_cap_and_skips_quiet_candidates(self):
        game = Game()
        game.play(7, 7)  # WHITE
        moves = [(6, 7), (7, 6), (8, 7)]
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        diag = SearchDiagnostics()

        def root_probe(_game, _attacker, *, max_fours, node_limit, budget):
            self.assertEqual(max_fours, 10)
            self.assertEqual(node_limit, 8000)
            self.assertEqual(budget.remaining, 8000)
            budget.used += 5498
            budget.remaining -= 5498
            return None, False

        with patch("search.mcts_v7._has_four_material", return_value=True), \
             patch("search.mcts_v7._budgeted_find_vcf",
                   side_effect=root_probe), \
             patch("search.mcts_v7._four_completions", return_value=[]), \
             patch("search.mcts_v7._candidate_changes_black_legality",
                   return_value=False):
            ranked = _apply_vcf_safety(
                game,
                context,
                moves,
                diag,
                max_fours=10,
                node_limit=4000,
                precheck_node_limit=8000,
                total_node_limit=8000,
            )

        self.assertEqual(ranked, moves)
        self.assertEqual(diag.v7_safety_precheck_nodes, 5498)
        self.assertEqual(diag.v7_safety_precheck_skipped, len(moves))
        self.assertEqual(diag.v7_safety_checked, 0)
        self.assertEqual(diag.v7_safety_nodes, 5498)

    def test_self_forbidden_fixture_avoids_logged_move(self):
        item = FIXTURES["seed777-g003-p031"]
        game = _game(item)
        avoid = _coord(item["expected"]["avoid"])
        diag = SearchDiagnostics()
        move = mcts_search_v7(
            game, simulations=1, tactical_simulations=1,
            random=random.Random(31), diagnostics=diag,
        )
        self.assertNotEqual(move, avoid)

        # M1/M2 precede M3 in the V7 decision flow, so the end-to-end move may
        # avoid the logged move before M3 runs. Test M3 itself on the recorded
        # V6 root ordering to prove the logged move is demoted when reached.
        root_moves = [_coord(value) for value in item["root_candidates_in_log"]]
        module_diag = SearchDiagnostics()
        ranked = _apply_self_forbidden_penalty(
            game, root_moves, module_diag,
            minimum_white=V7_FINAL["self_forbidden_min_white"],
        )
        self.assertGreater(module_diag.v7_self_forbidden_penalized, 0)
        penalized_start = len(ranked) - module_diag.v7_self_forbidden_penalized
        self.assertGreaterEqual(ranked.index(avoid), penalized_start)

    def test_stage4_fixture_records_no_discriminating_m4_choice(self):
        # The supplied seed777 fixture motivated M4, but replay audit shows all
        # three legal Stage-4 defenses leave the same opponent 43/VCF. M4 has no
        # evidence-backed discriminator here, so V7 must preserve V6's choice.
        item = FIXTURES["seed777-g003-p021"]
        game = _game(item)
        diag = SearchDiagnostics()
        move = mcts_search_v7(
            game, simulations=1, tactical_simulations=1,
            random=random.Random(41), diagnostics=diag,
        )
        self.assertEqual(move, _coord(item["actual_move_in_log"]))
        self.assertEqual(diag.forced_policy_stage, 4)
        self.assertFalse(diag.v7_stage4_tiebreak_applied)

    def test_stage4_tiebreak_applies_when_secondary_signal_exists(self):
        game = Game()
        game.play(7, 7)  # WHITE to play; candidate cells remain empty.
        first, second = (6, 7), (7, 6)
        context = _RootContext([first, second], SearchDiagnostics())
        diag = SearchDiagnostics()

        # Initial call finds the opponent creator; the next two calls are the
        # equal Stage-4 "remaining unstoppable" values for first/second.
        with patch("search.mcts_v7._unstoppable_four_moves",
                   side_effect=[[(5, 5)], [], []]), \
             patch("search.mcts_v7._threat_windows",
                   return_value=[((5, 5), first, second, (5, 6), (5, 7))]), \
             patch("search.mcts_v7._double_threat_moves",
                   side_effect=[[(3, 3)], []]), \
             patch("search.mcts_v7._budgeted_find_vcf",
                   return_value=(None, False)):
            chosen = _stage4_v7_move(
                game, context, first, diag, max_fours=10, node_limit=1000,
                total_node_limit=8000,
            )

        self.assertEqual(chosen, second)
        self.assertTrue(diag.v7_stage4_tiebreak_applied)

    def test_stage4_vcf_checks_share_total_budget_and_demote_inconclusive(self):
        game = Game()
        game.play(7, 7)  # WHITE
        first, second, third = (6, 7), (7, 6), (8, 7)
        context = _RootContext([first, second, third], SearchDiagnostics())
        diag = SearchDiagnostics()
        seen_budget_ids = []

        def probe(_game, _attacker, *, max_fours, node_limit, budget):
            seen_budget_ids.append(id(budget))
            use = min(4000, budget.remaining)
            budget.used += use
            budget.remaining -= use
            if budget.remaining == 0:
                budget.exhausted = True
            # first verified safe; later probes become inconclusive under cap
            return (None, False) if len(seen_budget_ids) == 1 else (None, True)

        with patch("search.mcts_v7._unstoppable_four_moves",
                   side_effect=[[(5, 5)], [], [], []]), \
             patch("search.mcts_v7._threat_windows",
                   return_value=[((5, 5), first, second, third, (5, 6))]), \
             patch("search.mcts_v7._double_threat_moves",
                   side_effect=[[], [], []]), \
             patch("search.mcts_v7._budgeted_find_vcf", side_effect=probe):
            chosen = _stage4_v7_move(
                game, context, second, diag, max_fours=10, node_limit=4000,
                total_node_limit=8000,
            )

        self.assertEqual(chosen, first)
        self.assertEqual(len(set(seen_budget_ids)), 1)
        self.assertEqual(diag.v7_stage4_vcf_nodes, 8000)
        self.assertEqual(diag.v7_stage4_vcf_inconclusive, 2)
        self.assertTrue(diag.v7_stage4_vcf_budget_exhausted)

    def test_seeded_determinism_and_state(self):
        game = _game(FIXTURES["seed44-g009-p039"])
        before = deepcopy(vars(game))
        moves = []
        diagnostics = []
        for _ in range(2):
            diag = SearchDiagnostics()
            moves.append(mcts_search_v7(
                game, simulations=2, tactical_simulations=2,
                random=random.Random(99), diagnostics=diag,
            ))
            diagnostics.append(diag)
        self.assertEqual(moves[0], moves[1])
        comparable = [
            (d.v7_own_vcf_found, d.v7_safety_checked, d.v7_safety_removed,
             d.v7_safety_augmented, d.v7_safety_fallback, d.v7_safety_nodes,
             d.v7_safety_precheck_nodes, d.v7_safety_precheck_skipped,
             d.v7_safety_inconclusive, d.v7_safety_budget_exhausted,
             d.v7_self_forbidden_penalized, d.v7_stage4_tiebreak_applied,
             d.v7_stage4_vcf_nodes, d.v7_stage4_vcf_inconclusive,
             d.v7_stage4_vcf_budget_exhausted, d.root_candidates)
            for d in diagnostics
        ]
        self.assertEqual(comparable[0], comparable[1])
        self.assertEqual(vars(game), before)


if __name__ == "__main__":
    unittest.main()
