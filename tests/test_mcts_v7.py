import json
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents import MCTSV7Agent
from renju import BLACK, WHITE, Game
from search.threat_patterns import placed
from search.mcts_v6 import V5_FINAL
from search.mcts_v5 import _RootContext
from search.mcts_v7 import (V7_FINAL, SearchDiagnostics, _VCFBudget,
                            _apply_self_forbidden_penalty, _apply_vcf_safety,
                            _candidate_allows_vcf, _penalize_within_tiers,
                            _stage4_v7_move, find_vcf, mcts_search_v7)


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

    def test_stage4_probes_only_best_tie_group_and_stops_at_first_safe(self):
        game = Game()
        game.play(7, 7)  # WHITE
        a, b, c, d = (6, 7), (7, 6), (8, 7), (7, 8)
        # Root key order is the move itself, so the probe order is a, b, c.
        context = SimpleNamespace(legal=[a, b, c, d], key=lambda _g, m: m)
        diag = SearchDiagnostics()
        probed = []

        def probe(_game, move, _player, _opponent, *, max_fours, node_limit,
                  budget):
            probed.append(move)
            budget.used += 100
            budget.remaining -= 100
            return {a: (False, None, True), b: (False, None, False)}[move]

        with patch("search.mcts_v7._unstoppable_four_moves",
                   side_effect=[[(5, 5)], [], [], [], []]), \
             patch("search.mcts_v7._threat_windows",
                   return_value=[((5, 5), a, b, c, d)]), \
             patch("search.mcts_v7._double_threat_moves",
                   side_effect=[[], [], [], [(3, 3)]]), \
             patch("search.mcts_v7._candidate_allows_vcf",
                   side_effect=probe):
            chosen = _stage4_v7_move(
                game, context, a, diag, max_fours=10, node_limit=4000,
                total_node_limit=8000,
            )

        # d is outside the best (remaining, double-threat) group and c comes
        # after the first verified-safe defense, so neither is probed.
        self.assertEqual(probed, [a, b])
        self.assertEqual(chosen, b)
        self.assertTrue(diag.v7_stage4_tiebreak_applied)
        self.assertEqual(diag.v7_stage4_vcf_nodes, 200)
        self.assertEqual(diag.v7_stage4_vcf_inconclusive, 1)
        self.assertFalse(diag.v7_stage4_vcf_budget_exhausted)

    def test_stage4_single_best_defense_skips_vcf_probes(self):
        game = Game()
        game.play(7, 7)  # WHITE
        a, b = (6, 7), (7, 6)
        context = SimpleNamespace(legal=[a, b], key=lambda _g, m: m)
        diag = SearchDiagnostics()
        with patch("search.mcts_v7._unstoppable_four_moves",
                   side_effect=[[(5, 5)], [], []]), \
             patch("search.mcts_v7._threat_windows",
                   return_value=[((5, 5), a, b, (5, 6), (5, 7))]), \
             patch("search.mcts_v7._double_threat_moves",
                   side_effect=[[(3, 3)], []]), \
             patch("search.mcts_v7._candidate_allows_vcf") as probe:
            chosen = _stage4_v7_move(
                game, context, a, diag, max_fours=10, node_limit=4000,
                total_node_limit=8000,
            )
        probe.assert_not_called()
        self.assertEqual(chosen, b)
        self.assertEqual(diag.v7_stage4_vcf_nodes, 0)

    def test_stage4_evaluates_four_defense_after_forced_block(self):
        # A raw VCF probe right after a four-making move misses the opponent
        # VCF (every attacker four is refuted by our pending win). M4 must use
        # the M2 forced-reply probe instead.
        item = next(
            row for row in REGRESSION_FIXTURES
            if row["id"] == "streak-seed44-g010-p027"
        )
        game = _game(item)
        player, opponent = game.to_play, -game.to_play
        four = (7, 4)
        with placed(game, player, four):
            raw = find_vcf(game, opponent, max_fours=10, node_limit=4000)
        unsafe, _, inconclusive = _candidate_allows_vcf(
            game, four, player, opponent, max_fours=10, node_limit=4000,
            budget=_VCFBudget(8000),
        )
        self.assertIsNone(raw)
        self.assertTrue(unsafe)
        self.assertFalse(inconclusive)

        other = next(m for m in game.legal_moves() if m != four)
        # The four-making defense comes first in root-key order, so it is
        # always probed and must be ranked as a confirmed VCF.
        context = SimpleNamespace(legal=[four, other],
                                  key=lambda _g, m: (m != four, m))
        diag = SearchDiagnostics()
        with patch("search.mcts_v7._unstoppable_four_moves",
                   side_effect=[[(5, 5)], [], []]), \
             patch("search.mcts_v7._threat_windows",
                   return_value=[((5, 5), four, other, (5, 6), (5, 7))]), \
             patch("search.mcts_v7._double_threat_moves",
                   side_effect=[[], []]), \
             patch("search.mcts_v7._candidate_allows_vcf",
                   wraps=_candidate_allows_vcf) as probe:
            _stage4_v7_move(
                game, context, other, diag, max_fours=10, node_limit=4000,
                total_node_limit=8000,
            )
        self.assertEqual(probe.call_args_list[0].args[1], four)
        self.assertEqual(diag.v7_stage4_vcf_inconclusive, 0)

    def test_safety_exhausted_budget_skips_augmentation(self):
        game = Game()
        game.play(7, 7)  # WHITE
        moves = [(6, 7), (7, 6), (8, 7)]
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        diag = SearchDiagnostics()

        def exhausted(_game, _move, _player, _opponent, *, max_fours,
                      node_limit, budget):
            budget.used += budget.remaining
            budget.remaining = 0
            budget.exhausted = True
            return False, None, True

        with patch("search.mcts_v7._has_four_material", return_value=False), \
             patch("search.mcts_v7._candidate_changes_black_legality",
                   return_value=True), \
             patch("search.mcts_v7._candidate_allows_vcf",
                   side_effect=exhausted), \
             patch("search.mcts_v7._own_four_creators") as creators:
            ranked = _apply_vcf_safety(
                game, context, moves, diag, max_fours=10, node_limit=4000,
                precheck_node_limit=8000, total_node_limit=8000,
            )
        creators.assert_not_called()
        self.assertEqual(ranked, moves)
        self.assertFalse(diag.v7_safety_augmented)
        self.assertTrue(diag.v7_safety_budget_exhausted)

    def test_safety_unverified_augmentation_is_not_added(self):
        game = Game()
        game.play(7, 7)  # WHITE
        moves = [(6, 7), (7, 6)]
        extra = (0, 0)
        context = _RootContext(game.legal_moves(), SearchDiagnostics())

        def run(extra_status):
            diag = SearchDiagnostics()
            statuses = {moves[0]: (False, None, True),
                        moves[1]: (False, None, True),
                        extra: extra_status}
            with patch("search.mcts_v7._has_four_material",
                       return_value=False), \
                 patch("search.mcts_v7._candidate_changes_black_legality",
                       return_value=True), \
                 patch("search.mcts_v7._candidate_allows_vcf",
                       side_effect=lambda _g, m, *_a, **_k: statuses[m]), \
                 patch("search.mcts_v7._own_four_creators",
                       return_value={extra}):
                ranked = _apply_vcf_safety(
                    game, context, moves, diag, max_fours=10,
                    node_limit=4000, precheck_node_limit=8000,
                    total_node_limit=8000,
                )
            return ranked, diag

        ranked, diag = run((False, None, True))
        self.assertEqual(ranked, moves)
        self.assertFalse(diag.v7_safety_augmented)

        ranked, diag = run((False, None, False))
        # Verified-safe augmentation leads, total stays at the V6 count.
        self.assertEqual(ranked, [extra, moves[0]])
        self.assertTrue(diag.v7_safety_augmented)

    def test_self_forbidden_penalty_stays_inside_m2_tiers(self):
        game = Game()
        safe_penalized, safe_normal = (1, 1), (1, 2)
        unverified = (2, 2)

        def risky(_game, _minimum_white):
            # Only the safe_penalized move creates a new risky point.
            return {(9, 9)} if _game.board[1][1] == BLACK else set()

        diag = SearchDiagnostics()
        with patch("search.mcts_v7._risky_black_forbidden_points",
                   side_effect=risky):
            ranked = _penalize_within_tiers(
                game, [[safe_penalized, safe_normal], [unverified]], diag,
                minimum_white=3,
            )
        self.assertEqual(ranked, [safe_normal, safe_penalized, unverified])
        self.assertEqual(diag.v7_self_forbidden_penalized, 1)

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
