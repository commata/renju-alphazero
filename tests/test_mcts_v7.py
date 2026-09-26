import json
from copy import deepcopy
from pathlib import Path
import random
import unittest

from agents import MCTSV7Agent
from renju import BLACK, WHITE, Game
from search.mcts_v6 import V5_FINAL
from search.mcts_v7 import V7_FINAL, SearchDiagnostics, find_vcf, mcts_search_v7


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mcts_v7_positions.json"


def _coord(value):
    return value[0] - 1, value[1] - 1


def _load_fixtures():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert data["format"] == "mcts-v7-fixtures-v1"
    return {item["id"]: item for item in data["fixtures"]}


FIXTURES = _load_fixtures()


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
                own_vcf_node_limit=2000,
                safety_vcf_max_fours=10,
                safety_vcf_node_limit=1000,
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
            result = find_vcf(game, game.to_play, max_fours=10, node_limit=2000)
            self.assertIsNotNone(result, item["id"])
            expected = {_coord(move) for move in item["expected"]["vcf_first_moves"]}
            self.assertIn(result.first_move, expected, item["id"])
            self.assertEqual(vars(game), before, item["id"])

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
            self.assertGreater(diag.v7_safety_checked, 0, item["id"])
            game.play(*move)
            if not game.done:
                result = find_vcf(
                    game, opponent,
                    max_fours=V7_FINAL["safety_vcf_max_fours"],
                    node_limit=V7_FINAL["safety_vcf_node_limit"],
                )
                self.assertIsNone(result, item["id"])

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
        self.assertGreater(diag.v7_self_forbidden_penalized, 0)

    def test_stage4_fixture_applies_v7_tiebreak(self):
        item = FIXTURES["seed777-g003-p021"]
        game = _game(item)
        diag = SearchDiagnostics()
        move = mcts_search_v7(
            game, simulations=1, tactical_simulations=1,
            random=random.Random(41), diagnostics=diag,
        )
        self.assertIn(move, game.legal_moves())
        self.assertEqual(diag.forced_policy_stage, 4)
        self.assertTrue(diag.v7_stage4_tiebreak_applied)

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
             d.v7_safety_augmented, d.v7_safety_fallback,
             d.v7_self_forbidden_penalized, d.v7_stage4_tiebreak_applied,
             d.root_candidates)
            for d in diagnostics
        ]
        self.assertEqual(comparable[0], comparable[1])
        self.assertEqual(vars(game), before)


if __name__ == "__main__":
    unittest.main()
