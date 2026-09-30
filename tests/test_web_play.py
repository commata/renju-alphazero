import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_web_play import VERSION_LABELS, PlaySession, create_agent
from renju import BLACK, WHITE


class WebPlayTest(unittest.TestCase):
    def test_all_versions_can_be_constructed(self):
        for key in VERSION_LABELS:
            agent = create_agent(key, seed=7)
            self.assertTrue(callable(agent.select_move))

    def test_new_white_session_starts_with_forced_center_black(self):
        session = PlaySession()
        state = session.reset(agent_key="v6", human_color=WHITE, seed=7)
        self.assertEqual(state["move_count"], 1)
        self.assertEqual(state["to_play"], "WHITE")
        self.assertEqual(state["human_color"], "WHITE")
        self.assertEqual(state["board"][7][7], BLACK)
        self.assertEqual(state["history"], [[7, 7]])
        self.assertEqual(session.move_records[0]["actor"], "OPENING_RULE")
        self.assertEqual(len(state["board"]), 15)
        self.assertTrue(all(len(row) == 15 for row in state["board"]))

    def test_v5_uses_final_configuration(self):
        agent = create_agent("v5", seed=7)
        self.assertEqual(agent.simulations, 50)
        self.assertEqual(agent.tactical_simulations, 100)
        self.assertEqual(agent.tactical_score_threshold, 1800)
        self.assertEqual(agent.candidate_limit, 20)
        self.assertEqual(agent.initial_width, 8)
        self.assertEqual(agent.neighborhood_radius, 2)
        self.assertEqual(agent.priority_top_k, 8)

    def test_v7_uses_frozen_final_configuration_and_reports_diagnostics(self):
        from search.mcts_v7 import V7_FINAL

        agent = create_agent("v7", seed=7)
        for key, value in V7_FINAL.items():
            self.assertEqual(getattr(agent, key), value, key)

        session = PlaySession()
        state = session.reset(agent_key="v7", human_color=WHITE, seed=7)
        self.assertEqual(state["agent_key"], "v7")
        state = session.play_human(7, 8)
        self.assertEqual(state["move_count"], 3)
        diagnostics = session.move_records[-1]["diagnostics"]
        self.assertEqual(session.move_records[-1]["actor"], "MCTS-v7")
        self.assertIn("v7_own_vcf_found", diagnostics)
        self.assertIn("v7_module_seconds", diagnostics)

    def test_v8_opponent_reports_v8_diagnostics_and_csv_columns(self):
        from analysis.mcts_v8 import V8_DEFAULTS

        agent = create_agent("v8", seed=7)
        self.assertEqual(agent.name, "MCTS-v8")
        for key, value in V8_DEFAULTS.items():
            self.assertEqual(getattr(agent, key), value, key)
        with TemporaryDirectory() as tmp:
            session = PlaySession(log_root=Path(tmp))
            session.reset(agent_key="v8", human_color=WHITE, seed=7)
            state = session.play_human(7, 8)
            self.assertEqual(state["move_count"], 3)
            diagnostics = session.move_records[-1]["diagnostics"]
            self.assertEqual(session.move_records[-1]["actor"], "MCTS-v8")
            self.assertIn("v8_route", diagnostics)
            self.assertIn("v8_attack_candidates", diagnostics)
            json.dumps(diagnostics)  # the game log must stay JSON-serialisable
            session.game.winner = WHITE
            session.game.done = True
            log_dir = session._save_completed_game_locked()
            header = (log_dir / "moves.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertIn("v8_route", header)
            self.assertIn("v8_attack_status", header)
            payload = json.loads((log_dir / "game.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["agent_info"]["config"]["own_vct_attack"], True)

    def test_completed_game_is_saved_once_as_json_and_csv(self):
        with TemporaryDirectory() as tmp:
            session = PlaySession(log_root=Path(tmp))
            session.game.board[7][7] = BLACK
            session.game.history = [(7, 7)]
            session.game.winner = BLACK
            session.game.done = True
            session.move_records = [{
                "ply": 1,
                "player": "BLACK",
                "actor": "HUMAN",
                "row0": 7,
                "col0": 7,
                "row": 8,
                "col": 8,
                "seconds": None,
                "diagnostics": {},
            }]

            first = session._save_completed_game_locked()
            second = session._save_completed_game_locked()

            self.assertEqual(first, second)
            self.assertTrue((first / "game.json").is_file())
            self.assertTrue((first / "moves.csv").is_file())
            payload = json.loads((first / "game.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["winner"], "BLACK")
            self.assertEqual(payload["result"], "HUMAN_WIN")
            self.assertEqual(payload["number_of_moves"], 1)
            self.assertTrue(session.saved_game)

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            create_agent("nope")


if __name__ == "__main__":
    unittest.main()
