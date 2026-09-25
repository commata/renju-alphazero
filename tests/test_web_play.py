import unittest

from scripts.run_web_play import VERSION_LABELS, PlaySession, create_agent
from renju import BLACK


class WebPlayTest(unittest.TestCase):
    def test_all_versions_can_be_constructed(self):
        for key in VERSION_LABELS:
            agent = create_agent(key, seed=7)
            self.assertTrue(callable(agent.select_move))

    def test_new_black_session_does_not_make_ai_move(self):
        session = PlaySession()
        state = session.reset(agent_key="v6", human_color=BLACK, seed=7)
        self.assertEqual(state["move_count"], 0)
        self.assertEqual(state["to_play"], "BLACK")
        self.assertEqual(state["human_color"], "BLACK")
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

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            create_agent("nope")


if __name__ == "__main__":
    unittest.main()
