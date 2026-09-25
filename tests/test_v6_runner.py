from dataclasses import asdict
import unittest

from agents import MCTSV6Agent
from evaluation.match import GameResult, MatchResult
from renju import BLACK, WHITE
from scripts.run_mcts_v6_vs_v5 import make_parser, summarize, annotate_defense_outcomes, v5_final
from search.mcts_v6 import V5_FINAL, SearchDiagnostics


class V6RunnerTest(unittest.TestCase):
    def test_baseline_equal_and_cli_small(self):
        self.assertEqual(make_parser().parse_args([]).games, 1)
        for agent in (v5_final(42), MCTSV6Agent(42)):
            self.assertEqual({k: getattr(agent, k) for k in V5_FINAL}, V5_FINAL)

    def test_color_counts_selections_and_timing(self):
        games = [GameResult(BLACK, 1, 2, 'MCTS-v6', 'MCTS-v5-final', ((7,7),)),
                 GameResult(None, 2, 4, 'MCTS-v5-final', 'MCTS-v6', ((7,7),(6,6)))]
        match = MatchResult(2,1,0,1,3,1.5,6,1/3,tuple(games),42)
        decisions = [dict(game_id=1, ply=1, agent='MCTS-v6', player=BLACK, seconds=2,
                          **asdict(SearchDiagnostics(black_43_candidates=2, v6_root_injection_count=2,
                                                     v6_selected_reasons=('black_43',)))),
                     dict(game_id=2, ply=1, agent='MCTS-v5-final', player=BLACK, seconds=1),
                     dict(game_id=2, ply=2, agent='MCTS-v6', player=WHITE, seconds=3,
                          **asdict(SearchDiagnostics(black_43_candidates=1, black_43_defense_injections=3,
                                                     v6_selected_reasons=('black_43_defense',))))]
        annotate_defense_outcomes([('test', match)], decisions)
        summary = summarize([('test', match)], decisions)
        self.assertEqual(summary['v6_by_color']['black']['wins'], 1)
        self.assertEqual(summary['v6_by_color']['white']['draws'], 1)
        self.assertEqual(summary['v6_by_color']['black']['detector_counts']['black_43_candidates'], 2)
        self.assertEqual(summary['v6_by_color']['white']['selected_counts']['black_43_defense'], 1)
        self.assertEqual(summary['v6_by_color']['white']['defense_followups_observed'], 0)
        self.assertEqual(summary['observed_overhead'], 1.5)
        self.assertIsNone(decisions[-1]['black_43_on_next_reply'])


if __name__ == '__main__':
    unittest.main()
