"""Three-state threat proofs of analysis.threats (VCF and depth-limited VCT)."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import unittest

from analysis.threats import SAFE, UNKNOWN, UNSAFE, ThreatSolver, decision_status
from renju import Game

ROOT = Path(__file__).resolve().parents[1]


def after_move_status(game, *, node_limit, four_chain, vct_depth=0):
    return ThreatSolver(node_limit=node_limit, four_chain=four_chain).after_move(game, vct_depth)[0]


def play(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


# Black (7,5)(7,6)(7,7), white blocks (7,4); white to move.
CLOSED_THREE = [(7, 7), (0, 0), (7, 6), (7, 4), (7, 5)]


class ThreatSolverTest(unittest.TestCase):
    def test_opponent_open_four_is_unsafe(self):
        game = play([(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0), (7, 8), (14, 14)])
        self.assertEqual(after_move_status(game, node_limit=1000, four_chain=6), UNSAFE)

    def test_completed_search_without_vcf_is_safe(self):
        game = play(CLOSED_THREE + [(14, 14)])
        self.assertEqual(after_move_status(game, node_limit=100_000, four_chain=6), SAFE)

    def test_exhausted_budget_is_unknown_not_safe(self):
        game = play(CLOSED_THREE + [(14, 14)])
        self.assertEqual(after_move_status(game, node_limit=1, four_chain=6), UNKNOWN)

    def test_own_four_chain_beyond_limit_is_unknown(self):
        # White makes a closed four (black holds (3,2)); with no chain budget the
        # position after black's forced block is not explored.
        game = play([(7, 7), (3, 3), (3, 2), (3, 4), (7, 6), (3, 5), (12, 12), (3, 6)])
        self.assertEqual(after_move_status(game, node_limit=100_000, four_chain=0), UNKNOWN)

    def test_decision_status_needs_no_safe_and_no_unknown_for_loss(self):
        self.assertEqual(decision_status(Counter({UNSAFE: 3})), UNSAFE)
        self.assertEqual(decision_status(Counter({UNSAFE: 3, UNKNOWN: 1})), UNKNOWN)
        self.assertEqual(decision_status(Counter({UNSAFE: 3, UNKNOWN: 1, SAFE: 1})), SAFE)

    def test_unsafe_carries_a_witness(self):
        game = play([(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0), (7, 8), (14, 14)])
        status, witness = ThreatSolver().after_move(game)
        self.assertEqual(status, UNSAFE)
        self.assertEqual(witness[0], 'five')
        self.assertIn(witness[1], [(7, 4), (7, 9)])

    def test_open_three_alone_is_not_a_depth_one_win(self):
        # Black (7,6)(7,7): an open three can be blocked, so white is SAFE at depth 1.
        game = play([(7, 7), (0, 0), (7, 6), (0, 14), (0, 1), (14, 14)])
        solver = ThreatSolver()
        self.assertEqual(solver.after_move(game, 0)[0], SAFE)
        self.assertEqual(solver.after_move(game, 1)[0], SAFE)

    def test_double_three_is_a_depth_one_threat(self):
        # White (3,3)(3,4) and (4,5)(5,5): (3,5) makes two open threes (a 3-3, legal
        # for white). Black is to lose at depth 1 but not at depth 0 (no white VCF).
        game = play([(7, 7), (3, 3), (12, 1), (3, 4), (12, 5), (4, 5), (1, 12), (5, 5),
                     (12, 9)])
        solver = ThreatSolver()
        self.assertEqual(solver.after_move(game, 0)[0], SAFE)
        status, witness = solver.after_move(game, 1)
        self.assertEqual(status, UNSAFE)
        self.assertEqual(witness, ('threat', (3, 5)))


class AnalysisIsolationTest(unittest.TestCase):
    def test_search_training_model_do_not_import_analysis(self):
        code = ('import search.alphazero, search.tactics, training.self_play, '
                'training.loop, model.config\nimport sys\n'
                'print(any(m == "analysis" or m.startswith("analysis.") for m in sys.modules))')
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            code = code.replace('training.loop, ', '')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             check=True, cwd=ROOT).stdout.strip()
        self.assertEqual(out, 'False')

    def test_no_source_outside_analysis_imports_it(self):
        offenders = []
        for path in (ROOT / 'src').rglob('*.py'):
            if 'analysis' in path.parts:
                continue
            text = path.read_text(encoding='utf-8')
            if 'import analysis' in text or 'from analysis' in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


class WebPlayGamesFixtureTest(unittest.TestCase):
    def test_fixture_replays_to_recorded_results(self):
        data = json.loads((ROOT / 'tests' / 'fixtures' / 'web_play_v7_human_games_v1.json')
                          .read_text(encoding='utf-8'))
        self.assertEqual(len(data['games']), 6)
        for record in data['games']:
            game = play(record['moves'])
            self.assertTrue(game.done)
            self.assertEqual('BLACK' if game.winner == 1 else 'WHITE', record['winner'])


if __name__ == '__main__':
    unittest.main()
