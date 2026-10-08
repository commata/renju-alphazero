"""S2-3 selective VCT2 detector (analysis.selective_vct)."""
import json
from pathlib import Path
import unittest

from analysis.selective_vct import (
    NO_TARGETED_VCT2_FOUND, PROVEN_LOSS, SelectiveSolver, classify, threat_moves, verify_witness,
)
from analysis.threats import ThreatSolver, ordered_moves
from renju import Game

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs' / 'mcts-v8-results'


def _game(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _witness(pair):
    positions = json.loads((RESULTS / 's2_witness_positions.json').read_text(encoding='utf-8'))['positions']
    return next(p for p in positions if p['seed'] == 8401 and p['pair'] == pair)


class QuietMoveHookTest(unittest.TestCase):
    def test_full_solver_tries_every_legal_move(self):
        game = _game(_witness(16)['moves'])
        self.assertEqual(ThreatSolver().quiet_moves(game), ordered_moves(game))

    def test_selective_solver_tries_threat_moves_only(self):
        pos = _witness(16)
        game = _game(pos['moves'])
        game.play(pos['decisive_move_1idx'][0] - 1, pos['decisive_move_1idx'][1] - 1)  # black (attacker) to move
        threats = threat_moves(game)
        self.assertTrue(threats)
        self.assertTrue(set(threats) < set(ordered_moves(game)))
        self.assertEqual(SelectiveSolver().quiet_moves(game), threats)


class DetectorTest(unittest.TestCase):
    def test_confirmed_witness_is_detected_and_verified(self):
        # 8401 pair 16: V8 played (8,9) (lost at depth 2); the policy top-1 (10,8) has no loss within depth 2.
        pos = _witness(16)
        game = _game(pos['moves'])
        played = (pos['decisive_move_1idx'][0] - 1, pos['decisive_move_1idx'][1] - 1)
        result = classify(game, played)
        self.assertEqual((result['status'], result['depth']), (PROVEN_LOSS, 2))
        self.assertEqual(result['witness'][0], 'threat')
        self.assertTrue(verify_witness(game, played, result))
        self.assertEqual(game.history, [tuple(m) for m in pos['moves']])  # restored

        alternative = classify(game, (9, 7))
        self.assertEqual(alternative['status'], NO_TARGETED_VCT2_FOUND)  # never called SAFE

    def test_shallow_losses_come_from_the_full_solver(self):
        probes = json.loads((RESULTS / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))
        game = _game(probes['moves'][:94])  # P94: every black move is lost within depth 1
        result = classify(game, (10, 12))   # (11,13)
        self.assertEqual(result['status'], PROVEN_LOSS)
        self.assertLessEqual(result['depth'], 1)
        self.assertTrue(verify_witness(game, (10, 12), result))

    def test_verify_rejects_non_losses(self):
        with self.assertRaises(ValueError):
            verify_witness(Game(), (7, 7), {'status': NO_TARGETED_VCT2_FOUND})


if __name__ == '__main__':
    unittest.main()


class VCT2VetoTest(unittest.TestCase):
    def _children(self, ranked):
        from types import SimpleNamespace
        return [SimpleNamespace(move=m, visits=v, mean_value=0.0) for m, v in ranked]

    def test_proven_loss_switches_to_next_unrefuted_child(self):
        from analysis.mcts_v8 import SearchDiagnostics, _vct2_veto
        pos = _witness(16)
        game = _game(pos['moves'])
        played = (pos['decisive_move_1idx'][0] - 1, pos['decisive_move_1idx'][1] - 1)  # (8,9): lost at depth 2
        # Tree order: the played move, then (8,7) (also lost at depth 2), then (10,8) (no loss within depth 2).
        children = self._children([(played, 10), ((7, 6), 5), ((9, 7), 3)])
        diag = SearchDiagnostics()
        budget = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 10_000}
        final = _vct2_veto(game, played, children, diag, budget=budget, max_children=4)
        self.assertEqual(final, (9, 7))
        self.assertTrue(diag.v8_vct2_switched)
        self.assertEqual(dict(diag.v8_vct2_checked)[played], 'UNSAFE')
        self.assertEqual(dict(diag.v8_vct2_checked)[(7, 6)], 'UNSAFE')
        self.assertEqual(game.history, [tuple(m) for m in pos['moves']])

    def test_unproven_move_stands(self):
        from analysis.mcts_v8 import SearchDiagnostics, _vct2_veto
        game = _game(_witness(16)['moves'])
        diag = SearchDiagnostics()
        budget = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 10_000}
        final = _vct2_veto(game, (9, 7), self._children([((9, 7), 10), ((7, 6), 5)]), diag,
                           budget=budget, max_children=4)
        self.assertEqual(final, (9, 7))
        self.assertFalse(diag.v8_vct2_switched)
        self.assertEqual(len(diag.v8_vct2_checked), 1)  # nothing else is checked
