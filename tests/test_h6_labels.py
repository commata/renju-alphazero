"""H6-0 label contract (hybrid.h6_labels, scripts/build_h6_labels.py) and the H6-5 runner option."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from analysis.threats import ThreatSolver
from hybrid.h6_labels import (
    LeakageError, Row, canonical_hash, check_game_provenance, check_leakage, check_reserved_seed, game_rows,
    mask_cross_split, outcome_value, probe_hashes, proof_label, split_for, touches_probe,
)
from renju import BLACK, WHITE

FOUR_BLACK_TO_MOVE = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [0, 6]]  # black wins at once
OPEN_FOUR_WHITE_TO_MOVE = FOUR_BLACK_TO_MOVE[:7]                                          # white cannot stop it


class PositionTest(unittest.TestCase):
    def test_canonical_hash(self):
        a = canonical_hash([[7, 7], [6, 8], [8, 9]])
        self.assertEqual(a, canonical_hash([[7, 7], [8, 6], [6, 5]]))       # rotated by 180 degrees
        self.assertEqual(a, canonical_hash([[8, 9], [6, 8], [7, 7]]))       # move order does not matter
        self.assertNotEqual(a, canonical_hash([[7, 7], [6, 8], [8, 10]]))

    def test_outcome_value_is_side_to_move(self):
        self.assertEqual([outcome_value(BLACK, p, real_draw=False) for p in (0, 1, 2)], [1, -1, 1])
        self.assertEqual([outcome_value(WHITE, p, real_draw=False) for p in (0, 1)], [-1, 1])
        self.assertEqual(outcome_value(None, 4, real_draw=True), 0)
        self.assertIsNone(outcome_value(None, 4, real_draw=False))  # agreed draw / unknown: masked

    def test_proof_labels(self):
        solver = ThreatSolver(node_limit=20_000)
        self.assertEqual(proof_label(FOUR_BLACK_TO_MOVE, solver), {'kind': 'immediate_win', 'value': 1, 'depth': 0})
        self.assertEqual(proof_label(OPEN_FOUR_WHITE_TO_MOVE, solver), {'kind': 'forced_loss', 'value': -1, 'depth': 0})
        self.assertIsNone(proof_label([[7, 7], [6, 8]], solver))


class GameRowsTest(unittest.TestCase):
    def test_proof_precedence_and_conflict_mask(self):
        moves = [[7, 7], [6, 8], [8, 9], [5, 5], [9, 9], [4, 4]]
        # Black won the game, but at ply 3 (white to move) a proof says white wins: outcomes from ply 3 are masked.
        rows = game_rows({'id': 'g', 'moves': moves, 'winner': BLACK}, source='t', split='train',
                         outcome_weight=0.1, proofs={3: {'kind': 'vcf', 'value': 1, 'depth': 0}})
        by_ply = {r.ply: r for r in rows}
        self.assertEqual(sorted(by_ply), [0, 1, 2, 3, 4, 5])
        self.assertEqual((by_ply[0].value_target, by_ply[0].value_mask, by_ply[0].weight), (1, False, 0.1))
        self.assertEqual((by_ply[1].value_target, by_ply[1].side_to_move), (-1, 'white'))
        self.assertEqual((by_ply[3].label_kind, by_ply[3].value_target, by_ply[3].value_mask), ('proof_vcf', 1, False))
        for ply in (4, 5):
            self.assertTrue(by_ply[ply].value_mask)
            self.assertIn('outcome_conflicts_proof', by_ply[ply].flags)
        self.assertEqual(by_ply[2].moves, moves[:2])
        self.assertEqual(by_ply[2].to_dict()['value_perspective'], 'side_to_move')

    def test_consistent_proof_keeps_outcomes(self):
        rows = game_rows({'id': 'g', 'moves': [[7, 7], [6, 8], [8, 9], [5, 5]], 'winner': WHITE}, source='t',
                         split='train', outcome_weight=0.1, proofs={1: {'kind': 'vcf', 'value': 1, 'depth': 0}})
        self.assertFalse(any(r.value_mask for r in rows))

    def test_agreed_draw_is_masked(self):
        rows = game_rows({'id': 'g', 'moves': [[7, 7], [6, 8]], 'winner': None}, source='t', split='train',
                         outcome_weight=0.1)
        self.assertTrue(all(r.value_mask and r.label_kind == 'unlabelled' for r in rows))


def row(game, split, moves, masked=False, engine=None):
    return Row(game_id=game, source='t', split=split, ply=len(moves), moves=moves, side_to_move='black',
               value_target=1, value_mask=masked, label_kind='outcome', weight=0.1,
               canonical_hash=canonical_hash(moves), engine_config_hash=engine)


class LeakageTest(unittest.TestCase):
    def test_probe_files(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, 'a.json').write_text(json.dumps({'probes': [{'moves': [[7, 7], [6, 8]]}]}), encoding='utf-8')
            Path(tmp, 'b.json').write_text(json.dumps({'moves': [[7, 7], [5, 5], [9, 9]],
                                                       'probes': {'P1': {'plies_played': 3}}}), encoding='utf-8')
            Path(tmp, 'c.json').write_text(json.dumps({'positions': [{'history': [[7, 7]]}]}), encoding='utf-8')
            probes = probe_hashes([Path(tmp, n) for n in ('a.json', 'b.json', 'c.json')])
        self.assertEqual(len(probes), 3)
        self.assertTrue(touches_probe([[7, 7], [8, 6], [1, 1]], probes))   # D4 image of the first probe at ply 2
        self.assertFalse(touches_probe([[7, 7], [7, 8], [1, 1], [2, 2]], probes - {canonical_hash([[7, 7]])}))

    def test_split_and_cross_split_mask(self):
        self.assertEqual(split_for('x'), split_for('x'))
        splits = [split_for(str(i)) for i in range(4000)]
        self.assertTrue(0.03 < splits.count('test') / 4000 < 0.07 and 0.03 < splits.count('val') / 4000 < 0.07)
        rows = [row('a', 'train', [[7, 7], [6, 8]]), row('b', 'val', [[7, 7], [8, 6]]),
                row('c', 'test', [[7, 7], [6, 8]]), row('d', 'val', [[7, 7], [1, 1]])]
        self.assertEqual(mask_cross_split(rows), {'train': 0, 'val': 1, 'test': 1})
        check_leakage(rows)
        rows[1].value_mask = False
        with self.assertRaises(LeakageError):
            check_leakage(rows)

    def test_provenance_and_reserved_seeds(self):
        check_game_provenance([row('a', 'train', [[7, 7]], engine='x'), row('a', 'train', [[7, 7], [6, 8]], engine='x')])
        with self.assertRaises(ValueError):
            check_game_provenance([row('a', 'train', [[7, 7]], engine='x'), row('a', 'train', [[7, 7], [6, 8]], engine='y')])
        check_reserved_seed(9001)
        for seed in (8403, 8413, 8418):
            with self.assertRaises(ValueError):
                check_reserved_seed(seed)


class BuilderTest(unittest.TestCase):
    def test_build_end_to_end(self):
        from scripts.build_h6_labels import build, main
        games = [
            {'id': 1, 'split': 'train', 'bresult': '1', 'end': 'five', 'masked_plies': [], 'moves': FOUR_BLACK_TO_MOVE + [[7, 11]]},
            {'id': 2, 'split': 'val', 'bresult': '0.5', 'end': 'unknown', 'masked_plies': [6], 'moves': FOUR_BLACK_TO_MOVE[:6] + [[3, 3], [4, 4]]},
            {'id': 3, 'split': 'train', 'bresult': '0', 'end': 'unknown', 'masked_plies': [], 'moves': [[7, 7], [1, 1], [2, 2], [3, 3], [4, 5], [5, 4]]},
        ]
        probes = {canonical_hash([[7, 7], [1, 1], [2, 2]])}
        rows, summary = build(games, splits={'train', 'val', 'test'}, from_ply=5, outcome_weight=0.1, proof_weight=1.0,
                              prove_losses=False, vct1=False, probes=probes)
        self.assertEqual(summary['dropped'], {'probe_game': 1, 'other_split': 0})
        g1 = {r.ply: r for r in rows if r.game_id == 'renjunet:1'}
        self.assertEqual((g1[8].label_kind, g1[8].value_target), ('proof_immediate_win', 1))
        self.assertEqual((g1[7].label_kind, g1[7].value_target), ('proof_forced_loss', -1))  # white to move
        self.assertEqual((g1[5].label_kind, g1[5].value_target, g1[5].weight), ('outcome', -1, 0.1))
        self.assertTrue(all(r.value_mask for r in rows if r.game_id == 'renjunet:2'))  # agreed draw
        self.assertIn('-1', summary['sign'])
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):  # RenjuNet-derived rows never go into the tracked tree
                main(['--games', str(Path(tmp, 'g.jsonl')), '--output', 'docs/h6.jsonl.gz'])


class RunnerTest(unittest.TestCase):
    def test_opponent_puct_c_key(self):
        from scripts.run_mcts_v8_benchmark import build_tasks
        args = SimpleNamespace(arm='full', opponent='v8:puct_policy', pairs=1, seed=8403, opening_random_plies=2,
                               opening_radius=2, counterfactual=False, search_overrides={}, policy_checkpoint=None,
                               opponent_puct_c=1.5, opponent_overrides={'puct_c': 1.5})
        tasks, _ = build_tasks(args)
        self.assertEqual(tasks[0]['key'], 'full@v8:puct_policy[puct_c=1.5]/8403/0/black')
        self.assertEqual(tasks[0]['v7_overrides'], {'puct_c': 1.5})

    def test_puct_opponent_needs_its_c(self):
        from scripts.run_mcts_v8_benchmark import main
        with self.assertRaises(SystemExit):
            main(['--arm', 'full', '--opponent', 'v8:puct_heur', '--pairs', '1'])
        with self.assertRaises(SystemExit):
            main(['--arm', 'full', '--opponent', 'v8:full', '--opponent-puct-c', '1.5', '--pairs', '1'])


if __name__ == '__main__':
    unittest.main()
