"""H2: RenjuNet RIF -> validated Track B game set (hybrid.renjunet)."""
import io
import subprocess
import sys
from pathlib import Path
import unittest

from hybrid.renjunet import (REASONS, Zobrist, build, leakage_check, parse_move, read_rif, sequence_key,
                             split_tournaments, transform, verify_games)

ROOT = Path(__file__).resolve().parents[1]

# 13 moves; black's last move g8 is a double three (forbidden) -> truncated to 12 moves.
FORBIDDEN_LAST = 'h8 a15 i8 c15 g7 e15 g6 g15 b3 o1 l3 m1 g8'
QUIET = 'h8 a15 i8 c15 g7 e15 g6 g15 b3 o1 l3 m1'
# Black makes five on row 8 (h8..l8) after quiet white moves.
FIVE = 'h8 a1 i8 a3 j8 a5 k8 a7 n2 a9 l8'


def _rif(games, rules=(('1', '1'), ('11', '2'))):
    rule_xml = ''.join(f'<rule id="{i}" name="r{i}" category="{c}"><info/></rule>' for i, c in rules)
    tournaments = {t for _, t, *_ in games}
    tour_xml = ''.join(f'<tournament id="{t}" name="t{t}" />' for t in sorted(tournaments))
    game_xml = ''.join(f'<game id="{i}" tournament="{t}" rule="{r}" bresult="{b}"><move>{m}</move></game>'
                       for i, t, r, b, m in games)
    xml = (f'<?xml version="1.0"?><database><rules>{rule_xml}</rules><tournaments>{tour_xml}</tournaments>'
           f'<games>{game_xml}</games></database>')
    return read_rif(io.BytesIO(xml.encode('utf-8')))


class ParseTest(unittest.TestCase):
    def test_coordinates(self):
        self.assertEqual(parse_move('h8'), (7, 7))
        self.assertEqual(parse_move('a15'), (0, 0))
        self.assertEqual(parse_move('o1'), (14, 14))
        for bad in ('p1', 'a0', 'a16', 'h', '8h'):
            self.assertIsNone(parse_move(bad))

    def test_transform_matches_model_symmetry_convention(self):
        self.assertEqual(transform((0, 1), 1), (13, 0))  # CCW: (r, c) -> (N-1-c, r)
        self.assertEqual(transform((0, 1), 4), (0, 13))  # mirror columns


class ClassifyTest(unittest.TestCase):
    def _build(self, games):
        return build(_rif(games))

    def test_every_reason_and_sum(self):
        games = [
            (1, 1, '1', '0', FORBIDDEN_LAST),                 # accepted, truncated forbidden
            (2, 1, '11', '1', FIVE),                          # gomoku category
            (3, 1, '1', '1', 'h8 z9 i8'),                     # bad coordinate
            (4, 1, '1', '1', 'h8 a1 h8 a2'),                  # occupied point
            (5, 1, '1', '1', 'a1 h8 b1 c1 d1 e1 f1 g1 i1 j1'),  # first move not centre
            (6, 1, '1', '1', 'h8 a15 i8 c15 g7 e15 g6 g15 g8 o1 b3 m1'),  # forbidden then more moves
            (7, 1, '1', '1', FIVE + ' o15'),                  # move after five
            (8, 1, '1', '1', 'h8 a1 i8'),                     # too short
            (9, 1, '1', '0', FIVE),                           # five by black, bresult says white
            (10, 2, '1', '1', FIVE),                          # accepted, five
            (11, 3, '1', '0.5', QUIET.replace('m1', 'n1')),   # accepted, unknown end
        ]
        accepted, report = self._build(games)
        counts = report['primary_reason']
        self.assertEqual(set(counts), set(REASONS))
        self.assertTrue(report['sum_check'])
        self.assertEqual(sum(counts.values()), 11)
        for reason in ('non_renju_rule', 'bad_coordinate', 'occupied_point', 'non_center_first',
                       'midgame_forbidden', 'moves_after_five', 'too_short', 'result_mismatch'):
            self.assertEqual(counts[reason], 1, reason)
        self.assertEqual(counts['accepted'], 3)
        by_id = {g['id']: g for g in accepted}
        self.assertEqual((by_id[1]['end'], by_id[1]['winner'], len(by_id[1]['moves'])), ('forbidden', 'white', 12))
        self.assertEqual((by_id[10]['end'], by_id[10]['winner']), ('five', 'black'))
        self.assertEqual(by_id[11]['end'], 'unknown')
        self.assertEqual(verify_games(accepted), 0)

    def test_forbidden_last_move_recorded_as_black_win_is_a_mismatch(self):
        _, report = self._build([(1, 1, '1', '1', FORBIDDEN_LAST)])
        self.assertEqual(report['primary_reason']['result_mismatch'], 1)

    def test_d4_duplicate_game_keeps_smaller_id(self):
        mirrored = ' '.join(f'{chr(ord("a") + 14 - (ord(t[0]) - ord("a")))}{t[1:]}' for t in FIVE.split())
        accepted, report = self._build([(5, 1, '1', '1', mirrored), (3, 2, '1', '1', FIVE)])
        self.assertEqual(report['primary_reason']['duplicate_game'], 1)
        self.assertEqual([g['id'] for g in accepted], [3])
        self.assertEqual(sequence_key([parse_move(t) for t in FIVE.split()]),
                         sequence_key([parse_move(t) for t in mirrored.split()]))


class SplitLeakageTest(unittest.TestCase):
    def test_split_is_deterministic_and_by_tournament(self):
        games = [{'tournament': t} for t in range(200) for _ in range(3)]
        a, b = split_tournaments(games), split_tournaments(list(reversed(games)))
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {'train', 'val', 'test'})
        test_games = sum(3 for t in a if a[t] == 'test')
        self.assertGreaterEqual(test_games, 0.05 * 600)
        self.assertLess(test_games, 0.05 * 600 + 3)

    def test_shared_positions_are_masked_outside_train(self):
        # Same opening in every tournament: the first plies must be masked in val/test.
        variants = [f'{c}{r}' for r in (10, 11, 12, 13) for c in 'abcdefghijklmno']
        games = [(i + 1, i + 1, '1', '0.5', QUIET.replace('m1', v)) for i, v in enumerate(variants)]
        accepted, report = build(_rif(games))
        self.assertEqual(report['leakage_after_masking'], {'train_val': 0, 'train_test': 0, 'val_test': 0})
        held_out = [g for g in accepted if g['split'] != 'train']
        self.assertTrue(held_out)
        self.assertTrue(all(0 in g['masked_plies'] for g in held_out))
        self.assertEqual(report['primary_reason']['accepted'], 60)
        self.assertEqual(leakage_check(accepted, Zobrist()), report['leakage_after_masking'])

    def test_zobrist_keys_are_d4_invariant(self):
        moves = [parse_move(t) for t in QUIET.split()]
        z = Zobrist()
        for s in range(8):
            self.assertEqual(z.position_keys([transform(m, s) for m in moves]), z.position_keys(moves))


class TrackIsolationTest(unittest.TestCase):
    def test_track_a_does_not_import_hybrid(self):
        code = ('import search.alphazero, search.tactics, training.self_play, model.config, agents\nimport sys\n'
                'print(any(m == "hybrid" or m.startswith("hybrid.") for m in sys.modules))')
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                             cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')}).stdout.strip()
        self.assertEqual(out, 'False')

    def test_no_source_outside_hybrid_imports_it(self):
        offenders = [str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')
                     if 'hybrid' not in p.parts and ('import hybrid' in (t := p.read_text(encoding='utf-8'))
                                                     or 'from hybrid' in t)]
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
