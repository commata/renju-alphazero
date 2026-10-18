"""scripts/h6_5_diagnose.py: not-SAFE decomposition and the paired game bootstrap (report only)."""
import unittest

from scripts.h6_5_diagnose import describe, paired_rate_bootstrap


def move(ply, status, played_status='SAFE', exhausted=False, top=True):
    tree_move, played = [ply % 15, 0], [ply % 15, 1]
    return {'route': 'tree', 'ply': ply, 'v7_move': tree_move, 'played': played, 'seconds': 1.0,
            'root': {'checked': [[tree_move, status], [played, played_status]], 'exhausted': exhausted, 'switch': ''},
            'tree': {'prior_top': tree_move if top else [9, 9]}}


def run(statuses_by_game, result='win', length=60):
    games = [{'key': f'arm/8403/{i}/black', 'result': result, 'length': length,
              'v8_moves': [move(10 + j, s) for j, s in enumerate(st)]} for i, st in enumerate(statuses_by_game)]
    return {'arm': 'a', 'opponent': 'v8:full', 'seed': 8403, 'games': games}


class DiagnoseTest(unittest.TestCase):
    def test_describe_counts(self):
        r = run([['SAFE', 'UNKNOWN'], ['UNSAFE', 'SAFE', 'SAFE']])
        d = describe(r)
        self.assertEqual(d['all'], {'not_safe': 2, 'tree_moves': 5, 'rate': 0.4})
        self.assertEqual(d['by_game_kind']['decisive']['not_safe'], 2)
        self.assertEqual(d['games_with_not_safe'], 2)
        self.assertEqual(d['tree_prior_top_rate'], 1.0)
        full = describe(run([['UNKNOWN']], result='draw', length=225))
        self.assertEqual(full['by_game_kind']['draw_full_board']['not_safe'], 1)

    def test_paired_bootstrap(self):
        a = run([['UNKNOWN', 'SAFE']] * 10)
        b = run([['SAFE', 'SAFE']] * 10)
        out = paired_rate_bootstrap(a, b)
        self.assertEqual((out['games'], out['diff'], out['ci95']), (10, 0.5, [0.5, 0.5]))
        self.assertEqual(out['share_above_margin'], 1.0)


if __name__ == '__main__':
    unittest.main()
