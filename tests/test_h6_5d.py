"""scripts/h6_5d.py: H6-5D positions, controls, V8-C rechecks, same-position comparison and the fixed rule."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import h6_5d as d
from tests.tb_engine_v1_regression import hashed_policy

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))
TINY = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000, 'share': 100, 'call_share': 5_000}
ENOUGH = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000, 'share': None, 'call_share': None}


def pos(i, arm, key, ply, status, kind='decisive'):
    return {'id': f'{arm}|{key}|{ply}', 'arm': arm, 'game_key': key, 'game_kind': kind, 'ply': ply,
            'moves': [], 'tree_move': [0, 0], 'recorded_status': status}


class ControlTest(unittest.TestCase):
    def test_same_game_nearest_then_same_kind(self):
        positions = [pos(0, 'a', 'g1', 40, 'UNKNOWN'), pos(1, 'a', 'g1', 37, 'SAFE'), pos(2, 'a', 'g1', 44, 'SAFE'),
                     pos(3, 'a', 'g1', 41, 'UNSAFE'), pos(4, 'a', 'g2', 90, 'UNKNOWN', 'draw_full_board'),
                     pos(5, 'a', 'g3', 70, 'SAFE', 'draw_full_board'), pos(6, 'b', 'g2', 90, 'SAFE', 'draw_full_board')]
        controls = d.match_controls(positions)
        got = {c['control_for']: (c['id'], c['match']) for c in controls}
        self.assertEqual(got['a|g1|40'], ('a|g1|37', 'same_game'))   # 37 and 44 tie at 3 apart: the earlier ply
        self.assertEqual(got['a|g1|41'], ('a|g1|44', 'same_game'))   # 37 is used once only
        self.assertEqual(got['a|g2|90'], ('a|g3|70', 'same_kind'))   # never another arm's position
        self.assertEqual(len({c['id'] for c in controls}), len(controls))


class StatusTest(unittest.TestCase):
    def test_p94_move_cut_then_resolved(self):
        moves, move = WEB['moves'][:94], WEB['moves'][94]
        cut = d.check_status(moves, move, TINY)
        self.assertEqual((cut['status'], cut['budget_cut'], cut['cut_kind']), ('UNKNOWN', True, 'nodes'))
        full = d.check_status(moves, move, ENOUGH)
        self.assertEqual((full['status'], full['budget_cut'], full['cut_kind']), ('UNSAFE', False, None))

    def test_d2_task_rechecks_only_when_not_safe(self):
        task = {'id': 'x', 'arm': 'a', 'set': 'main', 'game_key': 'g', 'game_kind': 'decisive', 'ply': 94,
                'moves': WEB['moves'][:94], 'tree_move': WEB['moves'][94], 'recorded_status': 'UNKNOWN',
                'standard': TINY, 'large': ENOUGH}
        row = d.d2_task(task)
        self.assertTrue(row['replay_matches'])
        self.assertEqual(row['large']['status'], 'UNSAFE')

    def test_budgets(self):
        self.assertEqual(d.STANDARD, {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000,
                                      'share': 200_000, 'call_share': 5_000})
        self.assertEqual((d.LARGE['node_limit'], d.LARGE['call_limit'], d.LARGE['node_budget']),
                         (200_000, 100_000, 4_000_000))


class SamePositionTest(unittest.TestCase):
    def test_d3_task_checks_both_moves(self):
        d._init(None, None, None, {'policy': hashed_policy, 'leaf': lambda g: 0.0})
        task = {'id': 'a|g|20', 'arm': 'a', 'game_key': 'g', 'ply': 20, 'game_kind': 'decisive', 'kind': 'control',
                'recorded_status': 'SAFE', 'moves': WEB['moves'][:20], 'standard': ENOUGH, 'large': ENOUGH}
        row = d.d3_task(task)
        for engine in d.ENGINES:
            r = row['engines'][engine]
            self.assertEqual(r['route'], 'tree')
            self.assertIn(r['status_standard'], ('SAFE', 'UNKNOWN', 'UNSAFE'))
            self.assertIsNotNone(r['tree_move_prior_rank'])
            self.assertGreater(r['visited_children'], 0)


def d3_row(i, kind, base_std, cand_std, base_large=None, cand_large=None, game=None):
    def engine(std, large):
        return {'tree_move': [i % 15, 0], 'status_standard': std, 'status_large': large or std,
                'prior_top': True, 'visited_children': 5}
    return {'id': f'{i}', 'arm': 'a', 'game_key': game or f'g{i}', 'kind': kind, 'same_tree_move': False,
            'engines': {'TB-Engine-v1': engine(base_std, base_large), 'TB-V1-Value': engine(cand_std, cand_large)}}


def d2_summary(safe, unsafe, unknown, mismatches=()):
    n = safe + unsafe + unknown
    pooled = {'n': n, 'share_safe': safe / n, 'share_unsafe': unsafe / n, 'share_unknown': unknown / n}
    return {'replay_mismatches': list(mismatches), 'main': {'pooled': pooled}}


class JudgeTest(unittest.TestCase):
    def test_rules_in_order(self):
        unsafe_more = [d3_row(i, 'not_safe', 'SAFE', 'UNSAFE') for i in range(20)]
        same = [d3_row(i, 'not_safe', 'UNKNOWN', 'UNKNOWN', 'SAFE', 'SAFE') for i in range(20)]
        self.assertEqual(d.judge(d2_summary(80, 5, 5), d.summarize_d3(unsafe_more))['cause'], 'A')
        self.assertEqual(d.judge(d2_summary(50, 5, 35), d.summarize_d3(same))['cause'], 'INCONCLUSIVE')
        self.assertEqual(d.judge(d2_summary(70, 8, 12), d.summarize_d3(same))['cause'], 'C')
        self.assertEqual(d.judge(d2_summary(50, 25, 15), d.summarize_d3(same))['cause'], 'B')
        harder = [d3_row(i, 'not_safe', 'SAFE', 'UNKNOWN') for i in range(20)]
        self.assertEqual(d.judge(d2_summary(50, 25, 15), d.summarize_d3(harder))['cause'], 'INCONCLUSIVE')
        self.assertEqual(d.judge(d2_summary(80, 5, 5, ['x']), d.summarize_d3(unsafe_more))['cause'], 'INCONCLUSIVE')
        broken = {**d2_summary(80, 5, 5), 'control': {'pooled': {'n': 9, 'share_unsafe': 8 / 9}}}
        self.assertEqual(d.judge(broken, d.summarize_d3(unsafe_more))['cause'], 'INCONCLUSIVE')

    def test_cluster_bootstrap_groups_games(self):
        rows = [d3_row(i, 'not_safe', 'SAFE', 'UNSAFE', game='one') for i in range(10)]
        rows += [d3_row(i, 'not_safe', 'SAFE', 'SAFE', game=f'g{i}') for i in range(10, 20)]
        out = d.summarize_d3(rows)['all']['diff_large_unsafe']
        self.assertEqual((out['n'], out['games'], out['mean']), (20, 11, 0.5))
        self.assertLessEqual(out['ci95'][0], 0.0)  # one game carries every difference: not established

    def test_judge_cli(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, 'd2.json').write_text(json.dumps({'summary': d2_summary(70, 8, 12)}), encoding='utf-8')
            same = [d3_row(i, 'not_safe', 'UNKNOWN', 'UNKNOWN', 'SAFE', 'SAFE') for i in range(20)]
            Path(tmp, 'd3.json').write_text(json.dumps({'summary': d.summarize_d3(same)}), encoding='utf-8')
            out = Path(tmp, 'v.json')
            self.assertEqual(d.main(['judge', '--d2', str(Path(tmp, 'd2.json')), '--d3', str(Path(tmp, 'd3.json')),
                                     '--output', str(out)]), 0)
            self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['verdict']['cause'], 'C')


if __name__ == '__main__':
    unittest.main()
