"""H7-c decision sensitivity (scripts/h7c_sensitivity.py): sample, measures, bootstrap, and a real replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rt30 import fast5
from scripts import h7c_sensitivity as sn
from tests.test_h7c_labels import game


def ply(n, color, a, b, route_a='tree', route_b='tree'):
    rec = lambda move, route: {'route': route, 'engine_move': move,  # noqa: E731
                               'visits': [[*move, 30, 0.2], [3, 3, 10, -0.1]] if route == 'tree' else []}
    return {'ply': n, 'color': color, 'sets': {'a': rec(a, route_a), 'b': rec(b, route_b)}}


class MeasureTest(unittest.TestCase):
    def test_sample_is_fixed_and_seeded_by_game_id(self):
        games = [game(i) for i in range(10)]
        first = [g['game_index'] for g in sn.sample_games(games, 4)]
        self.assertEqual(first, [g['game_index'] for g in sn.sample_games(list(reversed(games)), 4)])
        self.assertEqual(len(set(first)), 4)

    def test_parse(self):
        models = sn.parse_models(['x=p.pt,v.pt', 'y=p.pt,w.pt'])
        self.assertEqual(models['y'], {'policy': 'p.pt', 'value': 'w.pt'})
        self.assertEqual(sn.parse_compares(['x:y'], models), [('x', 'y')])
        for bad in (['x=p.pt'], ['x=p.pt,v.pt', 'x=p.pt,v.pt']):
            with self.assertRaises(ValueError):
                sn.parse_models(bad)
        for bad in (['x:x'], ['x:z']):
            with self.assertRaises(ValueError):
                sn.parse_compares(bad, models)

    def test_compare(self):
        records = [{'game_index': 0, 'plies': [ply(3, 'white', [7, 8], [7, 8]), ply(4, 'black', [6, 6], [5, 5]),
                                               ply(5, 'white', [1, 1], [1, 1], 'stage2', 'stage2')]},
                   {'game_index': 1, 'plies': [ply(3, 'white', [7, 8], [7, 8]), ply(4, 'black', [6, 6], [6, 6])]}]
        out = sn.compare(records, 'a', 'b')
        self.assertEqual((out['all_plies']['plies'], out['all_plies']['different_moves']), (5, 1))
        self.assertEqual((out['tree_plies']['plies'], out['tree_plies']['share']), (4, 0.25))
        self.assertEqual(out['games_with_a_different_move'], {'games': 1, 'share': 0.5, 'first_ply_median': 4})
        self.assertEqual(out['tree_plies_by_side']['black']['different_moves'], 1)
        lo, hi = out['all_plies']['ci95_games']
        self.assertTrue(0 <= lo <= 0.2 <= hi <= 1)
        self.assertGreater(out['tree_js_mean'], 0)
        self.assertEqual(sn.js({(0, 0): 1.0}, {(0, 0): 1.0}), 0.0)


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class ReplayTest(unittest.TestCase):
    """Real TB-V3 / fast5 agents: the actor's own models reproduce the recorded game; another value is compared."""

    def tearDown(self):
        from hybrid.h6_fast import install
        from scripts import h7c_arena as ar
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_actor_reproduces_and_compare_runs(self):
        from analysis import tb_v3_net
        from scripts import h7c_selfplay as sp
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            policy, value = write_models(Path(tmp) / 'a', seed=5)
            _, other_value = write_models(Path(tmp) / 'b', seed=6)
            sp._init_worker(str(policy), str(value), 'cpu')
            models = tb_v3_net.model_identity(policy, value)
            task = {'identity': {**sp.run_identity(8, 8808, models, 'cpu'),
                                 'generator_git': {'commit': 'x', 'dirty': False}},
                    'game_index': 0, 'game_seed': sp.game_seed(8808, 8, 0), 'opening_seed': sp.opening_seed(8808, 8, 0)}
            g = sp.play_game(task, max_plies=10)
            sets = {'actor': {'policy': str(policy), 'value': str(value)},
                    'other': {'policy': str(policy), 'value': str(other_value)}}
            record = sn.replay_game(g, sets, 'cpu')
            check = sn.actor_check([record], {0: g}, 'actor')
            self.assertTrue(check['reproduced'], check)
            self.assertEqual(check['plies'], len(g['plies']))
            out = sn.compare([record], 'other', 'actor')
            self.assertEqual(out['all_plies']['plies'], len(g['plies']))

            # run(): the evidence must name the file, one model set must be the actor; resumed without new work
            import argparse
            import hashlib
            import json
            from unittest import mock
            root = Path(__file__).resolve().parents[1]
            games = Path(tmp) / 'selfplay.jsonl'
            games.write_text(json.dumps(g) + '\n', encoding='utf-8')
            evidence = Path(tmp) / 'evidence.json'
            evidence.write_text(json.dumps({'files_sha256': {
                'runs/h7c/c8/selfplay.jsonl': hashlib.sha256(games.read_bytes()).hexdigest()}}))
            jsonl = root / 'runs' / '_test_sensitivity' / 'r.jsonl'
            self.addCleanup(lambda: (jsonl.unlink(missing_ok=True), jsonl.parent.rmdir()))
            args = argparse.Namespace(games=games, evidence=evidence, device='cpu', workers=1, jsonl=jsonl,
                                      games_sample=1, allow_dirty=True, output=Path(tmp) / 'o.json',
                                      model=[f'actor={policy},{value}', f'other={policy},{other_value}'],
                                      compare=['other:actor'])
            quiet = lambda *a, **k: None  # noqa: E731
            with mock.patch('h7c_labels.check_games', lambda games: None):  # the cut test game is not complete
                result = sn.run(args, log=quiet)
                self.assertTrue(result['actor_check']['reproduced'])
                self.assertEqual(sn.run(args, log=quiet)['jsonl_sha256'], result['jsonl_sha256'])
                for change in ({'model': [f'x={policy},{other_value}', f'y={policy},{other_value}']},
                               {'evidence': games}):
                    with self.assertRaises(ValueError):
                        sn.run(argparse.Namespace(**{**vars(args), **change}), log=quiet)


if __name__ == '__main__':
    unittest.main()
