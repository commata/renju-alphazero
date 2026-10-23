"""H7-c self-play: policy targets, the opening-sampling rule, seeds, resume and a real candidate game."""
import argparse
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from hybrid import h7c_targets as t
from rt30 import fast5
from scripts import h7c_selfplay as sp

ROOT = Path(__file__).resolve().parents[1]
VISITS = [[7, 8, 30, 0.1], [8, 8, 15, -0.2], [6, 6, 5, 0.0], [9, 9, 0, 0.0]]


def record(root=(), vct2=(), vct=()):
    return {'root': {'checked': [list(x) for x in root]}, 'vct2': {'checked': [list(x) for x in vct2]},
            'vct': {'checked': [list(x) for x in vct]}}


class TargetTest(unittest.TestCase):
    def test_proven_losses_from_every_check(self):
        r = record(root=[([7, 8], 'UNSAFE'), ([6, 6], 'SAFE')], vct2=[([8, 8], 'UNSAFE')], vct=[([1, 1], 'UNKNOWN')])
        self.assertEqual(t.proven_losses(r), [[7, 8], [8, 8]])

    def test_policy_target(self):
        proof = t.policy_target('own_vct2', [3, 4], VISITS, [])
        self.assertEqual((proof['kind'], proof['kl'], proof['target']), ('proof_one_hot', False, [[3, 4, 1.0]]))
        visits = t.policy_target('tree', [7, 8], VISITS, [[7, 8]])  # the most visited child is a proven loss
        self.assertEqual((visits['kind'], visits['kl']), ('visits', True))
        self.assertEqual(visits['target'], [[6, 6, 0.25], [8, 8, 0.75]])  # renormalised, row-major
        self.assertAlmostEqual(sum(p for *_, p in visits['target']), 1.0)
        self.assertEqual(t.policy_target('stage4', [7, 8], [], [])['kind'], 'none')  # forced: no distribution
        self.assertEqual(t.policy_target('tree', [7, 8], VISITS, [[7, 8], [8, 8], [6, 6]])['kind'], 'none')
        for route in t.PROVEN_WIN_ROUTES:
            self.assertEqual(t.policy_target(route, [1, 2], VISITS, [])['kind'], 'proof_one_hot')

    def test_sampling(self):
        self.assertEqual(t.sample_move('own_vcf', (3, 4), VISITS, [], Random(1)), ([3, 4], 'proven_win'))
        self.assertEqual(t.sample_move('stage2', (3, 4), [], [], Random(1)), ([3, 4], 'no_distribution'))
        self.assertEqual(t.sample_move('tree', (7, 8), VISITS, [[7, 8], [8, 8], [6, 6]], Random(1)),
                         ([7, 8], 'no_distribution'))  # everything visited is a proven loss: the engine's move
        rng = Random(7)
        counts = {}
        for _ in range(4000):
            move, event = t.sample_move('tree', (7, 8), VISITS, [[8, 8]], rng)
            self.assertEqual(event, 'sampled')
            counts[tuple(move)] = counts.get(tuple(move), 0) + 1
        self.assertNotIn((8, 8), counts)   # a proven loss is never sampled
        self.assertNotIn((9, 9), counts)   # nor an unvisited child
        self.assertAlmostEqual(counts[(7, 8)] / 4000, 30 / 35, delta=0.03)  # T = 1: proportional to visits
        one, two = Random(3), Random(3)  # the same seed: the same whole sequence of draws
        first = [t.sample_move('tree', (7, 8), VISITS, [], one)[0] for _ in range(200)]
        self.assertEqual(first, [t.sample_move('tree', (7, 8), VISITS, [], two)[0] for _ in range(200)])
        self.assertGreater(len({tuple(m) for m in first}), 1)


class ResumeFileTest(unittest.TestCase):
    IDENTITY = {'format': sp.FORMAT, 'cycle': 1, 'generator_git': {'commit': 'abc', 'dirty': False}}

    def write(self, tmp, lines):
        path = Path(tmp) / 'games.jsonl'
        path.write_bytes(b''.join(lines))
        return path

    def row(self, index, **changes):
        return (json.dumps({**self.IDENTITY, 'game_index': index, **changes}) + '\n').encode()

    def test_resume_file_rules(self):
        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            path = self.write(tmp, [self.row(0), self.row(1)])
            self.assertEqual(sorted(sp.load_finished(path, self.IDENTITY, log=quiet)), [0, 1])
            path = self.write(tmp, [self.row(0), self.row(0)])
            with self.assertRaises(ValueError):  # an index twice: never silently replaced
                sp.load_finished(path, self.IDENTITY, log=quiet)
            path = self.write(tmp, [self.row(0), self.row(1, generator_git={'commit': 'def', 'dirty': False})])
            with self.assertRaises(ValueError):  # another generator commit in the same file
                sp.load_finished(path, self.IDENTITY, log=quiet)
            path = self.write(tmp, [self.row(0), b'{"format": "h7c-sel', b'\n', self.row(1)])
            with self.assertRaises(ValueError):  # an unreadable line in the middle
                sp.load_finished(path, self.IDENTITY, log=quiet)
            good = self.row(0) + self.row(1)
            path = self.write(tmp, [good, self.row(2)[:25]])  # a crash while appending game 2
            messages = []
            self.assertEqual(sorted(sp.load_finished(path, self.IDENTITY, log=lambda m, **k: messages.append(m))),
                             [0, 1])
            self.assertEqual(path.read_bytes(), good)  # the torn line is cut, the file ends on a full line
            self.assertIn('incomplete', messages[0])


class SeedTest(unittest.TestCase):
    def test_seed_rules(self):
        sp.check_seeds(8801, 1, range(1000))
        sp.check_seeds(8800, 1, range(20), measurement=True)
        for bad in ((8799, 1), (9000, 1), (8451, 1), (8801, 0), (8800, 1)):
            with self.assertRaises(ValueError, msg=bad):
                sp.check_seeds(*bad, range(4))
        self.assertEqual(sp.game_seed(8801, 1, 5), sp.game_seed(8801, 1, 5))
        self.assertNotEqual(sp.game_seed(8801, 1, 5), sp.game_seed(8801, 2, 5))   # cycles never share games
        self.assertNotEqual(sp.game_seed(8801, 1, 5), sp.opening_seed(8801, 1, 5))


def args_for(tmp, policy, value, **kw):
    base = {'cycle': 1, 'run_seed': 8801, 'games': 2, 'start': 0, 'workers': 1, 'policy_checkpoint': policy,
                'value_checkpoint': value, 'device': 'cpu', 'jsonl': Path(tmp) / 'games.jsonl', 'output': None,
                'allow_dirty': True, 'max_plies': 8}
    return argparse.Namespace(**{**base, **kw})


def untimed(row):
    drop = {'seconds', 'game_seconds'}
    if isinstance(row, dict):
        return {k: untimed(v) for k, v in row.items() if k not in drop}
    if isinstance(row, list):
        return [untimed(v) for v in row]
    return row


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class GameTest(unittest.TestCase):
    """Real candidate agents (random stand-in networks of the frozen architecture), games cut at 8 plies."""

    def setUp(self):
        from tests.test_tb_v3_net import write_models

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy, self.value = write_models(self.tmp.name)

    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')

    def test_game_record(self):
        from analysis import tb_v3_net

        sp._init_worker(str(self.policy), str(self.value), 'cpu')
        models = tb_v3_net.model_identity(self.policy, self.value)
        task = {'identity': {**sp.run_identity(1, 8801, models, 'cpu'),
                             'generator_git': {'commit': 'x', 'dirty': False}},
                'game_index': 0, 'game_seed': sp.game_seed(8801, 1, 0), 'opening_seed': sp.opening_seed(8801, 1, 0)}
        row = sp.play_game(task, max_plies=9)
        self.assertEqual((row['length'], row['result'], row['truncated']), (9, 'truncated', True))
        self.assertEqual(len(row['opening']), 3)
        self.assertEqual(row['moves'][:3], row['opening'])
        self.assertFalse(row['frozen_models'])
        events = [p['event'] for p in row['plies']]
        self.assertTrue(all(e in ('sampled', 'proven_win', 'no_distribution') for e in events[:4]), events)
        self.assertTrue(all(e == 'engine' for e in events[4:]), events)
        for p in row['plies']:
            self.assertEqual(p['played'], row['moves'][p['ply']])
            if p['event'] == 'engine':
                self.assertEqual(p['played'], p['engine_move'])
            self.assertIn(p['target']['kind'], ('visits', 'proof_one_hot', 'none'))
            if p['target']['kind'] == 'visits':
                self.assertTrue(p['visits'])
                self.assertAlmostEqual(sum(x[2] for x in p['target']['target']), 1.0)
        again = sp.play_game(task, max_plies=9)  # the same seeds, the same game
        self.assertEqual(untimed(again), untimed(row))

    def test_run_and_resume(self):
        with TemporaryDirectory() as tmp:
            args = args_for(tmp, self.policy, self.value)
            first = sp.run(args, log=lambda *a, **k: None)
            self.assertEqual(first['summary']['games'], 2)
            lines = args.jsonl.read_text(encoding='utf-8').splitlines()
            self.assertEqual(sorted(json.loads(x)['game_index'] for x in lines), [0, 1])
            more = sp.run(args_for(tmp, self.policy, self.value, games=3), log=lambda *a, **k: None)
            self.assertEqual(more['summary']['games'], 3)
            self.assertEqual(len(args.jsonl.read_text(encoding='utf-8').splitlines()), 3)  # only game 2 was new
            with self.assertRaises(ValueError):  # another run seed: refused, never mixed
                sp.run(args_for(tmp, self.policy, self.value, run_seed=8802), log=lambda *a, **k: None)
            with self.assertRaises(ValueError):  # another cycle
                sp.run(args_for(tmp, self.policy, self.value, cycle=2), log=lambda *a, **k: None)

    def test_dirty_tree_refused(self):
        with TemporaryDirectory() as tmp:
            args = args_for(tmp, self.policy, self.value, allow_dirty=False)
            from unittest import mock
            with mock.patch('run_mcts_v8_benchmark._git_dirty', lambda: True), self.assertRaises(ValueError):
                sp.run(args, log=lambda *a, **k: None)



class ThroughputTest(unittest.TestCase):
    def test_compare_and_choice(self):
        from scripts import h7c_throughput as tp

        a = [{'game_index': 0, 'moves': [[7, 7]], 'plies': [{'seconds': 1.0, 'route': 'tree'}], 'game_seconds': 3}]
        b = [{'game_index': 0, 'moves': [[7, 7]], 'plies': [{'seconds': 2.0, 'route': 'tree'}], 'game_seconds': 9}]
        self.assertEqual(tp.compare_runs(a, b)['identical'], 1)  # times differ only
        c = [{'game_index': 0, 'moves': [[7, 7]], 'plies': [{'seconds': 1.0, 'route': 'own_vcf'}]}]
        self.assertEqual(tp.compare_runs(a, c)['differences'], {'0': '.plies[0].route'})
        rows = [{'workers': w, 'games_per_hour': g, 'same_games_as_first': 20}
                for w, g in ((6, 60), (8, 76), (10, 80), (12, 81), (14, 79))]
        self.assertEqual(tp.choose(rows), 10)  # 80 >= 0.95 * 81
        self.assertEqual(tp.sweep_verdict(rows, 20), {'verdict': 'SWEEP_PASS', 'diverged_workers': [],
                                                      'chosen_workers': 10})
        rows[3]['same_games_as_first'] = 19  # 12 workers played one game differently: no worker count is chosen
        self.assertEqual(tp.sweep_verdict(rows, 20), {'verdict': 'SWEEP_REPRO_FAIL', 'diverged_workers': [12],
                                                      'chosen_workers': None})


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class ThroughputRunTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')

    def test_repro_with_two_worker_counts(self):
        from scripts import h7c_throughput as tp
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            policy, value = write_models(tmp)
            args = argparse.Namespace(policy_checkpoint=policy, value_checkpoint=value, device='cpu',
                                      out=Path(tmp) / 'out', repro_workers=2, allow_dirty=True, max_plies=7)
            args.out.mkdir()
            result = tp.repro(args, log=lambda *a, **k: None)
            self.assertEqual((result['verdict'], result['games'], result['workers']), ('REPRO_PASS', 4, [1, 2]))
            with self.assertRaises(ValueError):  # a used directory is refused
                tp.repro(args, log=lambda *a, **k: None)


if __name__ == '__main__':
    unittest.main()
