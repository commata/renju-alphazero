"""H7-c Arena (scripts/h7c_arena.py) and X2 (scripts/h7c_external.py): seeds, pairing, verdict rules, real games."""
import argparse
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from rt30 import fast5
from scripts import h7c_arena as ar
from scripts import h7c_external as x2

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'scripts' / 'fake_piskvork_engine.py')


def game(pair, color, result, illegal=None):
    return {'key': f'a/{pair}/{color}', 'pair': pair, 'tested_color': color, 'result': result, 'length': 30,
            'illegal': illegal, 'truncated': result == 'truncated'}


def arm_summary(pair_results):
    games = []
    for pair, (black, white) in enumerate(pair_results):
        games += [game(pair, 'black', black), game(pair, 'white', white)]
    return ar.summarize_arm(games)


class SeedPairingTest(unittest.TestCase):
    def test_seeds(self):
        from hybrid.h6_labels import RESERVED_SEEDS, check_reserved_seed

        self.assertEqual([ar.arena_seed(c) for c in (1, 10)], [8731, 8740])
        for cycle in (0, 11):
            with self.assertRaises(ValueError):
                ar.arena_seed(cycle)
        self.assertTrue(set(range(8731, 8741)) <= RESERVED_SEEDS)
        with self.assertRaises(ValueError):  # never a training seed
            check_reserved_seed(8731)
        self.assertEqual([x2.x2_seed(c) for c in (1, 10)], [8711, 8720])
        for cycle in (0, 11):
            with self.assertRaises(ValueError):
                x2.x2_seed(cycle)

    def test_arms_share_openings_and_seeds(self):
        official, openings = ar.build_tasks('official', 1)
        self.assertEqual((len(official), len(openings)), (100, 50))
        self.assertEqual({ar.ARMS[a]['pairs'] for a in ar.ARMS if a != 'official'}, {25})
        for arm in ('p_only', 'v_only', 'mon_v8_full', 'mon_puct_policy', 'mon_e2'):
            tasks, _ = ar.build_tasks(arm, 1)
            self.assertEqual(len(tasks), 50)
            for a, b in zip(tasks, official[:50]):  # the first 25 openings of the official arm, the same game seeds
                self.assertEqual((a['opening'], a['seed'], a['tested_color'], a['pair']),
                                 (b['opening'], b['seed'], b['tested_color'], b['pair']))
                self.assertNotEqual(a['key'], b['key'])
        self.assertNotEqual(ar.build_tasks('official', 2)[0][0]['opening'] + [ar.build_tasks('official', 2)[0][0]['seed']],
                            official[0]['opening'] + [official[0]['seed']])

    def test_side_models(self):
        models = {'candidate': {'policy': 'cp', 'value': 'cv'}, 'parent': {'policy': 'pp', 'value': 'pv'}}
        self.assertEqual(ar.side_models('official', models), ({'policy': 'cp', 'value': 'cv'}, models['parent']))
        self.assertEqual(ar.side_models('p_only', models), ({'policy': 'cp', 'value': 'pv'}, models['parent']))
        self.assertEqual(ar.side_models('v_only', models), ({'policy': 'pp', 'value': 'cv'}, models['parent']))
        self.assertEqual(ar.side_models('mon_e2', models), ({'policy': 'cp', 'value': 'cv'}, None))


class VerdictRuleTest(unittest.TestCase):
    def test_summary(self):
        s = arm_summary([('win', 'win'), ('win', 'loss'), ('draw', 'loss'), ('loss', 'loss')])
        self.assertEqual((s['score'], s['wins'], s['draws'], s['losses'], s['complete_pairs']), (0.4375, 3, 1, 4, 4))
        self.assertEqual(s['score_by_color'], {'black': 0.625, 'white': 0.25})
        self.assertEqual(s['improvement'], -0.0625)
        self.assertEqual(s['improvement_ci95'], [round(x - 0.5, 4) for x in s['score_ci95']])
        broken = ar.summarize_arm([game(0, 'black', 'win'), game(0, 'white', 'truncated'),
                                   game(1, 'black', 'loss', {'side': 'tested'}), game(1, 'white', 'win')])
        self.assertEqual((broken['complete_pairs'], broken['incomplete_pairs'], broken['illegal']['tested']),
                         (1, [0], 1))

    def test_promotion(self):
        def official(score, lower):
            return {'score': score, 'improvement_ci95': [lower, 0.2]}
        self.assertEqual(ar.promotion(official(0.53, -0.04), True)['decision'], 'PROMOTE')
        self.assertEqual(ar.promotion(official(0.53, -0.05), True)['decision'], 'PROMOTE')   # bound inclusive
        for case in (official(0.50, 0.0), official(0.6, -0.06)):                         # score > 0.5 strictly
            self.assertEqual(ar.promotion(case, True)['decision'], 'KEEP_CHAMPION')
        kept = ar.promotion(official(0.7, 0.1), False)
        self.assertEqual((kept['decision'], kept['next_parent']), ('KEEP_CHAMPION', 'parent'))
        self.assertEqual(ar.promotion(official(0.53, -0.04), True)['next_parent'], 'candidate')

    def test_convergence(self):
        def cycle(c, lower, tactical='PASS'):
            return {'cycle': c, 'promotion': {'improvement_lower': lower}, 'tactical': {'verdict': tactical}}
        self.assertFalse(ar.convergence([cycle(1, -0.1), cycle(2, -0.1)])['converged'])          # fewer than 3
        self.assertTrue(ar.convergence([cycle(1, 0.1), cycle(2, -0.1), cycle(3, 0.0)])['converged'])
        self.assertFalse(ar.convergence([cycle(1, -0.1), cycle(2, 0.01), cycle(3, -0.1)])['converged'])
        self.assertFalse(ar.convergence([cycle(1, 0.1), cycle(2, -0.1), cycle(3, -0.1, 'FAIL')])['converged'])

    def test_tactical_judge(self):
        def side(unsound=0, sign=True, hits=12, rescue=0.6):
            return {'e1': {'metrics': {'unsound_witness': unsound, 'false_veto': 0,
                                       'rescue_rate': {'value': rescue, 'n': 'x'}, 'veto_recall': {'value': 1.0}}},
                    'value_probes': {'pass': sign}, 'must_block': {'hits': hits, 'pass': hits >= 9}}
        ok = ar.tactical_judge(side(rescue=0.5, hits=11), side())
        self.assertEqual(ok['verdict'], 'PASS')
        self.assertEqual(ok['regressions_reported'], ['must_block_hits', 'rescue_rate'])  # reported, not gated
        for bad in (side(unsound=1), side(sign=False), side(hits=8)):
            self.assertEqual(ar.tactical_judge(bad, side())['verdict'], 'FAIL')


def write_arm(tmp, arm, results, **identity):
    ident = {'arm': arm, 'cycle': 1, 'git': {'commit': 'abc'}, 'candidate': {'c': 1}, 'parent': {'p': 1},
             **identity}
    sig = ar.signature(ident)
    games = []
    for pair, (black, white) in enumerate(results):
        games += [{**game(pair, 'black', black), 'signature': sig}, {**game(pair, 'white', white), 'signature': sig}]
    path = Path(tmp) / f'{arm}.json'
    path.write_text(json.dumps({'identity': ident, 'signature': sig, 'games': games,
                                'summary': ar.summarize_arm(games), 'fingerprint': arm}), encoding='utf-8')
    return str(path)


class VerdictCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tactical = Path(self.tmp.name) / 'tactical.json'
        self.tactical.write_text(json.dumps({
            'cycle': 1, 'git': {'commit': 'abc'}, 'candidate': {'c': 1}, 'parent': {'p': 1},
            'judge': {'items': {'unsound_witness == 0': True}, 'compare': {}, 'regressions_reported': []}}))

    def arms(self, official, **extra):
        paths = [write_arm(self.tmp.name, 'official', official)]
        for arm in ar.ARMS:
            if arm != 'official':
                paths.append(write_arm(self.tmp.name, arm, [('win', 'loss')] * 25, **extra.get(arm, {})))
        return paths

    def run_verdict(self, arms, history=()):
        return ar.verdict(argparse.Namespace(arms=arms, tactical=self.tactical, history=list(history)))

    def test_promote_and_keep(self):
        promote = self.run_verdict(self.arms([('win', 'win')] * 30 + [('win', 'loss')] * 20))
        self.assertEqual(promote['promotion']['decision'], 'PROMOTE')
        self.assertEqual(promote['tactical']['forbidden_errors'], 0)
        self.assertEqual(set(promote['monitoring']), {'mon_v8_full', 'mon_puct_policy', 'mon_e2'})
        self.assertEqual(set(promote['diagnosis']), {'p_only', 'v_only'})
        self.assertFalse(promote['convergence']['converged'])
        keep = self.run_verdict(self.arms([('win', 'loss')] * 50))
        self.assertEqual(keep['promotion']['decision'], 'KEEP_CHAMPION')  # score 0.5: not above

    def test_forbidden_error_fails_the_gate(self):
        paths = self.arms([('win', 'win')] * 50)
        record = json.loads(Path(paths[1]).read_text())
        record['summary']['illegal']['tested'] = 1
        Path(paths[1]).write_text(json.dumps(record))
        out = self.run_verdict(paths)
        self.assertEqual((out['tactical']['verdict'], out['promotion']['decision']), ('FAIL', 'KEEP_CHAMPION'))

    def test_refusals(self):
        with self.assertRaises(ValueError):  # an incomplete official arm
            self.run_verdict(self.arms([('win', 'win')] * 49))
        with self.assertRaises(ValueError):  # a missing arm
            self.run_verdict(self.arms([('win', 'win')] * 50)[:-1])
        with self.assertRaises(ValueError):  # another candidate in one arm
            self.run_verdict(self.arms([('win', 'win')] * 50, mon_e2={'candidate': {'c': 2}}))
        with self.assertRaises(ValueError):  # cycle 1 takes no history
            self.run_verdict(self.arms([('win', 'win')] * 50), history=[str(self.tactical)])
        paths = self.arms([('win', 'win')] * 50)
        record = json.loads(Path(paths[0]).read_text())
        record['games'][0]['signature'] = 'x'
        Path(paths[0]).write_text(json.dumps(record))
        with self.assertRaises(ValueError):  # a game of another setup
            self.run_verdict(paths)


def play_args(tmp, policy, value, parent_policy, parent_value, **kw):
    base = {'cycle': 1, 'arm': 'official', 'candidate_policy': policy, 'candidate_value': value,
            'parent_policy': parent_policy, 'parent_value': parent_value, 'device': 'cpu', 'workers': 1,
            'jsonl': Path(tmp) / 'arena.jsonl', 'output': Path(tmp) / 'o.json', 'allow_dirty': True,
            'allow_stand_in': True, 'frozen_policy': None, 'pairs': 1, 'max_plies': 7}
    return argparse.Namespace(**{**base, **kw})


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class GameTest(unittest.TestCase):
    """Real TB-V3 / fast5 agents with random stand-in networks of the frozen architecture."""

    def setUp(self):
        from tests.test_tb_v3_net import write_models

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy, self.value = write_models(Path(self.tmp.name) / 'c', seed=1)
        self.parent_policy, self.parent_value = write_models(Path(self.tmp.name) / 'p', seed=2)

    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def args(self, tmp, **kw):
        return play_args(tmp, kw.pop('candidate_policy', self.policy), kw.pop('candidate_value', self.value),
                         kw.pop('parent_policy', self.parent_policy), kw.pop('parent_value', self.parent_value), **kw)

    def test_official_games_and_resume(self):
        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            first = ar.play(self.args(tmp), log=quiet)
            self.assertEqual(len(first['games']), 2)
            g = first['games'][0]
            self.assertEqual((g['length'], g['result'], g['moves'][:3]), (7, 'truncated', first['openings'][0]['moves']))
            self.assertTrue(all(m['route'] for m in g['tested_moves']))
            self.assertEqual(first['identity']['seed'], 8731)
            self.assertNotEqual(first['identity']['candidate'], first['identity']['parent'])
            again = ar.play(self.args(tmp), log=quiet)  # resumed: nothing new, the same games
            self.assertEqual([x['moves'] for x in again['games']], [x['moves'] for x in first['games']])
            self.assertEqual(len(Path(tmp, 'arena.jsonl').read_text().splitlines()), 2)
            with self.assertRaises(ValueError):  # another arm in the same file: refused, never mixed
                ar.play(self.args(tmp, arm='p_only'), log=quiet)

    def test_p_only_uses_the_parent_value(self):
        with TemporaryDirectory() as tmp:
            out = ar.play(self.args(tmp, arm='p_only', max_plies=5), log=lambda *a, **k: None)
            tested = out['identity']['tested']
            self.assertEqual(tested['models']['policy_checkpoint_sha256'],
                             out['identity']['candidate']['policy_checkpoint_sha256'])
            self.assertEqual(tested['models']['value_checkpoint_sha256'],
                             out['identity']['parent']['value_checkpoint_sha256'])

    def test_monitoring_opponent(self):
        with TemporaryDirectory() as tmp:
            out = ar.play(self.args(tmp, arm='mon_v8_full', max_plies=5), log=lambda *a, **k: None)
            self.assertEqual(out['identity']['opponent']['kind'], 'v8:full')
            self.assertEqual(len(out['games']), 2)

    def test_refusals(self):
        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):  # the candidate is the parent
                ar.play(self.args(tmp, parent_policy=self.policy, parent_value=self.value), log=quiet)
            with self.assertRaises(ValueError):  # cycle 1's parent must be the frozen models
                ar.play(self.args(tmp, allow_stand_in=False), log=quiet)
            with self.assertRaises(ValueError):  # the H5 opponent needs the frozen H3
                ar.play(self.args(tmp, arm='mon_puct_policy'), log=quiet)

    def test_illegal_move_is_a_recorded_loss(self):
        class Forbidden:
            diagnostics = None

            def select_move(self, game):
                return game.history[0]  # an occupied point

        task = {**ar.build_tasks('official', 1, 1)[0][1],  # tested side white: to move after the 3-stone opening
                'context': {'models': {'candidate': {'policy': '', 'value': ''}, 'parent': {}},
                            'device': 'cpu', 'signature': 's'}}
        row = ar.play_game(task, make_tested=lambda side, seed: Forbidden(), make_opponent=lambda seed: Forbidden())
        self.assertEqual((row['result'], row['illegal']['side'], row['illegal']['color']), ('loss', 'tested', 'white'))
        row = ar.play_game({**task, **ar.build_tasks('official', 1, 1)[0][0]}, make_tested=lambda side, seed: Forbidden(),
                           make_opponent=lambda seed: Forbidden())
        self.assertEqual((row['result'], row['illegal']['side']), ('win', 'opponent'))

    def test_value_and_policy_probes(self):
        probes = ar.value_probes(self.value)
        self.assertEqual(set(probes) >= {'P93', 'P94', 'P92', 'pass'}, True)
        block = ar.must_block(self.policy)
        self.assertEqual((block['anchor'], block['total'] > 0), (9, True))

    def test_tactical_record(self):
        from unittest import mock

        seen = []

        def e1(side, device, workers, label, log=print):
            seen.append((label, side['policy']))
            return {'metrics': {'unsound_witness': 0, 'false_veto': 0, 'rescue_rate': {'value': 0.5},
                                'veto_recall': {'value': 1.0}}}

        with TemporaryDirectory() as tmp, mock.patch.object(ar, 'e1_suite', e1):
            out = ar.tactical(self.args(tmp), log=lambda *a, **k: None)
        self.assertEqual(seen, [('candidate', str(self.policy)), ('parent', str(self.parent_policy))])
        self.assertEqual(set(out['judge']['items']), {'unsound_witness == 0', 'P93 / P94 value sign',
                                                      'must_block >= B400 anchor'})
        self.assertEqual(set(out['files']), {'value_probes', 'policy_probes', 'e1_manifest'})

    def test_x2_candidate_against_the_fake_engine(self):
        import scripts.run_external_match as runner

        info = x2.candidate_info(1, self.policy, self.value, 'cpu')
        self.assertTrue(info['label'].startswith('h7c:c1:'))
        self.assertFalse(info['provenance']['canonical'])
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / 'x2.json'
            argv = ['--ours', 'h7c', '--policy-checkpoint', str(self.policy), '--value-checkpoint', str(self.value),
                    '--value-device', 'cpu', '--engine', sys.executable, '--engine-arg', FAKE, '--seed', '1',
                    '--allow-unreserved-seed', '--pairs', '1', '--output', str(out)]
            self.assertEqual(runner.main(argv, h7c=info), 0)
            record = json.loads(out.read_text())
            self.assertEqual(record['ours'], info['label'])
            self.assertEqual(record['ours_provenance']['h7c']['model_hash'], info['provenance']['model_hash'])
            self.assertEqual(record['summary']['raw_scored_games'], 2)


class X2CommandTest(unittest.TestCase):
    def test_runner_refuses_h7c_without_the_tool(self):
        import scripts.run_external_match as runner

        err = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(err):
            runner.main(['--ours', 'h7c', '--engine', sys.executable, '--engine-arg', FAKE, '--seed', '1',
                         '--allow-unreserved-seed', '--pairs', '1'])
        self.assertIn('h7c_external', err.getvalue())

    def test_fixed_options(self):
        args = argparse.Namespace(cycle=2, policy_checkpoint='p', value_checkpoint='v', value_device='cuda')
        argv = x2.runner_argv(args, ['--engine', 'x', '--workers', '4'])
        self.assertEqual(argv[argv.index('--seed') + 1], '8712')
        self.assertEqual(argv[argv.index('--pairs') + 1], '25')
        for bad in (['--seed', '8711'], ['--pairs=1'], ['--allow-unreserved-seed'], ['--ours', 'tbv3']):
            with self.assertRaises(ValueError):
                x2.runner_argv(args, bad)


if __name__ == '__main__':
    unittest.main()
