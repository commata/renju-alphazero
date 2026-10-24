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


def write_candidate(directory, parent_policy, parent_value, cycle=1, seed=1):
    """Stand-in candidate models whose metadata name the cycle and the parent's self-play as their source."""
    from analysis import tb_v3_net
    from tests.test_tb_v3_net import write_models

    policy, value = write_models(directory, seed=seed)
    source = tb_v3_net.model_identity(parent_policy, parent_value)
    for path, extra in ((policy.with_suffix('.json'), {'stage': 'H7-c-policy'}),
                        (value.with_name('best.json'), {'h7c_stage': 'H7-c-value'})):
        meta = json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps({**meta, **extra, 'cycle': cycle, 'source_models': source}), encoding='utf-8')
    return policy, value


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
            return judge_side(unsound, hits, rescue, value_side(fresh() if sign else fresh(win=(70, 93))))
        ok = ar.tactical_judge(side(rescue=0.5, hits=10), side())  # 2 below the parent: allowed
        self.assertEqual((ok['verdict'], ok['must_block_min']), ('PASS', 10))
        self.assertEqual(ok['regressions_reported'], ['must_block_hits', 'rescue_rate'])  # reported, not gated
        self.assertEqual(ar.tactical_judge(side(hits=9), side(hits=10))['verdict'], 'PASS')  # the anchor 9 binds
        for bad, parent in ((side(unsound=1), side()), (side(sign=False), side()), (side(hits=8), side(hits=8)),
                            (side(hits=9), side(hits=12))):  # 3 below the parent
            self.assertEqual(ar.tactical_judge(bad, parent)['verdict'], 'FAIL')

    def test_fresh_rule_boundaries(self):
        self.assertTrue(fresh(win=(16, 20), loss=(16, 20))['pass'])          # exactly 0.80, exactly 20: PASS
        self.assertFalse(fresh(win=(15, 20), loss=(20, 20))['pass'])         # 0.75
        self.assertFalse(fresh(win=(19, 19), loss=(20, 20))['pass'])         # 19 probes: too few, even all right
        self.assertFalse(fresh(win=(20, 20), loss=(15, 20))['pass'])         # the loss sign alone fails
        one = fresh(win=(16, 20))
        self.assertEqual((one['win']['correct'], one['win']['probes'], one['win']['accuracy']), (16, 20, 0.8))

    def test_v2_value_item_is_the_fresh_gate_not_p93_p94(self):
        item = ar.FRESH_ITEM
        wrong_p9x = judge_side(value=value_side(p93=False, p94=False))  # the parent's own state: both wrong
        self.assertTrue(ar.tactical_judge(wrong_p9x, judge_side())['items'][item])
        right_p9x = judge_side(value=value_side(fresh(loss=(60, 81)), p93=True, p94=True))
        judged = ar.tactical_judge(right_p9x, judge_side())
        self.assertEqual((judged['items'][item], judged['verdict']), (False, 'FAIL'))
        self.assertNotIn('P93 / P94 value sign', judged['items'])
        evidence = judged['evidence']
        self.assertEqual((evidence['rule_version'], evidence['value_device'], evidence['inference']),
                         ('h7c-tactical-v2', 'cpu', 'd4_average'))
        self.assertEqual(evidence['p93_p94_diagnostic']['candidate']['P94']['pass'], True)   # recorded, not gated
        self.assertEqual(evidence['fresh']['parent_diagnostic']['loss']['correct'], 78)
        self.assertEqual(evidence['must_block'], {'candidate': 12, 'parent': 12, 'min': 10})
        self.assertEqual(set(judged['items']), {'unsound_witness == 0', item,
                                                'must_block >= max(B400 anchor, parent - 2)'})
        with self.assertRaises(ValueError):  # candidate and parent measured differently
            ar.tactical_judge(judge_side(value=value_side(device='cuda')), judge_side())


def fresh(win=(93, 93), loss=(78, 81)):
    """A ``value_probes['fresh']``-like record from (correct, probes) per sign, through the real rule."""
    rows = [{'expected_sign': 1, 'correct': i < win[0]} for i in range(win[1])] + \
           [{'expected_sign': -1, 'correct': i < loss[0]} for i in range(loss[1])]
    summary = {'probes': rows, 'win': {'games': win[1], 'mean_value': 0.5}, 'loss': {'games': loss[1], 'mean_value': -0.5}}
    return {**ar.fresh_rule(summary), 'file_sha256': 'f' * 64}


def value_side(fresh_record=None, p93=False, p94=False, device='cpu'):
    return {'device': device, 'inference': 'd4_average', 'fresh': fresh_record or fresh(),
            'p93_p94': {'P93': {'d4_mean': 0.1 if p93 else -0.01, 'expected_sign': 1, 'pass': p93},
                        'P94': {'d4_mean': -0.1 if p94 else 0.98, 'expected_sign': -1, 'pass': p94}}}


def judge_side(unsound=0, hits=12, rescue=0.6, value=None):
    return {'e1': {'metrics': {'unsound_witness': unsound, 'false_veto': 0,
                               'rescue_rate': {'value': rescue, 'n': 'x'}, 'veto_recall': {'value': 1.0}}},
            'value_probes': value or value_side(), 'must_block': {'hits': hits, 'anchor': 9}}


def judge_record(**items):
    """A v2 tactical judge for the verdict tests."""
    judged = ar.tactical_judge(judge_side(), judge_side())
    judged['items'].update(items)
    return judged


def write_arm(tmp, arm, results, **identity):
    ident = {'arm': arm, 'cycle': 1, 'git': {'commit': 'abc'}, 'candidate': {'c': 1}, 'parent': {'p': 1},
             'champion_record_sha256': None, **identity}
    sig = ar.signature(ident)
    games = []
    for pair, (black, white) in enumerate(results):
        for color, result in (('black', black), ('white', white)):
            illegal = {'side': 'opponent'} if result == 'opponent_illegal' else None
            games.append({**game(pair, color, result, illegal), 'signature': sig})
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
            'champion_record_sha256': None,
            'rule_version': ar.TACTICAL_RULE, 'judge': judge_record()}))

    def arms(self, official, monitoring=None, tmp=None, **extra):
        tmp = tmp or self.tmp.name
        paths = [write_arm(tmp, 'official', official, **extra.get('all', {}))]
        for arm in ar.ARMS:
            if arm != 'official':
                paths.append(write_arm(tmp, arm, monitoring or [('win', 'loss')] * 25,
                                       **{**extra.get('all', {}), **extra.get(arm, {})}))
        return paths

    def run_verdict(self, arms, history=(), tactical=None):
        return ar.verdict(argparse.Namespace(arms=arms, tactical=tactical or self.tactical, history=list(history)))

    def test_official_opponent_illegal_makes_the_arena_invalid(self):
        out = self.run_verdict(self.arms([('win', 'win')] * 49 + [('opponent_illegal', 'win')]))
        self.assertEqual((out['official']['valid'], out['official']['opponent_illegal_pairs']), (False, [49]))
        self.assertEqual((out['promotion']['decision'], out['promotion']['reason']), ('KEEP_CHAMPION', 'OFFICIAL_INVALID'))
        self.assertIsNone(out['promotion']['improvement_lower'])  # never a convergence bound
        self.assertEqual(out['next_champion'], {'p': 1})

    def test_monitoring_opponent_illegal_leaves_the_pair_out(self):
        out = self.run_verdict(self.arms([('win', 'win')] * 50,
                                         monitoring=[('win', 'loss')] * 24 + [('opponent_illegal', 'win')]))
        self.assertEqual(out['promotion']['decision'], 'PROMOTE')
        e2 = out['monitoring']['mon_e2']
        self.assertEqual((e2['complete_pairs'], e2['opponent_illegal_pairs'], e2['score']), (24, [24], 0.5))
        self.assertEqual(out['next_champion'], {'c': 1})

    def test_history_chain(self):
        c1 = Path(self.tmp.name) / 'c1_verdict.json'
        c1.write_text(json.dumps(self.run_verdict(self.arms([('win', 'win')] * 50))))  # promoted: champion {'c': 1}
        sha = ar._file_sha(c1)
        with TemporaryDirectory() as tmp:
            def cycle2(parent, champion=sha, history=(str(c1),)):
                tactical = Path(tmp) / 'tactical2.json'
                tactical.write_text(json.dumps({
                    'cycle': 2, 'git': {'commit': 'abc'}, 'candidate': {'c': 2}, 'parent': parent,
                    'champion_record_sha256': champion,
                    'rule_version': ar.TACTICAL_RULE, 'judge': judge_record()}))
                ident = {'cycle': 2, 'candidate': {'c': 2}, 'parent': parent, 'champion_record_sha256': champion}
                return self.run_verdict(self.arms([('win', 'loss')] * 50, tmp=tmp, all=ident), history, tactical)
            out = cycle2({'c': 1})
            self.assertEqual((out['cycle'], out['promotion']['decision'], out['next_champion']),
                             (2, 'KEEP_CHAMPION', {'c': 1}))
            self.assertEqual(out['convergence']['cycles'], [1, 2])
            with self.assertRaises(ValueError):  # the parent is not cycle 1's champion
                cycle2({'p': 1})
            with self.assertRaises(ValueError):  # the history file is not the champion record the runs named
                cycle2({'c': 1}, champion='f' * 64)
            with self.assertRaises(ValueError):  # cycle 2 needs the history of cycle 1
                cycle2({'c': 1}, history=())

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
        with self.assertRaises(ValueError):  # another champion record in one arm
            self.run_verdict(self.arms([('win', 'win')] * 50, mon_e2={'champion_record_sha256': 'f' * 64}))
        paths = self.arms([('win', 'win')] * 50)
        record = json.loads(Path(paths[0]).read_text())
        record['games'][0]['signature'] = 'x'
        Path(paths[0]).write_text(json.dumps(record))
        with self.assertRaises(ValueError):  # a game of another setup
            self.run_verdict(paths)


def dummy_models(directory, tag):
    directory.mkdir(parents=True)
    for name in ('p.pt', 'p.json', 'best.pt', 'best.json'):
        (directory / name).write_text(f'{tag}:{name}', encoding='utf-8')
    return directory / 'p.pt', directory / 'best.pt'


class NoCandidateTest(unittest.TestCase):
    def test_record(self):
        from analysis import tb_v3_net

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, value = dummy_models(tmp / 'parent', 'parent')
            records = {}
            for name, path, verdict in (('policy', policy, 'CANDIDATE'), ('value', value, 'NO_CANDIDATE')):
                records[name] = tmp / f'{name}_record.json'
                records[name].write_text(json.dumps({'cycle': 1, 'verdict': verdict, 'inputs': {
                    'parent_checkpoint_sha256': ar._file_sha(path)}}), encoding='utf-8')
            args = argparse.Namespace(cycle=1, policy_record=records['policy'], value_record=records['value'],
                                      parent_policy=policy, parent_value=value, champion=None, history=[],
                                      allow_stand_in=True)
            out = ar.no_candidate(args)
            parent = tb_v3_net.model_identity(policy, value)
            self.assertEqual((out['promotion']['decision'], out['promotion']['reason']), ('KEEP_CHAMPION', 'NO_CANDIDATE'))
            self.assertEqual((out['next_champion'], out['candidate'], out['promotion']['improvement_lower']),
                             (parent, None, None))
            self.assertEqual(out['training']['verdicts'], {'policy': 'CANDIDATE', 'value': 'NO_CANDIDATE'})
            self.assertFalse(out['convergence']['converged'])
            # a later cycle chains onto it: its history and champion record
            c1 = tmp / 'c1.json'
            c1.write_text(json.dumps(out), encoding='utf-8')
            for name in records:
                data = json.loads(records[name].read_text())
                records[name].write_text(json.dumps({**data, 'cycle': 2}), encoding='utf-8')
            two = ar.no_candidate(argparse.Namespace(**{**vars(args), 'cycle': 2, 'champion': c1,
                                                        'history': [str(c1)]}))
            self.assertEqual((two['cycle'], two['champion_record_sha256']), (2, ar._file_sha(c1)))
            self.assertEqual(two['convergence']['improvement_lower'], [None, None])  # never two stalled cycles
            with self.assertRaises(ValueError):  # cycle 2 without the previous verdict
                ar.no_candidate(argparse.Namespace(**{**vars(args), 'cycle': 2, 'history': [str(c1)]}))
            with self.assertRaises(ValueError):  # records of another cycle
                ar.no_candidate(args)
            for name in records:
                data = json.loads(records[name].read_text())
                records[name].write_text(json.dumps({**data, 'cycle': 1, 'verdict': 'CANDIDATE'}), encoding='utf-8')
            with self.assertRaises(ValueError):  # both trainings gave a candidate: the Arena runs
                ar.no_candidate(args)
            other, _ = dummy_models(tmp / 'other', 'other')
            with self.assertRaises(ValueError):  # the training did not start from the parent
                data = json.loads(records['value'].read_text())
                records['value'].write_text(json.dumps({**data, 'verdict': 'NO_CANDIDATE'}), encoding='utf-8')
                ar.no_candidate(argparse.Namespace(**{**vars(args), 'parent_policy': other}))


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
        self.parent_policy, self.parent_value = write_models(Path(self.tmp.name) / 'p', seed=2)
        self.policy, self.value = write_candidate(Path(self.tmp.name) / 'c', self.parent_policy, self.parent_value)

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
            with self.assertRaises(ValueError):  # the candidate's metadata name cycle 1
                ar.check_models(self.args(tmp, cycle=2), False)
            other_policy, other_value = write_candidate(Path(tmp) / 'x', self.policy, self.value)
            with self.assertRaises(ValueError):  # trained on another self-play than the parent's
                ar.check_models(self.args(tmp, candidate_policy=other_policy, candidate_value=other_value), False)
            mixed = write_candidate(Path(tmp) / 'y', self.parent_policy, self.parent_value, seed=4)
            meta = json.loads(mixed[1].with_name('best.json').read_text())
            mixed[1].with_name('best.json').write_text(json.dumps({**meta, 'source_models': {'x': 1}}))
            with self.assertRaises(ValueError):  # policy and value of two sources
                ar.check_models(self.args(tmp, candidate_policy=mixed[0], candidate_value=mixed[1]), False)

    def test_cycle_two_parent_is_the_previous_champion(self):
        from analysis import tb_v3_net

        with TemporaryDirectory() as tmp:
            # cycle 1 promoted self.policy / self.value; cycle 2 trains on their self-play
            c2_policy, c2_value = write_candidate(Path(tmp) / 'c2', self.policy, self.value, cycle=2, seed=7)
            champion = Path(tmp) / 'c1_verdict.json'
            champion.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1, 'next_champion':
                                            tb_v3_net.model_identity(self.policy, self.value)}))
            args = self.args(tmp, cycle=2, candidate_policy=c2_policy, candidate_value=c2_value,
                             parent_policy=self.policy, parent_value=self.value, champion=champion)
            models = ar.check_models(args, False)
            self.assertEqual(models['champion_record_sha256'], ar._file_sha(champion))
            with self.assertRaises(ValueError):  # no previous verdict
                ar.check_models(self.args(tmp, cycle=2, candidate_policy=c2_policy, candidate_value=c2_value,
                                          parent_policy=self.policy, parent_value=self.value), False)
            champion.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1, 'next_champion':
                                            tb_v3_net.model_identity(self.parent_policy, self.parent_value)}))
            with self.assertRaises(ValueError):  # the verdict kept the old champion, not this parent
                ar.check_models(args, False)
            with self.assertRaises(ValueError):  # cycle 1 takes no --champion
                ar.check_models(self.args(tmp, champion=champion), False)

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
        self.assertEqual((row['result'], row['illegal']['side']), ('opponent_illegal', 'opponent'))  # never scored

    def test_value_and_policy_probes(self):
        probes = ar.value_probes(self.value, 'cpu')
        self.assertTrue(set(probes) >= {'device', 'inference', 'fresh', 'p93_p94', 'p92'})
        self.assertEqual(probes['fresh']['win']['probes'] + probes['fresh']['loss']['probes'], 174)
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
        self.assertEqual(set(out['judge']['items']), {'unsound_witness == 0', ar.FRESH_ITEM,
                                                      'must_block >= max(B400 anchor, parent - 2)'})
        self.assertEqual(set(out['files']), {'value_probes', 'fresh_probes', 'policy_probes', 'e1_manifest'})
        self.assertEqual(out['rule_version'], 'h7c-tactical-v2')
        for who in ('candidate_result', 'parent_result'):  # the value probes ran on --device, both sides alike
            probes = out[who]['value_probes']
            self.assertEqual((probes['device'], probes['inference']), ('cpu', 'd4_average'))
            self.assertEqual(probes['fresh']['file_sha256'], out['files']['fresh_probes'])
            self.assertEqual((probes['fresh']['win']['probes'], probes['fresh']['loss']['probes']), (93, 81))
            self.assertEqual(set(probes['p93_p94']), {'P93', 'P94'})
        self.assertEqual(out['judge']['evidence']['value_device'], 'cpu')

    def test_x2_candidate_against_the_fake_engine(self):
        from analysis import tb_v3_net
        import scripts.run_external_match as runner

        info = x2.candidate_info(1, self.policy, self.value, 'cpu')
        self.assertTrue(info['label'].startswith('h7c:c1:'))
        self.assertEqual(info['provenance']['source_models'], tb_v3_net.model_identity(self.parent_policy,
                                                                                       self.parent_value))
        with self.assertRaises(ValueError):  # the candidate's metadata name cycle 1
            x2.candidate_info(2, self.policy, self.value, 'cpu')
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



class TacticalRuleVersionTest(unittest.TestCase):
    def test_v1_tactical_record_refused(self):
        with TemporaryDirectory() as tmp:
            tactical = Path(tmp) / 'tactical.json'
            tactical.write_text(json.dumps({'cycle': 1, 'git': {'commit': 'abc'}, 'candidate': {'c': 1},
                                            'parent': {'p': 1}, 'champion_record_sha256': None,
                                            'judge': {'items': {'P93 / P94 value sign': True}, 'compare': {},
                                                      'regressions_reported': []}}))
            arms = [write_arm(tmp, arm, [('win', 'win')] * ar.ARMS[arm]['pairs']) for arm in ar.ARMS]
            with self.assertRaises(ValueError) as caught:
                ar.verdict(argparse.Namespace(arms=arms, tactical=tactical, history=[]))
            self.assertIn('h7c-tactical-v2', str(caught.exception))


class FallbackVerdictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c1 = Path(self.tmp.name) / 'c1.json'
        self.c1.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1, 'next_champion': {'p': 1},
                                       'promotion': {'improvement_lower': -0.2}, 'tactical': {'verdict': 'FAIL'}}))
        self.ident = {'cycle': 2, 'candidate': {'c': 2}, 'parent': {'p': 1},
                      'champion_record_sha256': ar._file_sha(self.c1)}

    def tactical(self, mode):
        path = Path(self.tmp.name) / f'tactical_{mode}.json'
        path.write_text(json.dumps({**self.ident, 'git': {'commit': 'abc'}, 'candidate_mode': mode,
                                    'rule_version': ar.TACTICAL_RULE, 'judge': judge_record()}))
        return path

    def arms(self, mode, names):
        return [write_arm(self.tmp.name, arm, [('win', 'win')] * ar.ARMS[arm]['pairs'],
                          **self.ident, candidate_mode=mode) for arm in names]

    def run_verdict(self, arms, mode):
        return ar.verdict(argparse.Namespace(arms=arms, tactical=self.tactical(mode), history=[str(self.c1)]))

    def test_fallback_cycle(self):
        played = ('official', 'mon_v8_full', 'mon_puct_policy', 'mon_e2')
        out = self.run_verdict(self.arms(ar.FALLBACK, played), ar.FALLBACK)
        self.assertEqual(out['candidate_mode'], 'VALUE_ONLY_FALLBACK')
        self.assertEqual(out['diagnosis'], {'p_only': 'NOT_APPLICABLE', 'v_only': 'SAME_AS_OFFICIAL'})
        self.assertEqual(set(out['fingerprints']), set(played))
        self.assertEqual((out['promotion']['decision'], out['next_champion']), ('PROMOTE', {'c': 2}))
        self.assertIsNotNone(out['promotion']['improvement_lower'])  # a valid fallback Arena counts for convergence
        with self.assertRaises(ValueError):  # p_only / v_only are not played in a fallback cycle
            self.run_verdict(self.arms(ar.FALLBACK, played + ('p_only',)), ar.FALLBACK)
        with self.assertRaises(ValueError):  # a joint cycle needs every arm
            self.run_verdict(self.arms(ar.JOINT, played), ar.JOINT)
        with self.assertRaises(ValueError):  # the arms and the tactical record in different modes
            self.run_verdict(self.arms(ar.FALLBACK, played), ar.JOINT)
        mixed = self.arms(ar.FALLBACK, played[:-1]) + self.arms(ar.JOINT, played[-1:])
        with self.assertRaises(ValueError):  # arms of two modes
            self.run_verdict(mixed, ar.FALLBACK)


class NoCandidateFallbackBoundaryTest(unittest.TestCase):
    def test_policy_only_failure_needs_the_fallback_arena_from_cycle_2(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, value = dummy_models(tmp / 'parent', 'parent')
            records = {}
            for name, path, verdict in (('policy', policy, 'NO_CANDIDATE'), ('value', value, 'CANDIDATE')):
                records[name] = tmp / f'{name}.json'
                records[name].write_text(json.dumps({'cycle': 2, 'verdict': verdict, 'inputs': {
                    'parent_checkpoint_sha256': ar._file_sha(path)}}))
            c1 = tmp / 'c1.json'
            from analysis import tb_v3_net
            c1.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1,
                                      'next_champion': tb_v3_net.model_identity(policy, value),
                                      'promotion': {'improvement_lower': -0.2}, 'tactical': {'verdict': 'FAIL'}}))
            args = argparse.Namespace(cycle=2, policy_record=records['policy'], value_record=records['value'],
                                      parent_policy=policy, parent_value=value, champion=c1, history=[str(c1)],
                                      allow_stand_in=True)
            with self.assertRaises(ValueError) as caught:
                ar.no_candidate(args)
            self.assertIn('VALUE_ONLY_FALLBACK', str(caught.exception))
            for name in records:  # value NO_CANDIDATE stays a no-candidate cycle
                data = json.loads(records[name].read_text())
                records[name].write_text(json.dumps({**data, 'verdict': 'CANDIDATE' if name == 'policy'
                                                     else 'NO_CANDIDATE'}))
            self.assertEqual(ar.no_candidate(args)['promotion']['reason'], 'NO_CANDIDATE')


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class FallbackCandidateTest(unittest.TestCase):
    """The fallback candidate: the parent policy byte for byte, the cycle's value, the policy NO_CANDIDATE record."""

    def setUp(self):
        from analysis import tb_v3_net
        from tests.test_tb_v3_net import write_models

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.parent_policy, self.parent_value = write_models(tmp / 'p', seed=2)
        _, self.value = write_candidate(tmp / 'c2', self.parent_policy, self.parent_value, cycle=2, seed=5)
        self.record = tmp / 'policy_record.json'
        self.write_record(2, 'NO_CANDIDATE')
        self.champion = tmp / 'c1.json'
        self.champion.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1, 'next_champion':
                                             tb_v3_net.model_identity(self.parent_policy, self.parent_value)}))

    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def write_record(self, cycle, verdict, parent=None):
        self.record.write_text(json.dumps({'cycle': cycle, 'verdict': verdict, 'inputs': {
            'parent_checkpoint_sha256': parent or ar._file_sha(self.parent_policy)}}))

    def args(self, tmp, **kw):
        base = {'candidate_mode': ar.FALLBACK, 'policy_record': self.record, 'champion': self.champion, 'cycle': 2}
        return play_args(tmp, self.parent_policy, self.value, self.parent_policy, self.parent_value, **{**base, **kw})

    def test_fallback_candidate(self):
        source = ar.check_candidate(self.parent_policy, self.value, 2, ar.FALLBACK, self.record)
        from analysis import tb_v3_net
        self.assertEqual(source, tb_v3_net.model_identity(self.parent_policy, self.parent_value))
        with self.assertRaises(ValueError):  # a joint candidate needs a cycle policy
            ar.check_candidate(self.parent_policy, self.value, 2)
        with self.assertRaises(ValueError):  # not before cycle 2
            ar.check_candidate(self.parent_policy, self.value, 1, ar.FALLBACK, self.record)
        with self.assertRaises(ValueError):  # no NO_CANDIDATE record
            ar.check_candidate(self.parent_policy, self.value, 2, ar.FALLBACK, None)
        other_policy, _ = write_candidate(Path(self.tmp.name) / 'o', self.parent_policy, self.parent_value, 2, 9)
        with self.assertRaises(ValueError):  # the policy must be the parent's, byte for byte
            ar.check_candidate(other_policy, self.value, 2, ar.FALLBACK, self.record)
        for cycle, verdict, parent in ((2, 'CANDIDATE', None), (3, 'NO_CANDIDATE', None), (2, 'NO_CANDIDATE', 'e' * 64)):
            self.write_record(cycle, verdict, parent)
            with self.assertRaises(ValueError):  # the record must be this cycle's NO_CANDIDATE from the parent
                ar.check_candidate(self.parent_policy, self.value, 2, ar.FALLBACK, self.record)

    def test_fallback_arena_and_x2(self):
        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            out = ar.play(self.args(tmp, max_plies=5), log=quiet)
            self.assertEqual((out['identity']['candidate_mode'], out['identity']['seed']), ('VALUE_ONLY_FALLBACK', 8732))
            self.assertEqual(out['identity']['tested']['models']['policy_checkpoint_sha256'],
                             ar._file_sha(self.parent_policy))
            for arm in ('p_only', 'v_only'):
                with self.assertRaises(ValueError):
                    ar.play(self.args(tmp, arm=arm, jsonl=Path(tmp) / f'{arm}.jsonl'), log=quiet)
        info = x2.candidate_info(2, self.parent_policy, self.value, 'cpu', ar.FALLBACK, self.record)
        self.assertEqual(info['provenance']['candidate_mode'], 'VALUE_ONLY_FALLBACK')
        with self.assertRaises(ValueError):
            x2.candidate_info(2, self.parent_policy, self.value, 'cpu')  # joint mode refuses the parent policy

if __name__ == '__main__':
    unittest.main()
