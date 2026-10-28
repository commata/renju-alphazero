"""H7-R R1 pilot (docs/h7-rl-plan.md 9.9): the fixed run rules, the suite ranking by the opening-cluster bootstrap,
the session log and soft budget, and the runner end to end with an in-process fake game and a fake clock.
No real Arena game, GPU run or pilot sample."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7r_eval as ev
from hybrid import h7r_openings as op
from hybrid import h7r_pilot as pilot
from rt30 import fast5
from scripts import h7r_pilot as runner
from tests.test_h7r_cli import CLEAN, QUIET, world
from tests.test_h7r_eval import STATS

ROOT = Path(__file__).resolve().parents[1]


def fake_game(task, make_tested=None, make_opponent=None, max_plies=None):
    """A complete game decided by the opening and the arm: the tested side wins as black, and also as white at
    some openings, so pair scores vary between arms and openings."""
    from renju import BLACK
    from tests.h7r_helpers import winning_moves
    ctx = task['context']
    black = task['tested_color'] == BLACK
    bonus = (sum(sum(m) for m in task['opening']) + len(ctx['arm'])) % 3 == 0
    result = 'win' if black or bonus else 'loss'
    tested = BLACK if black else -BLACK
    winner = tested if result == 'win' else -tested
    moves = winning_moves(task['opening'], winner)
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if black else 'white', 'result': result,
            'winner': winner, 'length': len(moves), 'moves': moves, 'illegal': None,
            'truncated': False, 'tested_moves': [], 'opponent_routes': {}, 'game_seconds': 1.0}


def pilot_world(tmp: Path) -> dict:
    """Two suites (A, B), each three small arms of 6 pairs and an S400-style control of 8 pairs."""
    w = world(tmp)
    second = op.build_suite('O0', 30, 8782)
    (tmp / 'suites/o0b.json').write_text(json.dumps(second))
    suites = {'A': {'path': 'suites/o0.json'}, 'B': {'path': 'suites/o0b.json'}}
    for s in suites.values():
        s['sha256'] = ev.file_sha256(tmp / s['path'])
    arms, order, rank_suites = [], [], {}
    for i, (name, suite) in enumerate(suites.items()):
        arm = lambda n, t, o, tb, pairs, seed, pairing: {  # noqa: E731
            'name': f'{name}_{n}', 'tested': {'model': t, 'budget': tb}, 'opponent': {'model': o, 'budget': 1},
            'pairs': pairs, 'seed': seed, 'pairing': pairing, 'suite': suite}
        small = [arm('cand_champ', 'cand', 'champ', 1, 6, 8781 + i, f'{name}_small'),
                 arm('cand_orig', 'cand', 'orig', 1, 6, 8781 + i, f'{name}_small'),
                 arm('champ_orig', 'champ', 'orig', 1, 6, 8781 + i, f'{name}_small')]
        control = arm('control', 'champ', 'champ', 8, 8, 8784 + i, f'{name}_control')
        arms += [control, *small]
        order += [control['name'], *(a['name'] for a in small)]
        rank_suites[name] = {'small_arms': [a['name'] for a in small], 'control_arm': control['name']}
    body = {'format': ev.CONTRACT_FORMAT, 'contract_id': 'T-PILOT', 'purpose': 'pilot', 'status': 'fixed',
            'models': w['body']['models'], 'arms': arms, 'stats': STATS,
            'suite_selection': {'rank_by': {'version': pilot.RANK_VERSION,
                                            'bootstrap': {'unit': 'history_id', 'method': 'percentile',
                                                          'samples': 2000, 'seed': 20261029, 'quantile': 0.025},
                                            'order': ['lower', 'estimate'], 'tie': 'joint', 'suites': rank_suites},
                                'colour_bias': {'gate': 'none', 'report': 'per arm'}},
            'positive_control': {'arms': [s['control_arm'] for s in rank_suites.values()],
                                 'detection': 'lower_one_sided > 0'},
            'planning': {'cost': {'s400_arm_cost_multiplier_assumed': 3.0}},
            'execution_plan': {'workers': 1, 'arm_order': order,
                               'budget': {'version': pilot.BUDGET_VERSION, 'soft_wall_hours': 14,
                                          'heartbeat_seconds': 60, 'accounting': 'sessions', 'checks': 'arm ends',
                                          'on_reach': 'finish the arm', 'after_reach': 'approval',
                                          'errors': 'stop'}}}
    path = tmp / 'contracts/pilot.json'
    path.write_text(json.dumps(body))
    w['contract'].unlink()  # only the pilot contract in this tree
    return {**w, 'contract': path, 'body': body, 'order': order}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class RulesTest(unittest.TestCase):
    def test_fixed_needs_every_rule_and_a_draft_may_leave_them_unset(self):
        with TemporaryDirectory() as tmp:
            w = pilot_world(Path(tmp))
            pilot.check_rules(w['body'], 'ok')
            ev.check_contract(w['body'], 'fixed', tmp, check_files=False)
            cases = (
                lambda b: b['suite_selection'].update(rank_by=None),
                lambda b: b['suite_selection']['rank_by']['bootstrap'].update(unit='arm pair'),
                lambda b: b['suite_selection']['rank_by'].update(tie='suite order'),
                lambda b: b['suite_selection'].update(colour_bias={'gate': 0.45, 'report': 'pooled'}),
                lambda b: b['execution_plan']['budget'].update(soft_wall_hours=None),
                lambda b: b['execution_plan']['budget'].update(heartbeat_seconds=600),
                lambda b: b['execution_plan'].update(arm_order=w['order'][:-1]),
                lambda b: b['execution_plan'].update(workers=0),
                lambda b: b['positive_control'].update(detection='score > 0.5'),
                lambda b: b['suite_selection']['rank_by']['suites']['A']['small_arms'].reverse() or
                b['arms'][1].update(seed=8790),
            )
            for mutate in cases:
                body = deepcopy(w['body'])
                mutate(body)
                with self.subTest(body=mutate), self.assertRaises(ValueError):
                    ev.check_contract(body, 'bad', tmp, check_files=False)
                ev.check_contract({**body, 'status': 'draft'}, 'draft', tmp, check_files=False)

    def test_committed_contracts(self):
        pilot_contract, _ = ev.load_contract(ROOT / 'docs/h7r/contracts/R1-PILOT-1.json', ROOT, check_files=False)
        rules = pilot.check_rules(pilot_contract, 'R1-PILOT-1')
        self.assertEqual((pilot_contract['status'], rules['workers'], rules['soft_wall_hours']), ('fixed', 6, 14.0))
        self.assertEqual(rules['order'][:4], ['O0_c5s400_vs_c5', 'O0_c8_vs_c5', 'O0_mask_vs_c5', 'O0_c5_vs_orig'])
        self.assertEqual(rules['bootstrap'], {'unit': 'history_id', 'method': 'percentile', 'samples': 20000,
                                              'seed': 20261029, 'quantile': 0.025})
        smoke, _ = ev.load_contract(ROOT / 'docs/h7r/contracts/R1-SMOKE-1.json', ROOT, check_files=False)
        self.assertFalse(pilot.is_run_by_plan(smoke))  # a fixed pilot without a plan keeps its plain play


def table(rows, valid=True):
    return {'pairs': [{'opening_id': i, 'pair_score': s, 'score_split': s != 0.5, 'move_split': m}
                      for i, s, m in rows], 'valid': valid, 'opponent_illegal_pairs': [] if valid else ['x']}


class RankingTest(unittest.TestCase):
    ARMS = ['a', 'b', 'c']

    def test_q_per_opening_alignment_and_refusals(self):
        t = {'a': table([('o1', 1.0, True), ('o2', 0.5, True)]), 'b': table([('o2', 0.5, False), ('o1', 0.5, True)]),
             'c': table([('o1', 0.0, True), ('o2', 0.75, True)])}
        rows, problems = pilot.opening_rows(t, self.ARMS, {n: 2 for n in self.ARMS})
        self.assertEqual(problems, [])
        self.assertEqual([(r['opening_id'], round(r['q'], 4), round(r['m'], 4)) for r in rows],
                         [('o1', 0.6667, 1.0), ('o2', 0.3333, 0.6667)])
        bad = {
            'missing arm': {k: v for k, v in t.items() if k != 'c'},
            'other openings': {**t, 'c': table([('o1', 0.0, True), ('o3', 0.0, True)])},
            'twice': {**t, 'c': table([('o1', 0.0, True), ('o1', 1.0, True)])},
            'short': {**t, 'c': table([('o1', 0.0, True)])},
            'invalid': {**t, 'c': table([('o1', 0.0, True), ('o2', 1.0, True)], valid=False)},
        }
        for name, tables in bad.items():
            with self.subTest(name):
                rows, problems = pilot.opening_rows(tables, self.ARMS, {n: 2 for n in self.ARMS})
                self.assertIsNone(rows)
                self.assertTrue(problems)

    def test_bootstrap_is_deterministic_and_flags_zero_variance(self):
        boot = {'samples': 2000, 'seed': 20261029, 'quantile': 0.025}
        values = [0.0, 1 / 3, 2 / 3, 1.0] * 15
        first, again = pilot.cluster_bootstrap(values, boot), pilot.cluster_bootstrap(values, boot)
        self.assertEqual(first, again)
        self.assertLess(first['lower'], first['estimate'])
        self.assertFalse(first['zero_variance'])
        flat = pilot.cluster_bootstrap([1 / 3] * 60, boot)
        self.assertEqual((flat['lower'], flat['estimate'], flat['zero_variance']), (0.333333, 0.333333, True))

    def test_rank_by_lower_then_estimate_with_joint_ranks(self):
        stats = {'O0': {'lower': 0.2, 'estimate': 0.3}, 'O1': {'lower': 0.2, 'estimate': 0.35},
                 'O2': {'lower': 0.2, 'estimate': 0.3}, 'O3': {'lower': 0.25, 'estimate': 0.26}}
        self.assertEqual([(r['suite'], r['rank']) for r in pilot.rank(stats)],
                         [('O3', 1), ('O1', 2), ('O0', 3), ('O2', 3)])


class BudgetTest(unittest.TestCase):
    def test_sessions_pauses_crashes_and_extensions(self):
        E = lambda event, s, t, **k: {'format': pilot.LOG_FORMAT, 'event': event, 'session': s, 'elapsed': t, **k}  # noqa
        events = [E('session_start', 's1', 0), E('heartbeat', 's1', 3600 * 5), E('session_end', 's1', 3600 * 6),
                  E('session_start', 's2', 0), E('heartbeat', 's2', 3600 * 7.5)]  # s2 killed after its last beat
        state = pilot.budget_state(events, 14)
        self.assertEqual((state['cumulative_hours'], state['sessions'], state['reached']), (13.5, 2, False))
        self.assertTrue(pilot.budget_state(events, 14, ('s3', 1800.0))['reached'])  # 14.0 h with a running session
        events.append({'format': pilot.LOG_FORMAT, 'event': 'extension', 'session': 's3', 'hours': 2,
                       'approval': 'owner, 2026-10-30 chat'})
        state = pilot.budget_state(events, 14, ('s3', 1800.0))
        self.assertEqual((state['limit_hours'], state['reached'], state['extensions'][0]['hours']), (16.0, False, 2))

    def test_log_repair_identity_and_damage(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sessions.jsonl'
            pilot.append(path, {'event': 'session_start', 'session': 's', 'elapsed': 0, 'commit': 'c',
                                'contract_sha256': 'x'})
            with path.open('ab') as out:
                out.write(b'{"format": "h7r-pilot')  # an interrupted append
            self.assertEqual(len(pilot.load_log(path, repair=False)), 1)
            self.assertTrue(path.read_bytes().endswith(b'pilot'))  # not repaired while a session runs
            self.assertEqual(len(pilot.load_log(path)), 1)
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            pilot.check_identity(pilot.load_log(path), 'c', 'x')
            with self.assertRaisesRegex(ValueError, 'one commit'):
                pilot.check_identity(pilot.load_log(path), 'other', 'x')
            path.write_text('garbage\n' + path.read_text())
            with self.assertRaisesRegex(ValueError, 'damaged'):
                pilot.load_log(path)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = pilot_world(Path(self.tmp.name))
        self.root = self.w['root']
        self.clock = Clock()
        self.hours = {}
        git = mock.patch.object(runner.cli, '_git', return_value=dict(CLEAN))
        git.start()
        self.addCleanup(git.stop)

    def args(self, **change):
        return argparse.Namespace(**{'contract': self.w['contract'], 'runs_dir': self.root / 'runs/pilot',
                                     'device': 'cpu', 'budget_extension_hours': None, 'approval': None,
                                     'heartbeat_seconds': 0, 'allow_dirty': True, 'threads_for_tests': True,
                                     'root': self.root, **change})

    def run_pilot(self, fail=None, **change):
        real = runner.cli.play

        def play(args, log=print, root=None):
            if args.arm == fail:
                raise RuntimeError('worker lost')
            out = real(args, log=log, root=root)
            self.clock.t += 3600 * self.hours.get(args.arm, 0.1)
            return out

        with mock.patch('h7c_arena.play_game', fake_game), mock.patch.object(runner.cli, 'play', play):
            return runner.run(self.args(**change), log=QUIET, clock=self.clock)

    def report(self):
        return runner.report(argparse.Namespace(contract=self.w['contract'], runs_dir=self.root / 'runs/pilot',
                                                root=self.root))

    def test_complete_run_order_report_and_resume_without_replaying(self):
        out = self.run_pilot()
        self.assertEqual((out['reason'], out['arms_complete'], out['arms_remaining']), ('complete', self.w['order'], []))
        events = pilot.load_log(self.root / 'runs/pilot/sessions.jsonl')
        self.assertEqual([e['arm'] for e in events if e['event'] == 'arm_start'], self.w['order'])
        rep = self.report()
        self.assertEqual(rep['status'], 'COMPLETE')
        self.assertEqual(sorted(r['suite'] for r in rep['ranking']), ['A', 'B'])
        for suite in ('A', 'B'):
            entry = rep['suites'][suite]
            self.assertEqual(entry['score_split']['openings'], 6)
            self.assertIn('detected', entry['positive_control'])
            self.assertIn('lower', entry['move_split'])
        self.assertEqual(set(rep['arms']['A_control']['colour']), {'tested_black_score', 'tested_white_score',
                                                                    'absolute_gap'})
        self.assertAlmostEqual(rep['budget']['cumulative_hours'], 0.8, places=2)
        again = self.run_pilot()  # a later session only re-checks the finished arms
        self.assertEqual(again['arms_complete'], self.w['order'])
        self.assertEqual(sum(e['event'] == 'arm_start' for e in pilot.load_log(self.root / 'runs/pilot/sessions.jsonl')),
                         len(self.w['order']))

    def test_budget_stops_after_the_active_arm_and_needs_an_approval_to_resume(self):
        self.hours = {self.w['order'][0]: 10, self.w['order'][1]: 5}
        out = self.run_pilot()
        self.assertEqual((out['reason'], out['arms_complete']), ('budget', self.w['order'][:2]))
        self.assertGreaterEqual(out['budget']['cumulative_hours'], 14)  # the active arm finished past 14 h
        self.assertIsNone(self.report()['ranking'])
        with self.assertRaisesRegex(ValueError, 'separate approval'):
            self.run_pilot()
        with self.assertRaisesRegex(ValueError, 'approval'):
            self.run_pilot(budget_extension_hours=3)
        before = (self.root / 'runs/pilot/sessions.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'does not reopen'):  # 15 h used, 14 + 1 h: nothing recorded
            self.run_pilot(budget_extension_hours=1, approval='owner in chat, 2026-10-30')
        self.assertEqual((self.root / 'runs/pilot/sessions.jsonl').read_bytes(), before)
        self.hours = {self.w['order'][2]: 1.5}
        out = self.run_pilot(budget_extension_hours=2, approval='owner in chat, 2026-10-30')
        self.assertEqual((out['reason'], out['arms_complete']), ('budget', self.w['order'][:3]))  # 16.5 >= 16 h
        events = pilot.load_log(self.root / 'runs/pilot/sessions.jsonl')
        self.assertEqual([e['approval'] for e in events if e['event'] == 'extension'], ['owner in chat, 2026-10-30'])

    def test_heartbeat_lines_and_events_while_arms_run(self):
        lines = []
        real = runner.cli.play

        def slow(args, log=print, root=None):
            import time
            time.sleep(0.15)
            return real(args, log=log, root=root)

        with mock.patch('h7c_arena.play_game', fake_game), mock.patch.object(runner.cli, 'play', slow):
            runner.run(self.args(heartbeat_seconds=0.05), log=lambda *a, **k: lines.append(' '.join(map(str, a))),
                       clock=self.clock)
        events = pilot.load_log(self.root / 'runs/pilot/sessions.jsonl')
        self.assertGreater(sum(e['event'] == 'heartbeat' for e in events), 3)
        beat = [x for x in lines if x.startswith('[h7r-pilot]') and 'pair-units' in x]
        self.assertTrue(beat and all('cumulative' in x and 'ETA' in x for x in beat))

    def test_an_error_stops_the_session_and_a_rerun_resumes(self):
        with self.assertRaisesRegex(RuntimeError, 'worker lost'):
            self.run_pilot(fail=self.w['order'][1])
        events = pilot.load_log(self.root / 'runs/pilot/sessions.jsonl')
        self.assertEqual([(e['arm'], e['status']) for e in events if e['event'] == 'arm_end'],
                         [(self.w['order'][0], 'complete'), (self.w['order'][1], 'error')])
        self.assertEqual(events[-1]['reason'], 'error')
        self.assertEqual(self.report()['status'], 'INCOMPLETE')
        self.assertEqual(self.run_pilot()['reason'], 'complete')

    def test_another_commit_and_direct_play_are_refused(self):
        with mock.patch('h7c_arena.play_game', fake_game):
            with self.assertRaisesRegex(ValueError, 'h7r_pilot.py run'):
                runner.cli.play(argparse.Namespace(contract=self.w['contract'], arm=self.w['order'][0], device='cpu',
                                                   workers=1, jsonl=self.root / 'runs/x.jsonl', allow_dirty=True,
                                                   threads_for_tests=True), log=QUIET, root=self.root)
        self.hours = {self.w['order'][0]: 20}
        self.run_pilot()
        with mock.patch.object(runner.cli, '_git', return_value={'commit': 'other', 'dirty': False}):
            with self.assertRaisesRegex(ValueError, 'one commit'):
                self.run_pilot(budget_extension_hours=1, approval='x')
        with self.assertRaisesRegex(ValueError, 'runs/ or data/'):
            self.run_pilot(runs_dir=self.root / 'elsewhere')


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class PreflightTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        from scripts import h7c_arena as ar
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_models_inferred_without_any_sample(self):
        with TemporaryDirectory() as tmp:
            w = pilot_world(Path(tmp))
            with mock.patch.object(runner.cli, '_git', return_value=dict(CLEAN)):
                out = runner.preflight(argparse.Namespace(contract=w['contract'], device='cpu'), log=QUIET,
                                       root=w['root'])
            self.assertTrue(out['pass'], out['checks'])
            self.assertEqual(set(out['models']), {'cand', 'champ', 'orig'})
            self.assertTrue(out['runtime']['fast5_installed'])
            self.assertIn('present', out['numpy'])
            self.assertFalse((Path(tmp) / 'runs').exists())  # nothing played, nothing written


if __name__ == '__main__':
    unittest.main()
