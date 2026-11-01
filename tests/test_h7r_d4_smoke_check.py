"""R2-D4 smoke check: real smoke folders made by ``h7r_d4.run`` on synthetic sources (heavy dependencies replaced as in
``test_h7r_d4``), then the read-only check: verdicts, refusals of altered files, untouched inputs, log timing."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid import h7r_d4 as d4
import h7r_d4 as cli_top  # the module the check imports (scripts/ on the path)
from scripts import h7r_d4 as cli
from scripts import h7r_d4_smoke_check as chk
from tests.test_h7c_labels import game

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json').read_text())
SMOKE_EXPECTED = {'replay_plies': 12, 'eligible_plies': 8, 'warmup_plies': 4}  # two fixture games of 6 plies
GIT = {'commit': 'fixture', 'dirty': False}


def tree_hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


class Smoke:
    """One smoke run of the real wrapper in ``root``; ``disagree`` gives the seed models another move at ply 8."""

    def __init__(self, root: Path, disagree=True, branch=None, outcome=None):
        import h7c_labels as lab
        self.root, self.outcome = root, outcome
        c = deepcopy(CONTRACT)
        c['epsilon_result'] = str(ROOT / c['epsilon_result'])
        self.contract = c
        self.games = {i: game(i) for i in range(4)}
        info = {'game_ids': [lab.game_id(g) for g in self.games.values()],
                'game': [i for i in range(4) for _ in range(4)], 'ply': [5, 6, 7, 8] * 4}
        self.manifest = cli.eligible_manifest(c, 's', list(self.games.values()), info)
        self.sample = root / chk.SAMPLE
        self.sample.parent.mkdir(parents=True)
        self.sample.write_text(json.dumps(self.manifest))
        self.approval = root / 'execution-smoke.json'
        self.approval.write_text(json.dumps({'format': 'h7r-d4-execution-v1', 'contract_sha256': 's',
                                             'mode': 'smoke', 'approved': True}))
        self.identities = {n: {'value_checkpoint_sha256': m['value_checkpoint_sha256'],
                               'value_metadata_sha256': m['value_metadata_sha256'],
                               'frozen_state_sha256': {'recomputed': c['frozen_state_sha256']}}
                           for n, m in c['models'].items()}
        self.models = {n: {} for n in c['models']}
        self.folder = root / chk.SMOKE_DIR
        self.disagree, self.branch = disagree, branch

    def replay(self, tasks):
        rows = []
        for t in tasks:
            plies = []
            for p in t['game']['plies']:
                sets = {n: {k: deepcopy(p[k]) for k in ('route', 'engine_move', 'visits')} for n in self.models}
                if self.disagree and p['ply'] == 8:
                    for n in self.models:
                        if n != 'c5':
                            sets[n]['engine_move'] = [7, 6]
                plies.append({'ply': p['ply'], 'color': p['color'], 'actor_move': p['engine_move'],
                              'actor_route': p['route'], 'sets': sets})
            rows.append({'game_index': t['game_index'], 'identity_sha256': t['identity_sha256'], 'plies': plies})
        return rows

    @contextlib.contextmanager
    def patches(self):
        c = self.contract
        with contextlib.ExitStack() as stack:
            for target, kwargs in [
                    ((cli, 'ROOT'), {'new': self.root}),
                    ((cli, 'contract_at'), {'return_value': (c, 's')}),
                    ((cli, 'git_state'), {'return_value': GIT}),
                    ((cli, 'committed'), {'return_value': True}),
                    ((cli, 'inputs'), {'return_value': (self.manifest, self.games, {})}),
                    ((cli, 'checked_models'), {'return_value': (self.models, self.identities)}),
                    ((cli, 'ProcessPoolExecutor'),
                     {'side_effect': lambda max_workers, mp_context: ThreadPoolExecutor(max_workers)}),
                    ((cli, 'replay_chunk'), {'side_effect': self.replay})]:
                stack.enter_context(patch.object(*target, **kwargs))
                if target[1] in ('git_state', 'inputs', 'committed'):
                    stack.enter_context(patch.object(cli_top, target[1], **kwargs))
            for target, kwargs in [
                    ('analysis.tb_v3_fast5.runtime_hash', {'return_value': c['runtime_identity']['runtime_hash']}),
                    ('torch.cuda.get_device_name', {'return_value': 'fixture'}),
                    ('h7c_adjudicate.engine_spec', {'return_value': {'identity': c['rapfi_identity']}}),
                    ('hybrid.h7c_nmask.committed', {'return_value': True}),
                    ('hybrid.h7r_d4.load_contract', {'return_value': (c, 's')})]:
                stack.enter_context(patch(target, **kwargs))
            if self.branch is not None:
                stack.enter_context(patch('h7c_adjudicate._chunk',
                                          side_effect=lambda tasks: [self.branch(t) for t in tasks]))
            if self.outcome is not None:
                for module in (cli, cli_top):
                    stack.enter_context(patch.object(module, 'validate_branch',
                                                     side_effect=lambda *a: self.outcome(*a)))
            stack.enter_context(patch.object(chk, 'EXPECTED', SMOKE_EXPECTED))
            yield

    def run(self):
        args = SimpleNamespace(contract='unused', cache='unused', sample=self.sample, approval=self.approval,
                               mode='smoke', model=[], device='cuda', workers=2, rapfi_dir='unused',
                               out_dir=self.folder)
        with self.patches():
            return cli.run(args)

    def check(self, output='check.json', **change):
        args = SimpleNamespace(approval=self.approval, cache='unused', smoke_dir=self.folder, log=None,
                               lock_log=None, workers=6, output=self.root / output)
        for k, v in change.items():
            setattr(args, k, v)
        with self.patches():
            return chk.check(args, root=self.root)


def branch_record(status='VALID', reason=None, continuation=(), score=1.0, seconds=0.5):
    def make(task):
        return {'key': task['key'], 'order': task['order'], 'identity_sha256': task['identity_sha256'],
                'ply': len(task['moves']), 'mover': 'black' if len(task['moves']) % 2 == 0 else 'white',
                'move': task['move'], 'continuation': [list(m) for m in continuation], 'forbid_checked': 0,
                'status': status, 'reason': reason, 'score': score if status == 'VALID' else None,
                'length': len(task['moves']) + 1 + len(continuation), 'seconds': seconds}
    return make


def outcome_of(kind, reason=None):
    def judge(record, task, policy):
        return {'kind': kind, 'score': {'resolved': 1.0, 'structural': 0.5}.get(kind), 'reason': reason}
    return judge


class VerdictTest(unittest.TestCase):
    def test_real_branch_path_terminal_fixed_moves_and_untouched_inputs(self):
        with TemporaryDirectory() as tmp:
            smoke = Smoke(Path(tmp))
            run = smoke.run()
            self.assertEqual((run['reading'], run['branches']), ('SMOKE', 2))
            before = tree_hashes(smoke.folder)
            report = smoke.check()
            self.assertEqual(report['integrity'], 'PASS', report['failures'])
            self.assertEqual(report['verdict'], 'PASS_WITHOUT_RAPFI')  # every fixed move ends the game
            tally = report['branch_tally']
            self.assertEqual((tally['VALID'], tally['fixed_move_terminal'], tally['continuation']), (2, 2, 0))
            self.assertEqual(report['plies'], SMOKE_EXPECTED)
            self.assertTrue(report['actor_check']['reproduced'])
            self.assertEqual(tree_hashes(smoke.folder), before)  # the run files are only read
            self.assertEqual(report['lock_test'], {'status': 'NOT_TESTED'})
            self.assertEqual(report['resume_test'], {'status': 'NOT_TESTED'})
            self.assertEqual(report['timing']['status'], 'UNKNOWN')
            with self.assertRaisesRegex(chk.Failure, 'exists'):  # never overwritten
                smoke.check()

    def test_no_branch_means_no_branch_journal(self):
        with TemporaryDirectory() as tmp:
            smoke = Smoke(Path(tmp), disagree=False)
            smoke.run()
            self.assertFalse((smoke.folder / 'branches.jsonl').exists())
            report = smoke.check()
            self.assertEqual((report['verdict'], report['branches_expected']), ('PASS_WITHOUT_RAPFI', 0))
            (smoke.folder / 'branches.jsonl').write_text('')
            report = smoke.check('again.json')
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertIn('present / listed mismatch', report['failures'][0])

    def test_verdict_by_branch_outcome(self):
        cases = [
            ('continuation', branch_record(continuation=[(0, 0), (0, 1)]), outcome_of('resolved'), 'PASS'),
            ('structural', branch_record('INVALID', 'abnormal_end', continuation=[(0, 0)]),
             outcome_of('structural', 'abnormal_end'), 'PASS'),
            ('timeout', branch_record('INVALID', 'timeout', continuation=[(0, 0)]),
             outcome_of('unresolved', 'timeout'), 'FAIL'),
            ('forbid', branch_record('INVALID', 'forbid_mismatch'), outcome_of('unresolved', 'forbid_mismatch'),
             'FAIL'),
        ]
        for name, branch, outcome, verdict in cases:
            with self.subTest(name), TemporaryDirectory() as tmp:
                smoke = Smoke(Path(tmp), branch=branch, outcome=outcome)
                smoke.run()
                report = smoke.check()
                self.assertEqual(report['verdict'], verdict, report['failures'])
                tally = report['branch_tally']
                if name == 'structural':
                    self.assertEqual((tally['structural_draw'], tally['continuation']), (2, 2))
                if name == 'timeout':
                    self.assertEqual(tally['execution_failures']['timeout'], 2)
                    self.assertEqual(tally['continuation'], 0)  # an unresolved branch is no continuation

    def test_undefined_invalid_and_unmarked_structural_fail(self):
        with TemporaryDirectory() as tmp:
            smoke = Smoke(Path(tmp), branch=branch_record(), outcome=outcome_of('resolved'))
            smoke.run()
            smoke.outcome = lambda *a: (_ for _ in ()).throw(ValueError('INVALID reason is a data error'))
            self.assertEqual(smoke.check()['verdict'], 'FAIL')
            smoke.outcome = outcome_of('structural', 'abnormal_end')  # a VALID record called structural
            report = smoke.check('second.json')
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertIn('original INVALID mark', report['failures'][0])


class TamperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.smoke = Smoke(Path(self.tmp.name))
        self.smoke.run()
        self.pristine = self.smoke.folder.with_name('pristine')
        shutil.copytree(self.smoke.folder, self.pristine)
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def restore(self):
        shutil.rmtree(self.smoke.folder)
        shutil.copytree(self.pristine, self.smoke.folder)

    def failed(self, message):
        self.n += 1
        report = self.smoke.check(f'check{self.n}.json')
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertTrue(any(message in f for f in report['failures']), report['failures'])
        self.restore()

    def summary(self, change):
        path = self.smoke.folder / 'summary.json'
        body = json.loads(path.read_text())
        change(body)
        path.write_text(json.dumps(body))

    def rewrite(self, name, lines_fn, fix_summary=True):
        path = self.smoke.folder / name
        lines = path.read_bytes().split(b'\n')[:-1]
        path.write_bytes(lines_fn(lines))
        if fix_summary:  # a forger who also updates the summary hash
            self.summary(lambda s: s['files'].__setitem__(name, d4.file_sha256(path)))

    def test_summary_identity_and_shas(self):
        self.summary(lambda s: s['actor_check'].__setitem__('same_move', 0))
        self.failed('actor check differs')
        self.summary(lambda s: s.__setitem__('observations', 99))
        self.failed('observation count')
        self.summary(lambda s: s.__setitem__('reading', 'NOT_ESTABLISHED'))
        self.failed('not a smoke summary')
        body = json.loads((self.smoke.folder / 'identity.json').read_text())
        body['mode'] = 'full'
        (self.smoke.folder / 'identity.json').write_text(json.dumps(body))
        self.failed('not bound to identity.json')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines) + b'\n\n', fix_summary=False)
        self.failed('SHA differs from the summary')
        self.smoke.approval.write_text(json.dumps({'format': 'h7r-d4-execution-v1', 'contract_sha256': 'other',
                                                   'mode': 'smoke', 'approved': True}))
        self.failed('not the smoke approval')
        self.smoke.approval.write_text(json.dumps({'format': 'h7r-d4-execution-v1', 'contract_sha256': 's',
                                                   'mode': 'smoke', 'approved': True, 'note': 'x'}))
        self.failed('another contract, approval')  # another approval file than the run bound

    def test_journals_duplicates_missing_torn_and_corrupt(self):
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines + lines[:1]) + b'\n')
        self.failed('recorded twice')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines[:1]) + b'\n')
        self.failed('missing records')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines))
        self.failed('torn or unterminated')
        self.rewrite('branches.jsonl', lambda lines: b'not json\n' + b'\n'.join(lines) + b'\n')
        self.failed('not a JSON record')
        before = (self.smoke.folder / 'replay.jsonl').read_bytes()
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines) + b'\n{"torn"', fix_summary=False)
        torn = (self.smoke.folder / 'replay.jsonl').read_bytes()
        self.smoke.check('torn.json')
        self.assertEqual((self.smoke.folder / 'replay.jsonl').read_bytes(), torn)  # never cut or repaired
        self.assertNotEqual(torn, before)
        self.restore()

    def test_actor_reproduction_and_branch_set(self):
        def break_actor(lines):
            first = json.loads(lines[0])
            first['plies'][0]['sets']['c5']['visits'] = []
            return b'\n'.join([json.dumps(first).encode()] + lines[1:]) + b'\n'
        self.rewrite('replay.jsonl', break_actor)
        self.failed('reproduction')

        def foreign_branch(lines):
            other = json.loads(lines[0])
            other['key'] = '0' * 64
            return b'\n'.join(lines[1:] + [json.dumps(other).encode()]) + b'\n'
        self.rewrite('branches.jsonl', foreign_branch)
        self.failed('unknown record')
        self.rewrite('branches.jsonl', lambda lines: b'\n'.join(lines[:1]) + b'\n')
        self.summary(lambda s: s.__setitem__('branches', 1))
        self.failed('branch count')

    def test_eligible_list_and_commit(self):
        self.smoke.sample.write_text(json.dumps({**self.smoke.manifest, 'rows': 1}))
        self.failed('another contract, approval, list')
        self.restore()
        self.smoke.sample.write_text(json.dumps(self.smoke.manifest))
        args = SimpleNamespace(approval=self.smoke.approval, cache='unused', smoke_dir=self.smoke.folder, log=None,
                               lock_log=None, workers=6, output=self.smoke.root / 'later.json')
        with self.smoke.patches(), patch.object(cli_top, 'git_state', return_value={'commit': 'later', 'dirty': False}):
            report = chk.check(args, root=self.smoke.root)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('is not the checked commit', report['failures'][0])


class LogTest(unittest.TestCase):
    def write(self, folder, name, lines):
        path = Path(folder) / name
        path.write_text(''.join(f'{t} {text}\n' if t is not None else f'{text}\n' for t, text in lines))
        return path

    def full_session(self, folder, name='s1.log', base=1000.0):
        return self.write(folder, name, [
            (base, '__START__'), (base + 1, '[d4-preflight] checking sealed H-dev and eligible positions'),
            (base + 21, '[d4-preflight] PASS 396 games / 15985 eligible rows; five model identities'),
            (base + 22, '[d4-replay] resume 0/2, remaining 2'), (base + 52, '[d4-replay] 12/12 units  100.0%'),
            (base + 53, '[d4-rapfi] resume 0/2, remaining 2'), (base + 60, '[d4-rapfi] 2/2 units  100.0%'),
            (base + 75, '{"reading": "SMOKE", "summary": "x"}'), (base + 76, '__EXIT__ 0')])

    def test_stage_spans_and_reference(self):
        with TemporaryDirectory() as tmp:
            session = chk.read_session(self.full_session(tmp))
            t = chk.timing([session], {'a': {'seconds': 2.0}, 'b': {'seconds': 3.0}}, 6, 6)
            s = t['sessions'][0]['seconds']
            self.assertEqual((s['total'], s['preflight'], s['identity_and_lock'], s['replay'], s['observations'],
                              s['rapfi_elapsed'], s['revalidation_and_summary']), (76, 20, 1, 30, 1, 7, 15))
            self.assertEqual(t['branch_worker_seconds']['sum'], 5.0)
            self.assertIn('worker time', t['branch_worker_seconds']['note'])
            self.assertAlmostEqual(t['reference']['replay_seconds_per_engine_ply_all_models'], 5.0)
            self.assertIn('reference only', t['reference']['kind'])

    def test_incomplete_logs_give_null_not_guesses(self):
        with TemporaryDirectory() as tmp:
            path = self.write(tmp, 'cut.log', [(1000, '__START__'), (1001, '[d4-preflight] checking x'),
                                               (None, 'an untimed line'), (1030, '[d4-replay] resume 0/2, remaining 2')])
            t = chk.timing([chk.read_session(path)], {}, 6, 6)
            s = t['sessions'][0]['seconds']
            self.assertEqual((s['total'], s['preflight'], s['replay'], s['rapfi_elapsed']), (None, None, None, None))
            self.assertEqual(t['sessions'][0]['untimed_lines'], 1)
            self.assertIsNone(t['reference'])
            self.assertIsNone(t['branch_worker_seconds'])

    def test_lock_and_resume_evidence(self):
        with TemporaryDirectory() as tmp:
            session = chk.read_session(self.full_session(tmp))
            probe = Path(tmp) / 'lock.log'
            probe.write_text('LOCK_REFUSED 1030.5 confirmation directory is already running\n')
            self.assertEqual(chk.lock_test(probe, [session])['status'], 'TESTED_PASS')
            probe.write_text('LOCK_REFUSED 1010.0\n')  # before the run held the lock
            self.assertEqual(chk.lock_test(probe, [session])['status'], 'UNKNOWN')
            probe.write_text('LOCK_ACQUIRED 1030.5\n')
            self.assertEqual(chk.lock_test(probe, [session])['status'], 'TESTED_FAIL')
            self.assertEqual(chk.lock_test(None, [session])['status'], 'NOT_TESTED')
            first = chk.read_session(self.write(tmp, 'a.log', [
                (900, '__START__'), (905, '[d4-replay] resume 0/2, remaining 2'), (930, '__EXIT__ 130')]))
            second = chk.read_session(self.write(tmp, 'b.log', [
                (1000, '__START__'), (1022, '[d4-replay] resume 1/2, remaining 1'), (1076, '__EXIT__ 0')]))
            self.assertEqual(chk.resume_test([first, second])['status'], 'TESTED')
            self.assertEqual(chk.resume_test([second])['status'], 'NOT_TESTED')
            self.assertEqual(chk.resume_test([session, session])['status'], 'UNKNOWN')
            t = chk.timing([first, second], {}, 6, 6)
            self.assertEqual(len(t['sessions']), 2)
            self.assertIsNone(t['reference'])  # a resumed replay is no throughput sample


if __name__ == '__main__':
    unittest.main()
