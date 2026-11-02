"""R2-D4 Rapfi path probe and record copies: the probe runs the real branch path (``h7c_adjudicate._chunk`` in a
spawn pool, one fresh engine process per branch) with the fake protocol engine standing in for Rapfi; the checks
and refusals; log copies with their manifest and the exact staged-set check in a scratch git repository."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h7r_d4 as cli_top
from scripts import h7r_d4_rapfi_probe as probe
from scripts import h7r_d4_records as records

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / probe.PLAN).read_text())
GIT = {'commit': 'fixture', 'dirty': False}


class Probe:
    def __init__(self, tmp: Path, *engine_args, plan_change=None):
        from tests.test_h7c_adjudicate import spec
        self.tmp = tmp
        plan = deepcopy(PLAN)
        plan['out_dir'] = str(tmp / 'probe')  # absolute: outside the repository
        if plan_change:
            plan_change(plan)
        self.plan_path = tmp / 'plan.json'
        self.plan_path.write_text(json.dumps(plan))
        self.plan = plan
        self.spec = spec('--seed', '5', *engine_args)
        self.spec['identity'] = plan['rapfi_identity']

    def patches(self, identity=None):
        stack = [patch.object(cli_top, 'git_state', return_value=GIT),
                 patch('hybrid.h7c_nmask.committed', return_value=True),
                 patch('h7c_adjudicate.engine_spec', return_value={**self.spec, 'identity': identity or self.spec['identity']})]
        return stack

    def run(self, identity=None):
        args = SimpleNamespace(plan=self.plan_path, out_dir=self.tmp / 'probe', rapfi_dir='unused', root=ROOT)
        ps = self.patches(identity)
        for p in ps:
            p.start()
        try:
            return probe.run(args, log=lambda *a, **k: None)
        finally:
            for p in reversed(ps):
                p.stop()

    def check(self, name='check.json'):
        args = SimpleNamespace(plan=self.plan_path, out_dir=self.tmp / 'probe', output=self.tmp / name, root=ROOT)
        ps = self.patches()
        for p in ps:
            p.start()
        try:
            return probe.check(args)
        finally:
            for p in reversed(ps):
                p.stop()


class PlanTest(unittest.TestCase):
    def test_committed_plan(self):
        with patch('hybrid.h7c_nmask.committed', return_value=True):
            plan, sha, contract = probe.checked_plan(ROOT / probe.PLAN, ROOT)
        self.assertEqual(sorted(b['mover'] for b in plan['branches']), ['black', 'white'])
        self.assertEqual(plan['rapfi_identity'], contract['rapfi_identity'])
        self.assertEqual(plan['workers'], 2)
        self.assertNotIn('smoke', plan['out_dir'])
        self.assertNotIn('full', plan['out_dir'])

    def test_refusals(self):
        def refused(change, message, committed=True):
            with TemporaryDirectory() as tmp:
                p = Probe(Path(tmp), plan_change=change)
                with patch('hybrid.h7c_nmask.committed', return_value=committed), \
                        self.assertRaisesRegex(ValueError, message):
                    probe.checked_plan(p.plan_path, ROOT)

        refused(lambda p: None, 'not committed', committed=False)
        refused(lambda p: p.update(approved=False), 'not an approved')
        refused(lambda p: p['branches'][0].update(move=[7, 7]), 'illegal')
        terminal = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [0, 6]]  # (7, 11) makes five

        def immediate(p):
            p['branches'][0].update(prefix=terminal, move=[7, 11])
        refused(immediate, 'ends the game')
        refused(lambda p: p['branches'][1].update(mover='black'), 'one black and one white')
        refused(lambda p: p['branches'][0].update(key='0' * 64), 'key differs')
        refused(lambda p: p['helper_sources'][0].update(sha256='0' * 64), 'differs from the plan')
        refused(lambda p: p.update(out_dir='runs/h7r/R2-D4-BASELINE-1/smoke/x'), 'outside the D4 smoke')
        refused(lambda p: p['rapfi_identity'].update(extra=1), 'another Rapfi')


class RunTest(unittest.TestCase):
    def test_normal_continuation_through_the_spawn_pool(self):
        with TemporaryDirectory() as tmp:
            p = Probe(Path(tmp))
            summary = p.run()
            self.assertEqual(summary['branches'], 2)
            self.assertEqual(summary['identity']['mp_context'], 'spawn')
            report = p.check()
            self.assertEqual(report['verdict'], 'PASS', report['failures'])
            for row in report['branches'].values():
                self.assertEqual(row['kind'], 'resolved')
                self.assertGreater(row['continuation'], 0)
            self.assertGreater(report['branches']['white-mover']['forbid_checked'], 0)
            self.assertIn('parallel', report['timing']['note'])
            with self.assertRaisesRegex(ValueError, 'exists'):  # never run again into the folder
                p.run()
            with self.assertRaisesRegex(ValueError, 'exists'):  # never overwrite a check
                p.check()

            journal = Path(tmp) / 'probe/branches.jsonl'
            original = journal.read_bytes()
            rows = [json.loads(x) for x in original.splitlines()]
            rows[0]['continuation'] = rows[0]['continuation'][:-1]
            journal.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(p.check('tampered.json')['verdict'], 'FAIL')  # summary SHA differs
            summary_path = Path(tmp) / 'probe/summary.json'
            body = json.loads(summary_path.read_text())
            body['files']['branches.jsonl'] = probe.d4.file_sha256(journal)
            summary_path.write_text(json.dumps(body))
            forged = p.check('forged.json')  # the replay catches the shortened continuation
            self.assertEqual(forged['verdict'], 'FAIL')
            self.assertEqual(forged['integrity'], 'FAIL')

    def test_execution_failures_fail(self):
        for name, args, reason in (('crash', ('--mode', 'crash'), 'engine_error'),
                                   ('forbid-lie', ('--forbid-lie',), 'forbid_mismatch')):
            with self.subTest(name), TemporaryDirectory() as tmp:
                p = Probe(Path(tmp), *args)
                p.run()  # the run records what happened; the check judges it
                report = p.check()
                self.assertEqual(report['verdict'], 'FAIL')
                self.assertTrue(any(reason in f for f in report['failures']), report['failures'])

    def test_engine_identity_and_head(self):
        with TemporaryDirectory() as tmp:
            p = Probe(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'registered identity'):
                p.run(identity={'other': True})
            self.assertFalse((Path(tmp) / 'probe').exists())
            p.run()
            with patch.object(cli_top, 'git_state', return_value={'commit': 'later', 'dirty': False}):
                args = SimpleNamespace(plan=p.plan_path, out_dir=Path(tmp) / 'probe', output=Path(tmp) / 'c.json',
                                       root=ROOT)
                with patch('hybrid.h7c_nmask.committed', return_value=True):
                    report = probe.check(args)
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertIn('not the checked commit', report['failures'][0])


class RecordsTest(unittest.TestCase):
    def test_copy_logs_with_manifest(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / 'runs/logs'
            src.mkdir(parents=True)
            (src / 'smoke-1.log').write_text('1 __START__\n2 __EXIT__ 0\n')
            (src / 'other.log').write_text('not copied\n')
            body = records.copy_logs(src, 'smoke-*.log', tmp / 'records/run-logs', tmp / 'records/log_manifest.json',
                                     root=tmp)
            self.assertEqual([e['original'] for e in body['logs']], ['runs/logs/smoke-1.log'])
            self.assertEqual(body['logs'][0]['sha256'], records.sha256(src / 'smoke-1.log'))
            self.assertEqual((tmp / 'records/run-logs/smoke-1.log').read_bytes(), (src / 'smoke-1.log').read_bytes())
            self.assertTrue((src / 'smoke-1.log').exists())  # the original stays
            with self.assertRaisesRegex(ValueError, 'exists'):
                records.copy_logs(src, 'smoke-*.log', tmp / 'records/run-logs', tmp / 'records/log_manifest.json',
                                  root=tmp)
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                records.copy_logs(src, 'smoke-*.log', tmp / 'records/run-logs', tmp / 'records/m2.json', root=tmp)
            with self.assertRaisesRegex(ValueError, 'nothing copied'):
                records.copy_logs(src, 'none-*.log', tmp / 'records/x', tmp / 'records/m3.json', root=tmp)

    def test_records_manifest(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'records/full'
            (folder / 'run-logs').mkdir(parents=True)
            (folder / 'check.json').write_text('{"integrity": "PASS"}\n')
            (folder / 'run-logs/full-1.log').write_text('1 __START__\n')
            body = records.records_manifest(folder, folder / 'records_manifest.json')
            self.assertEqual([f['path'] for f in body['files']], ['check.json', 'run-logs/full-1.log'])
            self.assertEqual(body['files'][0]['sha256'], records.sha256(folder / 'check.json'))
            with self.assertRaisesRegex(ValueError, 'exists'):
                records.records_manifest(folder, folder / 'records_manifest.json')

    def test_verify_staged_exactly(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            env = {**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
                   'GIT_COMMITTER_EMAIL': 't@t'}
            git = lambda *a: subprocess.run(['git', *a], cwd=repo, check=True, capture_output=True, env=env)  # noqa: E731
            git('init', '-q')
            (repo / '.gitignore').write_text('logs/\n')
            (repo / 'tracked.txt').write_text('a\n')
            git('add', '.gitignore', 'tracked.txt')
            git('commit', '-q', '-m', 'base')
            (repo / 'records/run-logs').mkdir(parents=True)
            (repo / 'records/run-logs/s.log').write_text('x\n')
            (repo / 'records/log_manifest.json').write_text('{}\n')
            expected = ['records/log_manifest.json', 'records/run-logs/s.log']
            self.assertTrue(any('staged set differs' in p for p in records.verify_staged(expected, repo)))
            git('add', *expected)
            self.assertEqual(records.verify_staged(expected, repo), [])
            self.assertTrue(any('not staged' in p for p in records.verify_staged(expected + ['records/x.json'], repo)))
            (repo / 'records/logs').mkdir()
            (repo / 'records/logs/s.log').write_text('x\n')
            git('add', '-f', 'records/logs/s.log')
            problems = records.verify_staged(expected + ['records/logs/s.log'], repo)
            self.assertTrue(any('ignored by .gitignore' in p for p in problems))
            git('reset', '-q', 'records/logs/s.log')
            (repo / 'tracked.txt').write_text('changed\n')
            self.assertTrue(any('unstaged tracked changes' in p for p in records.verify_staged(expected, repo)))


if __name__ == '__main__':
    unittest.main()
