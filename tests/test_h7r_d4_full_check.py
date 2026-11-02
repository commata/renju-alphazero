"""R2-D4 full check: full runs made by the real ``h7r_d4.run`` on synthetic sources (heavy dependencies replaced as in
``test_h7r_d4_smoke_check``), then the read-only check: integrity and the recomputed reading, refusals of altered
records, INVALID handling, untouched inputs."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid import h7r_d4 as d4
from scripts import h7r_d4_full_check as full
from tests.test_h7r_d4_smoke_check import Smoke, branch_record, outcome_of, tree_hashes


class Full(Smoke):
    def __init__(self, root: Path, **kw):
        super().__init__(root, **kw)
        self.approval.write_text(json.dumps({'format': 'h7r-d4-execution-v1', 'contract_sha256': 's',
                                             'mode': 'full', 'approved': True}))
        self.folder = root / full.FULL_DIR

    def run(self):
        import h7r_d4 as unused  # noqa: F401  (the module the check uses is patched in Smoke.patches)
        from scripts import h7r_d4 as cli
        args = SimpleNamespace(contract='unused', cache='unused', sample=self.sample, approval=self.approval,
                               mode='full', model=[], device='cuda', workers=2, rapfi_dir='unused', out_dir=self.folder)
        with self.patches():
            return cli.run(args)

    def check(self, output='check.json', **change):
        args = SimpleNamespace(approval=self.approval, cache='unused', full_dir=self.folder, log=None, lock_log=None,
                               output=self.root / output)
        for k, v in change.items():
            setattr(args, k, v)
        with self.patches():
            return full.check(args, root=self.root)


class FullCheckTest(unittest.TestCase):
    def test_normal_result_reading_and_untouched_inputs(self):
        with TemporaryDirectory() as tmp:
            run = Full(Path(tmp))
            summary = run.run()
            before = tree_hashes(run.folder)
            report = run.check()
            self.assertEqual(report['integrity'], 'PASS', report['failures'])
            self.assertEqual(report['reading']['joint'], summary['reading'])  # a reading, whatever it is
            self.assertIn(report['reading']['joint'], ('ALL_THREE_BETTER', 'ALL_THREE_WORSE', 'MIXED_DIRECTION',
                                                       'PARTIAL_SIGNAL', 'NOT_ESTABLISHED'))
            self.assertEqual((report['games'], report['plies']), (4, {'eligible': 16, 'warmup': 8}))
            self.assertEqual((report['observations'], report['branches']), (16, 2))
            self.assertTrue(report['actor_check']['reproduced'])
            self.assertIsNone(report['reading']['comparisons']['c8:c5']['label95'])
            self.assertEqual(tree_hashes(run.folder), before)
            self.assertEqual((report['resume_test']['status'], report['lock_test']['status']), ('NOT_TESTED',) * 2)
            with self.assertRaisesRegex(full.Failure, 'exists'):
                run.check()

    def test_inconclusive_reading_is_not_an_integrity_failure(self):
        with TemporaryDirectory() as tmp:
            run = Full(Path(tmp))
            summary = run.run()
            self.assertEqual(summary['reading'], 'NOT_ESTABLISHED')  # 4 fixture games: too few observations
            self.assertEqual(run.check()['integrity'], 'PASS')

    def test_known_execution_failures_stay_unresolved(self):
        with TemporaryDirectory() as tmp:
            run = Full(Path(tmp), branch=branch_record('INVALID', 'timeout'), outcome=outcome_of('unresolved', 'timeout'))
            summary = run.run()
            report = run.check()
            self.assertEqual(report['integrity'], 'PASS', report['failures'])
            self.assertEqual(report['branch_outcomes'], {'resolved': 0, 'structural': 0, 'unresolved': 2})
            self.assertEqual(report['not_resolved_by_reason'], {'timeout': 2})
            self.assertEqual(report['branch_outcomes'], summary['branch_outcomes'])
            run.outcome = lambda *a: (_ for _ in ()).throw(ValueError('undefined INVALID'))
            self.assertEqual(run.check('undefined.json')['integrity'], 'FAIL')

    def test_structural_draw_needs_its_invalid_mark(self):
        with TemporaryDirectory() as tmp:
            run = Full(Path(tmp), branch=branch_record(), outcome=outcome_of('structural', 'abnormal_end'))
            run.run()
            report = run.check()
            self.assertEqual(report['integrity'], 'FAIL')
            self.assertIn('original INVALID mark', report['failures'][0])


class TamperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.run_ = Full(Path(self.tmp.name))
        self.run_.run()
        self.pristine = self.run_.folder.with_name('pristine')
        shutil.copytree(self.run_.folder, self.pristine)
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def restore(self):
        shutil.rmtree(self.run_.folder)
        shutil.copytree(self.pristine, self.run_.folder)

    def failed(self, message):
        self.n += 1
        report = self.run_.check(f'check{self.n}.json')
        self.assertEqual(report['integrity'], 'FAIL')
        self.assertTrue(any(message in f for f in report['failures']), report['failures'])
        self.restore()

    def summary(self, change):
        path = self.run_.folder / 'summary.json'
        body = json.loads(path.read_text())
        change(body)
        path.write_text(json.dumps(body))

    def rewrite(self, name, lines_fn):
        path = self.run_.folder / name
        lines = path.read_bytes().split(b'\n')[:-1]
        path.write_bytes(lines_fn(lines))
        self.summary(lambda s: s['files'].__setitem__(name, d4.file_sha256(path)))  # a forger fixes the hash

    def test_reading_and_statistics_must_match(self):
        self.summary(lambda s: s.__setitem__('reading', 'ALL_THREE_BETTER'))
        self.failed('reading')
        self.summary(lambda s: s['comparisons']['seed31001:c5'].__setitem__('label95', 'BETTER'))
        self.failed('label95')
        self.summary(lambda s: s['comparisons']['seed31002:c5'].__setitem__('observations', 99))
        self.failed('observations')
        self.summary(lambda s: s['sensitivity']['seed31003:c5']['all'].__setitem__('share', 0.123))
        self.failed('share')

        def shift(s):
            lower = s['comparisons']['seed31001:c5']['intervals']['0.95']['lower']
            if lower['ci'] is not None:
                lower['ci'][0] += 1e-6
            else:
                lower['undefined'] += 1
        self.summary(shift)
        self.failed('intervals')
        self.summary(lambda s: s['branch_outcomes'].__setitem__('resolved', 0))
        self.failed('branch_outcomes')

    def test_identity_summary_and_journals(self):
        body = json.loads((self.run_.folder / 'identity.json').read_text())
        body['mode'] = 'smoke'
        (self.run_.folder / 'identity.json').write_text(json.dumps(body))
        self.failed('not bound to identity.json')
        self.summary(lambda s: s.__setitem__('reading', 'SMOKE'))
        self.failed('not a full summary')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines + lines[:1]) + b'\n')
        self.failed('recorded twice')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines[:-1]) + b'\n')
        self.failed('missing records')
        self.rewrite('replay.jsonl', lambda lines: b'\n'.join(lines))
        self.failed('torn or unterminated')
        self.rewrite('branches.jsonl', lambda lines: b'\n'.join(lines[:1]) + b'\n')
        self.failed('missing records')

        def foreign(lines):
            other = json.loads(lines[0])
            other['key'] = '0' * 64
            return b'\n'.join(lines + [json.dumps(other).encode()]) + b'\n'
        self.rewrite('branches.jsonl', foreign)
        self.failed('unknown record')

    def test_actor_and_approval(self):
        def break_actor(lines):
            first = json.loads(lines[0])
            first['plies'][0]['sets']['c5']['engine_move'] = [14, 14]
            return b'\n'.join([json.dumps(first).encode()] + lines[1:]) + b'\n'
        self.rewrite('replay.jsonl', break_actor)
        self.failed('reproduction')
        self.run_.approval.write_text(json.dumps({'format': 'h7r-d4-execution-v1', 'contract_sha256': 's',
                                                  'mode': 'smoke', 'approved': True}))
        self.failed('not the full approval')


if __name__ == '__main__':
    unittest.main()
