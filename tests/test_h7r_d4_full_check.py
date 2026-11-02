"""R2-D4 full check: full runs made by the real ``h7r_d4.run`` on synthetic sources (heavy dependencies replaced as in
``test_h7r_d4_smoke_check``), then the read-only check: integrity and the recomputed reading, refusals of altered
records, INVALID handling, untouched inputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
        for name, identity in self.identities.items():
            identity.update({k:self.contract['models'][name][k] for k in
                             ('policy_checkpoint_sha256', 'policy_metadata_sha256')})
            identity['frozen_state_sha256'].update(declared=self.contract['frozen_state_sha256'],
                                                   contract=self.contract['frozen_state_sha256'])
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
    def test_model_coverage_policy_and_frozen_declarations_are_required(self):
        for mutation in ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'declared', 'extra', 'malformed'):
            with self.subTest(mutation), TemporaryDirectory() as tmp:
                run = Full(Path(tmp))
                if mutation == 'extra': run.identities['extra'] = run.identities['c5']
                elif mutation == 'malformed': run.identities['c5'] = ['bad']
                elif mutation == 'declared': run.identities['c5']['frozen_state_sha256']['declared'] = '0'*64
                else: run.identities['c5'][mutation] = '0'*64
                run.run()
                report = run.check()
                self.assertEqual(report['integrity'], 'FAIL', mutation)
                self.assertIn('model', report['failures'][0])

    def test_missing_log_is_a_preserved_failure_report(self):
        with TemporaryDirectory() as tmp:
            run = Full(Path(tmp)); run.run()
            report = run.check(log=[str(Path(tmp)/'missing.log')])
            self.assertEqual(report['integrity'], 'FAIL')
            self.assertIn('FileNotFoundError', report['failures'][0])
            self.assertTrue((Path(tmp)/'check.json').is_file())
            report = run.check('lock-failure.json', lock_log=str(Path(tmp)/'missing-lock.log'))
            self.assertEqual(report['integrity'], 'FAIL')
            self.assertEqual(report['lock_test']['status'], 'UNKNOWN')

    def test_count_types_and_strict_json(self):
        self.assertTrue(full.differences(40.000000001, 40))
        self.assertTrue(full.differences(40.0, 40))
        self.assertFalse(full.differences(.1+1e-12, .1))
        for text in ('{"a":1,"a":2}', '{"unused":NaN}', '{"unused":1e999}'):
            with self.assertRaises(full.Failure): full.strict_json(text)

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
            self.assertEqual(report['artifacts']['summary.json'], d4.file_sha256(run.folder/'summary.json'))
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
        self.summary(lambda s: s['comparisons']['seed31001:c5'].__setitem__('observations', 4.000000001))
        self.failed('integer')
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

    def test_nonfinite_or_duplicate_fields_cannot_hide_in_a_journal(self):
        self.rewrite('replay.jsonl', lambda lines: lines[0][:-1] + b',"unused":NaN}\n' + b'\n'.join(lines[1:])+b'\n')
        self.failed('non-finite')
        self.rewrite('replay.jsonl', lambda lines: lines[0][:-1] + b',"game_index":0}\n' + b'\n'.join(lines[1:])+b'\n')
        self.failed('duplicate JSON key')

    def test_nonobject_summary_is_a_failure_report(self):
        (self.run_.folder/'summary.json').write_text('[]')
        self.failed('not an object')

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


class RecordsTest(unittest.TestCase):
    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux process inventory required')
    def test_documented_interrupt_selects_the_unbuffered_runner_in_this_checkout_only(self):
        import re
        plan = (Path(__file__).resolve().parents[1]/'docs/h7-rl-plan.md').read_text()
        section = plan.split('### 9.22', 1)[1].split('**B-R.', 1)[1].split('**C.', 1)[0]
        code = re.search(r"python3 - <<'PY'\n(.*?)\nPY", section, re.S).group(1)
        with TemporaryDirectory() as tmp:
            processes = []
            try:
                for name in ('selected', 'other-checkout'):
                    root = Path(tmp)/name
                    (root/'scripts').mkdir(parents=True)
                    script = root/'scripts/h7r_d4.py'
                    script.write_text('import time\ntime.sleep(60)\n')
                    processes.append(subprocess.Popen([sys.executable, '-u', str(script), 'run', '--mode', 'full'],
                                                       cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                result = subprocess.run([sys.executable, '-c', code], cwd=Path(tmp)/'selected',
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(str(processes[0].pid), result.stdout)
                processes[0].wait(timeout=5)
                self.assertIsNone(processes[1].poll())
            finally:
                for process in processes:
                    if process.poll() is None: process.terminate()
                    process.wait(timeout=5)

    @unittest.skipUnless(shutil.which('bash'), 'Bash required for the desktop procedure')
    def test_documented_D_stops_before_copy_when_fresh_check_fails(self):
        import re
        plan = (Path(__file__).resolve().parents[1]/'docs/h7-rl-plan.md').read_text()
        section = plan.split('### 9.22', 1)[1].split('**D. 결과 사본만 커밋**', 1)[1]
        block = re.search(r'```bash\n(.*?)```', section, re.S).group(1)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'.venv/bin').mkdir(parents=True)
            (root/'.venv/bin/activate').write_text('export PATH="$PWD/.venv/bin:$PATH"\n')
            stub = root/'.venv/bin/python'
            stub.write_text('#!/bin/sh\necho "fixture: fresh verification failed"\nexit 19\n')
            stub.chmod(0o755)
            raw = root/full.FULL_DIR
            raw.mkdir(parents=True)
            for name in ('identity.json', 'summary.json'): (raw/name).write_text('{}')
            log = root/'runs/h7r/R2-D4-BASELINE-1/logs/full-fixture.log'
            log.parent.mkdir(parents=True); log.write_text('fixture')
            # A previous PASS must never bypass fresh verification.
            dest = root/'docs/h7r/r2/R2-D4-BASELINE-1/records/full'
            dest.mkdir(parents=True)
            (dest/'check.json').write_text('{"integrity":"PASS"}')
            script = block.replace('cd ~/renju-h6', 'cd "$D4_SCRATCH"', 1)
            result = subprocess.run(['bash'], input=script, text=True, capture_output=True,
                                    env={**os.environ, 'D4_SCRATCH':str(root)})
            self.assertEqual(result.returncode, 19, result.stderr)
            self.assertFalse((dest/'summary.json').exists())
            self.assertFalse((dest/'records_manifest.json').exists())
            self.assertEqual((raw/'summary.json').read_text(), '{}')

    def test_records_manifest_refuses_raw_data_before_staging(self):
        from scripts.h7r_d4_records import records_manifest
        for name in ('replay.jsonl', 'branches.jsonl.gz', 'best.pt', 'cache.npy'):
            with self.subTest(name), TemporaryDirectory() as tmp:
                folder = Path(tmp)
                (folder/name).write_bytes(b'artifact')
                with self.assertRaisesRegex(ValueError, 'raw journal / model / tensor'):
                    records_manifest(folder, folder/'records_manifest.json')
                self.assertFalse((folder/'records_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
