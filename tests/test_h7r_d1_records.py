"""Interrupted record copies and the D1 desktop recovery block, without training or GPU access."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts import h7r_d4_records as records

ROOT = Path(__file__).resolve().parents[1]
D1 = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1'
STAMP = '20261107-130000'


class RecoveryTest(unittest.TestCase):
    def test_partial_copy_resumes_without_rewriting_and_detects_changed_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dest = root / 'runs/logs', root / 'records/run-logs'
            src.mkdir(parents=True)
            for name in ('a.log', 'b.log'):
                (src / name).write_text(name)
            manifest = root / 'records/manifest.json'
            real_copy = shutil.copy2

            def interrupted(source, target):
                if Path(source).name == 'b.log':
                    raise OSError('interrupted copy')
                return real_copy(source, target)

            with mock.patch.object(records.shutil, 'copy2', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    records.copy_logs(src, '*.log', dest, manifest, root=root)
            self.assertFalse(manifest.exists())
            before = (dest / 'a.log').stat().st_mtime_ns
            body = records.copy_logs(src, '*.log', dest, manifest, root=root, resume=True)
            self.assertEqual(len(body['logs']), 2)
            self.assertEqual((dest / 'a.log').stat().st_mtime_ns, before)
            saved = manifest.read_bytes()
            self.assertEqual(records.copy_logs(src, '*.log', dest, manifest, root=root, resume=True), body)
            self.assertEqual(manifest.read_bytes(), saved)
            (src / 'a.log').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'existing copy SHA-256 differs'):
                records.copy_logs(src, '*.log', dest, manifest, root=root, resume=True)
            self.assertEqual((dest / 'a.log').read_text(), 'a.log')
            self.assertEqual(manifest.read_bytes(), saved)

    def test_existing_manifests_are_revalidated_and_models_are_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dest = root / 'src', root / 'dest'
            src.mkdir()
            (src / 'a.log').write_text('log')
            log_manifest = dest / 'logs.json'
            records.copy_logs(src, '*.log', dest, log_manifest, root=root)
            changed = json.loads(log_manifest.read_text())
            changed['logs'][0]['sha256'] = '0' * 64
            log_manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'existing manifest differs'):
                records.copy_logs(src, '*.log', dest, log_manifest, root=root, resume=True)
            folder = root / 'records'
            folder.mkdir()
            record = folder / 'record.json'
            record.write_text('{}')
            manifest = folder / 'manifest.json'
            records.records_manifest(folder, manifest)
            before = manifest.read_bytes()
            records.records_manifest(folder, manifest, resume=True)
            record.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'existing manifest differs'):
                records.records_manifest(folder, manifest, resume=True)
            self.assertEqual(manifest.read_bytes(), before)
            (folder / 'unwanted.pt').write_bytes(b'model')
            with self.assertRaisesRegex(ValueError, 'model / tensor'):
                records.records_manifest(folder, manifest, resume=True)


class DesktopRecoveryTest(unittest.TestCase):
    """Run the documented shell block against real git and the real records tool."""

    def test_training_block_success_failure_and_head_change(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash or not Path(bash).is_file():
            self.skipTest('bash unavailable')
        doc = (ROOT / 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1-draft.md').read_text(encoding='utf-8')
        section = doc.split('\n2. 학습', 1)[1].split('\n2-F.', 1)[0]
        block = re.search(r'```bash\n(.*?)```', section, re.S)[1]
        fake_trainer = '''import os, subprocess, sys
from pathlib import Path
command = sys.argv[1]
with Path('runs/calls.txt').open('a') as f:
    f.write(' '.join(sys.argv[1:]) + '\\n')
if command == 'fit':
    output = Path('docs/h7r/r2/R2-D1-HEAD-WIDTH-1/1k-v2/fit.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{}')
else:
    arm = sys.argv[sys.argv.index('--arm') + 1]
    seed = sys.argv[sys.argv.index('--seed') + 1]
    run = Path('runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2') / (arm + '-' + seed)
    run.mkdir()
    (run / 'record.json').write_text('{}')
    (run / 'final.pt').write_bytes(b'stand-in model')
    if os.environ['CASE'] == 'failed' and arm == 'wide' and seed == '31002':
        sys.exit(7)
    (run / 'd1_train.json').write_text('{}')
    if os.environ['CASE'] == 'head_changed' and arm == 'wide' and seed == '31003':
        Path('state.txt').write_text('changed')
        subprocess.run(['git', 'commit', '-qam', 'concurrent commit'], check=True)
'''
        for case, completed, code in (('success', 6, 0), ('failed', 3, 7), ('head_changed', 6, 97)):
            with self.subTest(case=case), TemporaryDirectory() as tmp:
                root, remote = Path(tmp) / 'repo', Path(tmp) / 'remote.git'
                root.mkdir()
                env = {**os.environ, 'CASE': case, 'GIT_AUTHOR_NAME': 'test',
                       'GIT_AUTHOR_EMAIL': 'test@example.com', 'GIT_COMMITTER_NAME': 'test',
                       'GIT_COMMITTER_EMAIL': 'test@example.com'}

                def git(*args):
                    return subprocess.check_output(['git', *args], cwd=root, env=env,
                                                   stderr=subprocess.STDOUT).decode()

                git('init', '-q', '-b', 'feat/h7-c1')
                git('config', 'core.autocrlf', 'false')
                subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True, env=env)
                git('remote', 'add', 'origin', str(remote))
                for path in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', '.gitignore'):
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / path, target)
                (root / 'scripts/h7r_d1.py').write_text(fake_trainer)
                (root / 'scripts/check_track_b_frozen.py').write_text("print('stand-in checker')\n")
                (root / 'state.txt').write_text('initial')
                (root / '.venv/bin').mkdir(parents=True)
                (root / '.venv/bin/activate').write_text('')
                git('add', '.')
                git('commit', '-qm', 'fixture')
                git('push', '-q', '-u', 'origin', 'feat/h7-c1')
                script = block.replace('cd ~/renju-h6', f'cd {shlex.quote(root.as_posix())}')
                script = script.replace('python ', shlex.quote(Path(sys.executable).as_posix()) + ' ')
                out = subprocess.run([bash], input=script, text=True, capture_output=True, env=env)
                self.assertEqual(out.returncode, code, out.stdout + out.stderr)
                self.assertIn(f'__RECORDED__ {code}', out.stdout)
                self.assertIn(f'runs completed {completed}/6', out.stdout)
                calls = (root / 'runs/calls.txt').read_text().splitlines()
                self.assertEqual(any(c == 'fit --d1-contract ' + D1 + '/d1-1k-v2-contract.json'
                                     for c in calls), case == 'success')
                self.assertEqual(len(calls), 7 if case == 'success' else 4 if case == 'failed' else 6)
                files = git('show', '--name-only', '--format=', 'HEAD').splitlines()
                self.assertTrue(files)
                self.assertTrue(all(p.startswith(D1 + '/1k-v2/') for p in files))
                self.assertFalse(any(p.endswith('.pt') for p in files))

    def test_recovery_after_partial_copy_and_after_commit_before_push(self):
        bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
        if not bash or not Path(bash).is_file():
            self.skipTest('bash unavailable')
        doc = (ROOT / 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1-draft.md').read_text(encoding='utf-8')
        section = doc.split('\n2-F.', 1)[1].split('\n3.', 1)[0]
        block = re.search(r'```bash\n(.*?)```', section, re.S)[1]
        with TemporaryDirectory() as tmp:
            root, remote = Path(tmp) / 'repo', Path(tmp) / 'remote.git'
            root.mkdir()
            env = {**os.environ, 'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.com',
                   'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@example.com'}

            def git(*args, cwd=root):
                return subprocess.check_output(['git', *args], cwd=cwd, env=env, stderr=subprocess.STDOUT).decode()

            git('init', '-q', '-b', 'feat/h7-c1')
            git('config', 'core.autocrlf', 'false')
            subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True, env=env)
            git('remote', 'add', 'origin', str(remote))
            for path in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', '.gitignore'):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, target)
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/activate').write_text('')
            git('add', '.')
            git('commit', '-qm', 'fixture')
            git('push', '-q', '-u', 'origin', 'feat/h7-c1')
            run = root / 'runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2'
            log = run / 'logs' / f'train-narrow-31001-{STAMP}.log'
            log.parent.mkdir(parents=True)
            log.write_text('1 __START__\n2 __EXIT__ 1\n')
            folder = root / D1 / '1k-v2'
            copies = folder / 'train-logs'
            copies.mkdir(parents=True)
            shutil.copy2(log, copies / log.name)  # interrupted copy, no manifest yet
            script = block.replace('cd ~/renju-h6', f'cd {shlex.quote(root.as_posix())}')
            script = script.replace('python ', shlex.quote(Path(sys.executable).as_posix()) + ' ')

            def recover():
                result = subprocess.run([bash], input=script, text=True, capture_output=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            # Reject the first push after the recovery commit has already succeeded.
            hook = remote / 'hooks/pre-receive'
            hook.write_text('#!/bin/sh\nexit 1\n', newline='\n')
            hook.chmod(0o755)
            first = subprocess.run([bash], input=script, text=True, capture_output=True, env=env)
            self.assertNotEqual(first.returncode, 0)
            committed = git('rev-parse', 'HEAD').strip()
            self.assertNotEqual(committed, git('rev-parse', 'refs/heads/feat/h7-c1', cwd=remote).strip())
            hook.unlink()
            recovered = recover()
            self.assertIn('nothing left to record; push checked', recovered.stdout)
            self.assertEqual(committed, git('rev-parse', 'refs/heads/feat/h7-c1', cwd=remote).strip())
            recover()  # idempotent, with the same files and no extra commit
            self.assertEqual(committed, git('rev-parse', 'HEAD').strip())
            files = git('show', '--name-only', '--format=', 'HEAD').splitlines()
            self.assertEqual(sorted(files), sorted([
                f'{D1}/1k-v2/train-logs/{log.name}',
                f'{D1}/1k-v2/train-logs/{STAMP}.manifest.json',
            ]))


if __name__ == '__main__':
    unittest.main()
