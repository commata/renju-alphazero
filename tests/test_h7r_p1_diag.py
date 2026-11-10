"""scripts/h7r_p1_diag.py: a run records its settings, inputs, initial state, fingerprints, curve, selection, state
and fixed-input logits without changing the training (the passive wrappers); on CPU the old and the new trainer and
the diagnostic option agree exactly; compare finds the first differing step; nothing is overwritten."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from hybrid import h7c_train as tr
import h7r_p1 as p1
import h7r_p1_diag as dg
from tests.test_h7c_train import guard_data, policy_rows, small_model
from tests.test_h7r_p1 import QUIET, draft_contract, git, guard_eval

ROOT = Path(__file__).resolve().parents[1]


class DiagTest(unittest.TestCase):
    def setUp(self):
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        h3 = root / p1.PATHS['h3']
        h3.parent.mkdir(parents=True)
        save_checkpoint(h3, small_model())
        cfg = ModelConfig()
        (root / p1.h3_meta_path()).write_text(json.dumps({'policy_trained': True, 'value_trained': False,
                                                           'config': {'channels': cfg.channels, 'blocks': cfg.blocks}}))
        rows = {'train': policy_rows(6), 'val': policy_rows(2, 'val')}
        self.checked = {'rows': rows, 'h3': {'policy_checkpoint_sha256': 'a' * 64},
                        'cache': {'train': guard_data(policy_rows(4)), 'val': guard_data(policy_rows(2, 'val'))}}
        c = draft_contract()
        c['training'] = {**c['training'], 'batch_size': 16, 'epochs': 1.0, 'evaluations': 3, 'guard_drop': 1.0}
        self.c = c

    def run_it(self, name, trainer, diag=False):
        return dg.run_one(self.c, self.checked, trainer, diag, False, 'cpu', self.root / 'runs' / name, self.root,
                          log=QUIET)

    def test_runs_agree_on_cpu_and_the_wrappers_change_nothing(self):
        old = self.run_it('old-1', 'old')
        plain = tr.train_policy(p1.policy_config(self.c, dg.SEED, 'cpu'), small_model(), small_model(),
                                tr.policy_tensors(self.checked['rows']['train']),
                                tr.policy_tensors(self.checked['rows']['val']), self.checked['cache']['val'],
                                guard_eval, log=QUIET)['record']
        self.assertEqual(old['evaluations'], plain['evaluations'])  # the fingerprints are passive
        self.assertEqual(old['selected_step'], plain['best']['step'])
        fp = old['fingerprints']
        self.assertEqual(fp['steps'], plain['total_steps'])
        self.assertTrue(all(x is not None for x in fp['loss']))
        self.run_it('old-2', 'old')
        self.run_it('new-1', 'new')
        self.run_it('new-2', 'new')
        self.run_it('new-diag-1', 'new', diag=True)
        c2 = {'evaluations': plain['evaluations'], 'best': plain['best'], 'teacher_guard': plain['teacher_guard'],
              'parent_val': plain['parent_val']}
        (self.root / 'c2.json').write_text(json.dumps(c2))
        out = dg.compare(self.root / 'runs', 'c2.json', self.root)
        kinds = {p['kind']: p for p in out['pairs']}
        self.assertEqual(set(kinds), {'repeat new', 'repeat old', 'old vs new', 'new vs new-diag'})
        for p in out['pairs']:
            self.assertIsNone(p['loss_first_differing_step'], p['kind'])
            self.assertTrue(p['selected_state_equal'] and p['fixed_logits_equal'], p['kind'])
            self.assertTrue(all(p['fingerprints_equal'].values()), p['kind'])
            self.assertEqual(p['fixed_argmax_agreement'], 1.0)
        self.assertTrue(all(r['selected_step_equal'] and r['eligibility_equal'] for r in out['against_c2']))
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.run_it('old-1', 'old')

    def test_compare_finds_where_runs_part(self):
        a = self.run_it('old-1', 'old')
        b_dir = self.root / 'runs' / 'old-2'
        b_dir.mkdir(parents=True)
        b = copy.deepcopy(a)
        b['fingerprints']['loss'][1] += 1e-4
        b['evaluations'][1]['val']['objective'] += 2e-5
        b['selected_state_sha256'] = '0' * 64
        (b_dir / 'record.json').write_text(json.dumps(b))
        for f in ('state.pt', 'logits.pt'):
            t = torch.load(self.root / 'runs' / 'old-1' / f, weights_only=True)
            if isinstance(t, dict):
                k = sorted(t)[0]
                t = {**t, k: t[k] + 1e-3}
            else:
                t = t + 0.0
            torch.save(t, b_dir / f)
        (self.root / 'c2.json').write_text(json.dumps({'evaluations': a['evaluations'], 'best': {'step': a['selected_step']},
                                                       'teacher_guard': {'top1': a['teacher_guard_top1']},
                                                       'parent_val': a['parent_val']}))
        out = dg.compare(self.root / 'runs', 'c2.json', self.root)
        p = out['pairs'][0]
        self.assertEqual(p['kind'], 'repeat old')
        self.assertEqual(p['loss_first_differing_step'], 2)
        self.assertEqual(p['evaluations']['first_differing_evaluation'], a['evaluations'][1]['step'])
        self.assertAlmostEqual(p['evaluations']['objective'], 2e-5, places=9)
        self.assertFalse(p['selected_state_equal'])
        self.assertAlmostEqual(p['selected_state_max_abs_diff'], 1e-3, places=5)
        self.assertTrue(p['fixed_logits_equal'])  # the copied logits

    def test_settings_are_recorded(self):
        s = dg.settings('cpu')
        for k in ('torch', 'cuda', 'cudnn', 'device_name', 'cudnn_deterministic', 'cudnn_allow_tf32',
                  'matmul_allow_tf32', 'float32_matmul_precision', 'deterministic_algorithms', 'threads',
                  'environment'):
            self.assertIn(k, s)
        self.assertIn('CUBLAS_WORKSPACE_CONFIG', s['environment'])

    def test_deterministic_cuda_needs_the_cublas_setting(self):
        from unittest import mock
        with mock.patch.dict('os.environ', {'CUBLAS_WORKSPACE_CONFIG': ''}):
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                dg.run_one(self.c, self.checked, 'old', False, True, 'cuda', self.root / 'runs' / 'x', self.root,
                           log=QUIET)
        self.assertFalse((self.root / 'runs' / 'x').exists())


STUB_DIAG = """import json, os, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == 'run':
    out = Path(a[a.index('--out-dir') + 1])
    print('run', out.name, 'CUBLAS=' + str(os.environ.get('CUBLAS_WORKSPACE_CONFIG')), ' '.join(a), flush=True)
    if out.name in os.environ.get('DIAG_FAIL', '').split(','):
        print('STOP: RuntimeError: no deterministic implementation', flush=True)
        sys.exit(1)
    out.mkdir(parents=True)
    (out / 'state.pt').write_text('state')
    (out / 'logits.pt').write_text('logits')
    (out / 'record.json').write_text(json.dumps({'run': out.name, 'argv': a}))
else:
    if os.environ.get('DIAG_COMPARE_FAIL'):
        print('STOP: ValueError: compare', flush=True)
        sys.exit(1)
    out = Path(a[a.index('--output') + 1])
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent.name for p in Path(a[a.index('--run-dir') + 1]).glob('*/record.json'))
    out.write_text(json.dumps({'runs': runs}))
"""


class BlockTest(unittest.TestCase):
    """The desktop block of the draft (section 4) run by bash on a real temporary repository with a bare origin, the
    repository's real .gitignore and the real record tool; the diagnosis tool is a stand-in."""

    def setUp(self):
        import re
        import shutil
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.origin, self.work = base / 'origin.git', base / 'work'
        git(base, 'init', '-q', '--bare', str(self.origin))
        git(base, 'init', '-q', '-b', 'feat/h7-c1', str(self.work))
        w = self.work
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', w / '.gitignore')
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")\n')
        (w / 'scripts/h7r_p1_diag.py').write_text(STUB_DIAG)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-P1-DIAG-1-draft.md').read_text()
        blocks = re.findall(r'```bash\n(.*?)```', doc.split('\n## 4.', 1)[1], re.S)
        self.assertEqual(len(blocks), 1)
        self.block = blocks[0].replace('~/renju-h6', str(w))
        self.D = 'docs/h7r/r2/R2-P1-DIAG-1'

    def bash(self, **env):
        import os
        import subprocess
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        e = {**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.block], cwd=self.work, capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def test_a_failed_run_is_logged_and_the_rest_go_on(self):
        rc, out = self.bash(DIAG_FAIL='old-det-1')
        self.assertEqual(rc, 0, out)
        recs = self.files(f'{self.D}/run-records')
        self.assertEqual(len(recs), 9)
        self.assertNotIn(f'{self.D}/run-records/old-det-1/record.json', recs)
        logs = [f for f in self.files(f'{self.D}/stage-logs') if f.endswith('.log')]
        self.assertEqual(len(logs), 11)  # ten runs and the comparison
        failed = (self.work / next(f for f in logs if '/run-old-det-1-' in f)).read_text()
        self.assertIn('__EXIT__ 1', failed)
        for f in logs:
            text = (self.work / f).read_text()
            self.assertEqual('CUBLAS=:4096:8' in text, '-det-' in f, f)
        self.assertEqual(json.loads((self.work / self.D / 'diagnosis.json').read_text())['runs'],
                         sorted(r.split('/')[-2] for r in recs))
        self.assertFalse(self.files('runs'))  # states and logits stay under runs/
        self.assertIn('compare exit 0', git(self.work, 'log', '-1', '--format=%s'))
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        rc, out = self.bash()
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)

    def test_a_failed_comparison_is_recorded_and_a_rerun_completes_it(self):
        import time
        rc, out = self.bash(DIAG_COMPARE_FAIL='1')
        self.assertEqual(rc, 1, out)
        self.assertFalse(self.files(f'{self.D}/diagnosis.json'))
        self.assertEqual(len(self.files(f'{self.D}/run-records')), 10)
        self.assertIn('compare exit 1', git(self.work, 'log', '-1', '--format=%s'))
        time.sleep(1.1)  # a new log name
        rc, out = self.bash()
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count('done: '), 10)  # finished runs are not run again
        self.assertTrue(self.files(f'{self.D}/diagnosis.json'))
        logs = [f for f in self.files(f'{self.D}/stage-logs') if f.endswith('.log')]
        self.assertEqual(len(logs), 12)  # the failed comparison's log is kept
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))

    def test_a_run_dir_without_a_record_stops_the_block(self):
        d = self.work / 'runs/h7r/R2-P1-DIAG-1/old-1'
        d.mkdir(parents=True)
        (d / 'state.pt').write_text('partial')
        rc, out = self.bash()
        self.assertEqual(rc, 1)
        self.assertIn('holds files without a record', out)
        self.assertEqual(len(git(self.work, 'log', '--format=%h').splitlines()), 1)  # nothing committed


if __name__ == '__main__':
    unittest.main()
