"""R2-D3 R0: the reproduction check of the c8 recipe with the current trainer. The R0 contract's pins checked against
the real committed R2-0B / c8 records, the verdict (A1 verdict AND direct matches with the recorded fingerprint and
step), and the train / check wiring to the R2-0B machinery with new outputs only (heavy steps replaced)."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from hybrid import h7r_r2 as r2
from scripts import h7r_d3_r0 as r0

ROOT = Path(__file__).resolve().parents[1]
FP = 'e142e971a29da275a6edc908baca816a66d5472061ecf46806e9c17f649f7224'
A1 = 'docs/h7r/r2/R2-DIAG-1-A1.json'
R2C = 'docs/h7r/r2/R2-DIAG-1.contract.json'
REPRO = 'docs/h7r/r2/R2-DIAG-1/r2_0b/reproduction.json'
RUN = 'docs/h7r/r2/R2-DIAG-1/r2_0b/runs/seed-20261025/r2_train.json'
C8 = 'docs/h7c/c8/value_record.json'
QUIET = lambda *a, **k: None  # noqa: E731
HEAD = {'commit': 'c' * 40, 'dirty': False}
CONTEXT = ('r2 contract', 'r2 sha', 'amendment', 'amendment sha')


def ref(path, root=ROOT):
    return {'path': path, 'sha256': r2.file_sha256(root / path)}


def contract(root=ROOT, **change):
    c = {'format': r0.FORMAT, 'contract_id': r0.CONTRACT_ID, 'status': 'fixed', 'approved': True,
         'r2_contract': ref(R2C, root), 'amendment': ref(A1, root),
         'trainer_sources': {s: r2.file_sha256(root / s) for s in r0.TRAINER_SOURCES},
         'recorded': {'reproduction': ref(REPRO, root), 'run': ref(RUN, root), 'reference_record': ref(C8, root),
                      'assignment_sha256': FP, 'best_step': 147},
         'seed': 20261025, 'tolerance': {'max_abs_prediction': 0.001, 'abs_primary': 0.0002},
         'output': {'run_dir': 'runs/h7r/R2-D3-NO-OUTCOME-1/r0/seed-20261025',
                    'reproduction': 'runs/h7r/R2-D3-NO-OUTCOME-1/r0/reproduce.json',
                    'result': 'docs/h7r/r2/R2-D3-NO-OUTCOME-1/r0/r0.json'}}
    for key, value in change.items():
        c[key] = value
    return c


class World:
    """A temp root holding copies of the pinned records and trainer sources, and an R0 contract over them."""

    def __init__(self, tmp, **change):
        self.root = Path(tmp)
        for path in (A1, R2C, REPRO, RUN, C8, *r0.TRAINER_SOURCES):
            (self.root / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, self.root / path)
        self.contract = contract(self.root, **change)
        self.path = self.root / 'docs/h7r/r2/R2-D3-NO-OUTCOME-1/r0-contract.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.contract))

    def load(self):
        with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
            return r0.load_r0(self.path, self.root)


class ContractTest(unittest.TestCase):
    def test_the_real_records_hold_the_pinned_fingerprint_and_step(self):
        with TemporaryDirectory() as tmp:
            c, sha, amendment = World(tmp).load()
            self.assertEqual((c['recorded']['assignment_sha256'], c['recorded']['best_step']), (FP, 147))
            self.assertEqual(c['tolerance'], amendment['r2_0b']['reproduction']['tolerance'])

    def test_wrong_pins_are_refused(self):
        bad_fp = FP[:-1] + ('0' if FP[-1] != '0' else '1')
        cases = {
            'recorded fingerprint': lambda c: c['recorded'].update(assignment_sha256=bad_fp),
            'recorded step': lambda c: c['recorded'].update(best_step=146),
            'A1 tolerances': lambda c: c['tolerance'].update(abs_primary=0.001),
            'reproduction seed': lambda c: c.update(seed=31001),
            'checkout differs': lambda c: c['trainer_sources'].update({'src/hybrid/h7c_train.py': '0' * 64}),
            'fixed, approved': lambda c: c.update(approved=False),
            'keys differ': lambda c: c.update(extra=1),
            'under runs/': lambda c: c['output'].update(run_dir='docs/run'),
            'reference record': lambda c: c['recorded'].update(reference_record=ref(RUN)),
        }
        for message, change in cases.items():
            with self.subTest(message), TemporaryDirectory() as tmp:
                w = World(tmp)
                c = deepcopy(w.contract)
                change(c)
                w.path.write_text(json.dumps(c))
                with self.assertRaisesRegex(ValueError, message):
                    w.load()

    def test_a_changed_record_is_refused(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            data = json.loads((w.root / REPRO).read_text())
            data['checks']['best_step'] = [146, 146]
            (w.root / REPRO).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'not the R2-0B reproduction record R0 pins'):
                w.load()

    def test_uncommitted_contract_is_refused(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            with mock.patch('hybrid.h7c_nmask.committed', return_value=False), \
                    self.assertRaisesRegex(ValueError, 'not committed'):
                r0.load_r0(w.path, w.root)


class VerdictTest(unittest.TestCase):
    recorded = {'assignment_sha256': FP, 'best_step': 147}
    a1 = {'version': 'h7r-r2-reproduction-v1', 'verdict': 'METRIC_ONLY', 'reasons': []}

    def test_pass_needs_the_a1_verdict_and_every_direct_match(self):
        self.assertEqual(r0.r0_verdict(self.a1, self.recorded, FP, FP, 147)['verdict'], 'PASS')
        bitwise = {**self.a1, 'verdict': 'BITWISE_REPRODUCED'}
        self.assertEqual(r0.r0_verdict(bitwise, self.recorded, FP, FP, 147)['verdict'], 'PASS')

    def test_a_consistent_change_of_the_assignment_code_does_not_pass(self):
        # both sides computed by the current code agree (so A1 passes), but they are not the recorded fingerprint
        other = 'a' * 64
        v = r0.r0_verdict(self.a1, self.recorded, other, other, 147)
        self.assertEqual(v['verdict'], 'NOT_REPRODUCED')
        self.assertEqual(len(v['reasons']), 2)

    def test_each_failure_alone_fails(self):
        cases = [({'verdict': 'NOT_REPRODUCED', 'reasons': ['x']}, FP, FP, 147),
                 (self.a1, 'b' * 64, FP, 147), (self.a1, FP, 'b' * 64, 147), (self.a1, FP, FP, 146),
                 (self.a1, FP, FP, 147.0), (self.a1, FP, FP, None), (None, FP, FP, 147)]
        for a1, rerun, ref_fp, step in cases:
            with self.subTest(rerun=rerun[:4], ref=ref_fp[:4], step=step):
                v = r0.r0_verdict(a1, self.recorded, rerun, ref_fp, step)
                self.assertEqual((v['verdict'], len(v['reasons'])), ('NOT_REPRODUCED', 1))


class WiringTest(unittest.TestCase):
    def args(self, w, **kw):
        return SimpleNamespace(r0_contract=w.path, root=w.root, device='cpu', **kw)

    def test_train_calls_the_r2_0b_trainer_once_into_a_new_run_dir(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            seen = {}
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True), \
                    mock.patch('h7r_r2_seed.train', side_effect=lambda a, log=None: seen.update(vars(a)) or
                               {'verdict': 'CANDIDATE', 'best_step': 147}):
                r0.train(self.args(w, cycle_cache='c', cycle_rows='r', replay_cache='p', parent='m'), log=QUIET)
            self.assertEqual(seen['seed'], 20261025)
            self.assertEqual(seen['out_dir'], w.root / w.contract['output']['run_dir'])
            self.assertIsNone(seen['reproduction'])
            self.assertEqual((seen['contract'], seen['amendment']), (w.root / R2C, w.root / A1))
            self.assertFalse(seen['allow_dirty'])
            (w.root / w.contract['output']['run_dir']).mkdir(parents=True)
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True), \
                    self.assertRaisesRegex(ValueError, 'never overwritten'):
                r0.train(self.args(w, cycle_cache='c', cycle_rows='r', replay_cache='p', parent='m'), log=QUIET)

    def fake_reproduce(self, w, rerun_fp=FP, ref_fp=FP, step=147, verdict='METRIC_ONLY', git=None):
        def reproduce(a, log=None, validation_context=None):
            # the no-publish mode: R0 writes the comparison itself, after checking the commit again
            assert validation_context == CONTEXT, validation_context
            body = {'git': dict(git or HEAD), 'tolerance': w.contract['tolerance'],
                    'verdict': {'version': 'h7r-r2-reproduction-v1', 'verdict': verdict, 'reasons': []},
                    'assignment': {'rerun_sha256': rerun_fp, 'reference': {'sha256': ref_fp}},
                    'best_step': {'reference': 147, 'rerun': step},
                    'checks': {'assignment': [ref_fp, rerun_fp], 'best_step': [147, step]},
                    'environment': json.loads((w.root / RUN).read_text())['environment']}
            return body
        return reproduce

    def run_dir(self, w, trainer=None, git=None):
        d = w.root / w.contract['output']['run_dir']
        d.mkdir(parents=True)
        (d / 'r2_train.json').write_text(json.dumps({
            'sources': {'src/hybrid/h7c_train.py': trainer or w.contract['trainer_sources']['src/hybrid/h7c_train.py']},
            'git': HEAD if git is None else git}))

    def check(self, w, fake, head=None, heads=None):
        """``heads``: the git states R0 sees in order (before the A1 comparison, after it)."""
        states = mock.patch.object(r0, 'git_state', side_effect=list(heads)) if heads else \
            mock.patch.object(r0, 'git_state', return_value=head or HEAD)
        with mock.patch('hybrid.h7c_nmask.committed', return_value=True), \
                mock.patch('h7r_r2_seed.approved', return_value=CONTEXT), \
                mock.patch('h7r_r2_seed.reproduce', side_effect=fake), states:
            return r0.check(self.args(w, cache='h', reference_model='m,j'), log=QUIET)

    def test_check_passes_and_records_the_direct_comparison(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            self.run_dir(w)
            body = self.check(w, self.fake_reproduce(w))
            self.assertEqual(body['verdict']['verdict'], 'PASS')
            self.assertEqual(body['direct']['recorded_assignment_sha256'], FP)
            self.assertTrue(body['environment']['same'])
            self.assertEqual(body['train_git'], body['git'])
            saved = json.loads((w.root / w.contract['output']['result']).read_text())
            self.assertEqual(saved['verdict'], body['verdict'])
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                self.check(w, self.fake_reproduce(w))

    def test_a_drifted_assignment_code_is_recorded_as_not_reproduced(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            self.run_dir(w)
            body = self.check(w, self.fake_reproduce(w, rerun_fp='c' * 64, ref_fp='c' * 64))
            self.assertEqual((body['verdict']['verdict'], body['verdict']['a1_verdict']['verdict']),
                             ('NOT_REPRODUCED', 'METRIC_ONLY'))
            self.assertTrue((w.root / w.contract['output']['result']).is_file())  # the failure is kept

    def test_a_run_of_another_trainer_is_refused(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            self.run_dir(w, trainer='0' * 64)
            with self.assertRaisesRegex(ValueError, 'not trained with the trainer'):
                self.check(w, self.fake_reproduce(w))


    def test_train_and_check_must_share_one_clean_commit(self):
        cases = {'another commit': ({'commit': 'd' * 40, 'dirty': False}, None),
                 'dirty train': ({'commit': 'c' * 40, 'dirty': True}, None),
                 'dirty check': (None, {'commit': 'c' * 40, 'dirty': True}),
                 'no git record': ('none', None)}
        for name, (train_git, head) in cases.items():
            with self.subTest(name), TemporaryDirectory() as tmp:
                w = World(tmp)
                self.run_dir(w, git={} if train_git == 'none' else train_git)
                with self.assertRaisesRegex(ValueError, 'train and check run at one commit'):
                    self.check(w, self.fake_reproduce(w), head=head)
                self.assertFalse((w.root / w.contract['output']['reproduction']).exists())  # refused before A1
                self.assertFalse((w.root / w.contract['output']['result']).exists())

    def test_a_commit_change_during_the_check_is_refused(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            self.run_dir(w)
            with self.assertRaisesRegex(ValueError, 'commit changed during the check'):
                self.check(w, self.fake_reproduce(w, git={'commit': 'e' * 40, 'dirty': False}))
            self.assertFalse((w.root / w.contract['output']['result']).exists())


    def test_a_head_change_after_the_a1_comparison_started_leaves_no_record(self):
        # A1 saw the right commit when it started; the HEAD moved (or the tree got dirty) before it returned
        for after in ({'commit': 'f' * 40, 'dirty': False}, {'commit': 'c' * 40, 'dirty': True}):
            with self.subTest(after=after), TemporaryDirectory() as tmp:
                w = World(tmp)
                self.run_dir(w)
                with self.assertRaisesRegex(ValueError, 'commit changed during the check'):
                    self.check(w, self.fake_reproduce(w), heads=[HEAD, after])
                self.assertFalse((w.root / w.contract['output']['reproduction']).exists())
                self.assertFalse((w.root / w.contract['output']['result']).exists())

    def test_the_comparison_is_written_by_r0_only_after_the_second_check(self):
        with TemporaryDirectory() as tmp:
            w = World(tmp)
            self.run_dir(w)
            body = self.check(w, self.fake_reproduce(w), heads=[HEAD, HEAD])
            written = json.loads((w.root / w.contract['output']['reproduction']).read_text())
            self.assertEqual(written['assignment'], {'rerun_sha256': FP, 'reference': {'sha256': FP}})
            self.assertEqual(body['reproduction']['sha256'],
                             r2.file_sha256(w.root / w.contract['output']['reproduction']))


class ProcedureTest(unittest.TestCase):
    def test_the_draft_blocks_parse_and_fail_closed(self):
        import re
        import subprocess
        doc = (ROOT / 'docs/h7r/r2/R2-D3-NO-OUTCOME-1-draft.md').read_text()
        section = doc.split('### 2.5', 1)[1].split('\n## ', 1)[0]
        blocks = re.findall(r'```bash\n(.*?)```', section, re.S)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(subprocess.run(['bash', '-n'], input=block, text=True).returncode, 0)
            self.assertIn('set -euo pipefail', block)
        self.assertIn('verify-staged', blocks[1])
        self.assertIn('git rev-parse HEAD', blocks[1])


if __name__ == '__main__':
    unittest.main()
