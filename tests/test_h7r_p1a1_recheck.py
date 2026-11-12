"""scripts/h7r_p1a1_recheck.py: the O0 class description by hand; the recheck on a real temporary repository holding
copies of the committed A1 contract, Arena contract, sources and evidence (stand-in arm records): arm files checked
against the judge first, bounds recomputed with the judge's functions, a differing judge reported (exit 2), nothing
overwritten; and the desktop block."""
import json
import os
from pathlib import Path
import random
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7r_p1 as p1
import h7r_p1a1 as a1
import h7r_p1a1_recheck as rc
from tests.test_h7r_p1 import git

ROOT = Path(__file__).resolve().parents[1]


class ClassAnalysisTest(unittest.TestCase):
    def test_by_hand(self):
        rows = [(0.0, 0.5, 'a'), (0.5, 0.5, 'a'), (1.0, 0.5, 'b'), (1.0, 0.0, 'b'), (0.5, 1.0, 'c'), (0.0, 0.0, 'c')]
        out = rc.class_analysis(rows, other_t_mean=0.6, other_d_mean=0.1)
        self.assertEqual((out['pairs'], out['classes'], out['pairs_per_class']), (6, 3, [2]))
        self.assertAlmostEqual(out['T_score'], 0.5)
        self.assertAlmostEqual(out['P_combined'], (0.5 + 0.6) / 2 - 0.5)
        by = {c['class_id']: c for c in out['per_class']}
        self.assertAlmostEqual(by['a']['T_minus_half_sum'], -0.5)
        self.assertAlmostEqual(by['b']['T_minus_half_mean'], 0.5)
        self.assertAlmostEqual(by['c']['T_minus_C_mean'], -0.25)
        # leaving a out: O0 T = (1 + 1 + 0.5 + 0) / 4
        self.assertAlmostEqual(by['a']['leave_out']['T_score'], 0.625)
        self.assertAlmostEqual(by['a']['leave_out']['P_change'], (0.625 - 0.5) / 2, places=5)
        d_all = sum(t - c for t, c, _ in rows) / 6
        d_wo = ((1.0 - 0.5) + (1.0 - 0.0) + (0.5 - 1.0) + 0.0) / 4
        self.assertAlmostEqual(by['a']['leave_out']['D_change'], (d_wo - d_all) / 2, places=5)  # stored to 6 places
        self.assertEqual(out['class_signs'], {'negative': 2, 'zero': 0, 'positive': 1})
        conc = out['deficit_concentration']
        self.assertEqual(conc['worst_classes'], ['a', 'c'])  # both -0.5: sorted stably by class id
        self.assertEqual(conc['share_of_negative_sum']['worst_1'], 0.5)
        self.assertEqual(conc['share_of_negative_sum']['worst_3'], 1.0)
        self.assertIn('post-hoc', out['note'])


class RecheckTest(unittest.TestCase):
    """A temporary repository with copies of the committed contracts, sources and evidence, a stand-in judge record
    made with the judge's own functions from synthetic pair tables, and stand-in arm files."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        c = json.loads((ROOT / a1.CONTRACT).read_text())
        rels = [a1.CONTRACT, a1.ARENA_CONTRACT, '.gitignore', *(s['path'] for s in c['sources']),
                *(e['path'] for e in c['evidence'].values())]
        for rel in rels:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        self.arena = json.loads((ROOT / a1.ARENA_CONTRACT).read_text())
        rng = random.Random(7)
        self.tables = {}
        o0 = [(i, i // 4) for i in range(300)]
        o1 = [(1000 + i, 1000 + i) for i in range(300)]
        for a in self.arena['arms']:
            ids = o0 if a['name'].startswith('O0') else o1 if a['name'].startswith('O1') else \
                [(2000 + i, 2000 + i // 2) for i in range(a['pairs'])]
            bias = 0.15 if '_T_' in a['name'] else 0.0
            pairs = [{'opening_id': i, 'class_id': k,
                      'pair_score': min(1.0, rng.choice([0.0, 0.5, 0.5, 1.0]) + (0.5 if rng.random() < bias else 0))}
                     for i, k in ids]
            self.tables[a['name']] = {'valid': True, 'pairs': pairs}
        self.summaries = {n: {'pairs': len(t['pairs']), 'score': sum(p['pair_score'] for p in t['pairs']) /
                              len(t['pairs'])} for n, t in self.tables.items()}
        self.arms_dir = root / a1.RUNS / 'arena'
        self.arms_dir.mkdir(parents=True)
        for n in self.tables:
            (self.arms_dir / f'{n}.json').write_text(json.dumps({'arm': n}))
        self.c, _, self.eid = a1.load_contract(root / a1.CONTRACT, ROOT, check_sources=False)
        st = self.c['reading']['statistics']
        b = st['bootstrap']
        strata = p1.paired_strata(self.tables)
        main = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'])
        sens = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'], unit='class')
        self.judge = {'format': 'h7r-p1a1-judge-v1', 'contract_id': a1.CONTRACT_ID, 'experiment_sha256': self.eid,
                      'arena_contract_sha256': p1.file_sha256(root / a1.ARENA_CONTRACT),
                      'arm_records_sha256': {n: p1.file_sha256(self.arms_dir / f'{n}.json') for n in self.tables},
                      'readings': p1.labels(main), 'bounds': main,
                      'sensitivity_class_level': {'bounds': sens, 'readings': p1.labels(sens)},
                      'per_arm': self.summaries}
        self.write_judge(self.judge)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'records')
        self.assertEqual(git(root, 'ls-files', 'runs'), '')

    def write_judge(self, body, commit=False):
        path = self.root / a1.D / 'judge.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=1))
        if commit:
            git(self.root, 'commit', '-qam', 'judge')

    def load(self, path, contract, sha, root):
        name = Path(path).stem
        self.assertEqual(sha, self.judge['arena_contract_sha256'])
        return {'table': self.tables[name], 'summary': self.summaries[name]}

    def run_main(self, out):
        import h7r_eval as cli
        with mock.patch.object(cli, 'load_arm', side_effect=self.load):
            return rc.main(['--arms-dir', str(self.arms_dir), '--output', str(out)], root=self.root)

    def test_the_recheck_matches_and_never_overwrites(self):
        out = self.root / 'recheck.json'
        self.assertEqual(self.run_main(out), 0)
        body = json.loads(out.read_text())
        self.assertTrue(body['recomputed_equal_judge'])
        self.assertTrue(all(body['checks']['arm_summaries'].values()))
        self.assertEqual(body['recomputed']['bounds'], self.judge['bounds'])
        self.assertEqual(body['experiment_sha256'], self.eid)
        o0 = body['o0_classes']
        self.assertEqual((o0['pairs'], o0['classes'], o0['pairs_per_class']), (300, 75, [4]))
        t = [p['pair_score'] for p in self.tables['O0_T_vs_champion']['pairs']]
        self.assertAlmostEqual(o0['T_score'], sum(t) / 300, places=6)
        self.assertAlmostEqual(o0['P_combined'], self.judge['bounds']['P']['estimate'], places=5)
        self.assertAlmostEqual(o0['D_combined'], self.judge['bounds']['D']['estimate'], places=5)
        before = out.read_text()
        self.assertEqual(self.run_main(out), 1)  # never overwritten
        self.assertEqual(out.read_text(), before)

    def test_arm_files_are_checked_against_the_judge_first(self):
        (self.arms_dir / 'O1_C_vs_champion.json').write_text('{"arm": "changed"}')
        import h7r_eval as cli
        with mock.patch.object(cli, 'load_arm', side_effect=AssertionError('loaded before the SHA check')):
            with self.assertRaisesRegex(ValueError, r"differ from the ones the judge read: \['O1_C_vs_champion'\]"):
                rc.recheck(self.arms_dir, self.root)
        out = self.root / 'recheck.json'
        self.assertEqual(self.run_main(out), 1)
        self.assertFalse(out.exists())

    def test_a_judge_that_differs_is_reported(self):
        bad = json.loads(json.dumps(self.judge))
        bad['bounds']['D']['lower_one_sided'] += 0.001
        bad['per_arm']['O0_T_vs_champion']['score'] += 0.01
        self.write_judge(bad, commit=True)
        out = self.root / 'recheck.json'
        self.assertEqual(self.run_main(out), 2)
        body = json.loads(out.read_text())
        self.assertFalse(body['recomputed_equal_judge'])
        self.assertFalse(body['checks']['opening_bounds'])
        self.assertTrue(body['checks']['class_bounds'])
        self.assertFalse(body['checks']['arm_summaries']['O0_T_vs_champion'])


STUB = """import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--output') + 1])
if os.environ.get('RECHECK_FAIL'):
    print('STOP: ValueError: arm records differ'); sys.exit(1)
if os.environ.get('RECHECK_FORBID'):
    print('STOP: must not run again'); sys.exit(9)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'recomputed_equal_judge': True}))
print('ok')
"""


class BlockTest(unittest.TestCase):
    """The desktop block of the draft run by bash on a real temporary repository with a bare origin, the real
    .gitignore and record tool; the recheck tool is a stand-in."""

    def setUp(self):
        import re
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
        (w / 'scripts/h7r_p1a1_recheck.py').write_text(STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-P1-A1-RECHECK-draft.md').read_text()
        blocks = re.findall(r'```bash\n(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 1)
        self.block = blocks[0].replace('~/renju-h6', str(w))
        self.O = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/arena-recheck.json'

    def bash(self, **env):
        import subprocess
        e = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.block], cwd=self.work, capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def test_pass_then_already_recorded(self):
        code, out = self.bash()
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(self.O))
        self.assertEqual(len(self.files('docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/stage-logs')), 2)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        code, out = self.bash()
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_a_stop_is_logged_and_an_uncommitted_result_is_recorded_without_a_rerun(self):
        import time
        code, out = self.bash(RECHECK_FAIL='1')
        self.assertEqual(code, 1, out)
        self.assertFalse(self.files(self.O))
        logs = [f for f in self.files('docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/stage-logs') if f.endswith('.log')]
        self.assertEqual(len(logs), 1)
        self.assertIn('__EXIT__ 1', (self.work / logs[0]).read_text())
        self.assertIn('Arena recheck exit 1', git(self.work, 'log', '-1', '--format=%s'))
        time.sleep(1.1)
        (self.work / self.O).write_text('{"recomputed_equal_judge": true}')  # finished, not committed
        code, out = self.bash(RECHECK_FORBID='1')
        self.assertEqual(code, 0, out)
        self.assertIn('no new recheck', out)
        self.assertTrue(self.files(self.O))


if __name__ == '__main__':
    unittest.main()
