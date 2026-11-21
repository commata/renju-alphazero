"""TB-V3-BUDGET-DIAG-1 (``scripts/h7r_budget_diag.py``): the aggregate judgement; the deepen items (UNKNOWN moves
only, once, S400's unresolved runs first); the per-move reading of B1 / B2 / B3; a real process killed at its limit
(TIMEOUT) and the total cap (NOT_RUN); the label rule of ``e1_evaluate.evaluate_position`` kept apart from the
diagnostic recheck; prepare, replay, deepen and verify on a temporary repository, deepen also after NOT_REPRODUCED;
the desktop blocks, and d-P with the real tool. The engine, the truth solver and the deepen runner are stand-ins."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock

import h7r_budget_diag as dg
from tests.test_h7r_budget import B_STUB, SITE, stand_in
from tests.test_h7r_s400_train import QUIET, git

ROOT = Path(__file__).resolve().parents[1]
E1P = [p for p in json.loads((ROOT / dg.E1_MANIFEST).read_text())['positions'] if p['class'] == 'E1-P']


def label_moves(position):
    """A manifest move labelled VCT2_CLEAR and one labelled UNKNOWN in this position."""
    t = position['truths']
    clear = next(k for k, v in t.items() if v['status'] == 'VCT2_CLEAR')
    unknown = next((k for k, v in t.items() if v['status'] == 'UNKNOWN'), None)
    return [int(x) for x in clear.split(',')], unknown and [int(x) for x in unknown.split(',')]


def run(seed, pre, played, outcome, labels, mode='tactical'):
    b0 = {'status': labels['pre'], 'lost_depth': None, 'nodes_used': 10, 'vcf_calls': 2, 'vcf_cut_calls': 1,
          'budget_exhausted': False, 'seconds': 0.1}
    return {'seed': seed, 'route': 'tree', 'simulation_mode': mode, 'selected_simulations': 800,
            'tree_simulations': 800, 'root_tactical_score': 99, 'pre': pre, 'played': played, 'switched': False,
            'outcome': outcome, 'labels': labels, 'label_source': {'pre': 'manifest', 'played': 'manifest'},
            'b0_recheck': {'pre': b0, 'played': b0}, 'checked': [], 'children': [], 'consistent': True}


def synthetic_rows(failures=3):
    """S50: every run avoided the loss; S400: ``failures`` runs on the first position's UNKNOWN move."""
    rows = []
    for f in dg.FACTORS:
        for i, p in enumerate(E1P):
            clear, unknown = label_moves(p)
            runs = []
            for j, seed in enumerate(p['e1_seeds']):
                fail = f == 8 and i == 0 and j < failures and unknown is not None
                runs.append(run(seed, unknown if fail else clear, unknown if fail else clear,
                                'TRUTH_UNRESOLVED' if fail else 'TREE_AVOIDED',
                                {'pre': 'UNKNOWN' if fail else 'VCT2_CLEAR', 'played': 'UNKNOWN' if fail else
                                 'VCT2_CLEAR'}))
            rows.append({'key': p['key'], 'factor': f, 'runs': runs, 'truths_added': {}})
    return rows


class RuleTest(unittest.TestCase):
    def test_the_aggregate_judgement(self):
        rows = [{**r, 'position': r['key']} for r in synthetic_rows()]
        self.assertEqual(dg.replay_summary(rows)['judgement'], 'AGGREGATE_REPRODUCED')
        rows2 = [{**r, 'position': r['key']} for r in synthetic_rows(failures=2)]
        s = dg.replay_summary(rows2)
        self.assertEqual((s['judgement'], s['outcomes']['8']), ('NOT_REPRODUCED', {'TREE_AVOIDED': 13,
                                                                                     'TRUTH_UNRESOLVED': 2}))
        self.assertIn('aggregate judgement only', s['note'])

    def test_items_once_failures_first(self):
        rows = [{**r, 'position': r['key']} for r in synthetic_rows()]
        rows[0]['runs'][0]['labels']['pre'] = 'UNKNOWN'  # an S50 UNKNOWN first candidate too (same position)
        items = dg.items_of(rows, ROOT)
        self.assertEqual(len(items), 2)  # the S400 failure move once (three runs), then the S50 one
        self.assertEqual((items[0]['from']['factor'], items[0]['from']['outcome']), (8, 'TRUTH_UNRESOLVED'))
        self.assertEqual(items[1]['from']['factor'], 1)

    def test_the_reading(self):
        r = lambda *s: {b: {'status': x} for b, x in zip(dg.BUDGETS, s)}  # noqa: E731
        self.assertEqual(dg.resolved_by(r('VCT2_CLEAR', 'UNKNOWN', 'VCT2_CLEAR')), 'B1 only')
        self.assertEqual(dg.resolved_by(r('UNKNOWN', 'PROVEN_LOSS', 'PROVEN_LOSS')), 'B2 only')
        self.assertEqual(dg.resolved_by(r('UNKNOWN', 'UNKNOWN', 'VCT2_CLEAR')), 'B3 only')
        self.assertEqual(dg.resolved_by(r('VCT2_CLEAR', 'VCT2_CLEAR', 'VCT2_CLEAR')), 'B1 and B2')
        self.assertEqual(dg.resolved_by(r('UNKNOWN', 'UNKNOWN', 'UNKNOWN')), 'none')
        self.assertEqual(dg.resolved_by(r('UNKNOWN', 'TIMEOUT', 'VCT2_CLEAR')), 'incomplete')
        self.assertEqual(dg.final_status(r('UNKNOWN', 'TIMEOUT', 'NOT_RUN')), 'INCOMPLETE')
        item = lambda res, mode='tactical': {'from': {'factor': 8, 'outcome': 'TRUTH_UNRESOLVED',  # noqa: E731
                                                      'simulation_mode': mode}, 'results': res}
        rep = dg.report([item(r('PROVEN_LOSS', 'PROVEN_LOSS', 'PROVEN_LOSS')),
                         item(r('UNKNOWN', 'UNKNOWN', 'UNKNOWN'), 'normal')])
        self.assertEqual(rep['failure_final'], {'PROVEN_LOSS': 1, 'UNKNOWN': 1})  # mixed: both counts
        self.assertTrue(rep['tactical_800_with_proven_loss'])
        self.assertEqual(len(rep['use']), 2)
        clear = dg.report([item(r('VCT2_CLEAR', 'UNKNOWN', 'VCT2_CLEAR'))])
        self.assertIn('all VCT2_CLEAR', clear['use'][0])


class TimeoutTest(unittest.TestCase):
    def test_a_real_process_is_killed_at_its_limit(self):
        started = time.perf_counter()
        result, seconds = dg.run_with_timeout(time.sleep, (30,), 1.0)
        self.assertEqual(result, 'TIMEOUT')
        self.assertLess(time.perf_counter() - started, 20)
        result, _ = dg.run_with_timeout(math.sqrt, (16.0,), 30)
        self.assertEqual(result, 4.0)


class FakeDiag:
    def __init__(self, move):
        self.v8_vct2_checked = [(tuple(move), 'UNKNOWN')]
        self.v8_vct2_switched = False
        self.v8_root_visits = [(tuple(move), 40, 0.1)]
        self.v8_vct2_seconds, self.v8_vct2_nodes, self.v8_route = 0.0, 0, 'tree'
        self.simulation_mode, self.selected_simulations, self.v8_tree_simulations = 'tactical', 800, 800
        self.best_root_tactical_score = 77


class FakeAgent:
    def __init__(self, move):
        self.move, self.diagnostics = move, None

    def select_move(self, game):
        self.diagnostics = FakeDiag(self.move)
        return tuple(self.move)


class ReplayPositionTest(unittest.TestCase):
    """``evaluate_position`` itself: the manifest label decides the outcome; the B0 recheck sits beside it."""

    def test_labels_and_recheck_apart(self):
        position = next(p for p in E1P if label_moves(p)[1] is not None)
        clear, unknown = label_moves(position)
        rechecks = []

        def recheck(history, move, budget):
            rechecks.append((tuple(move), budget))
            return {'status': 'VCT2_CLEAR', 'lost_depth': None, 'nodes_used': 1, 'vcf_calls': 1, 'vcf_cut_calls': 0,
                    'budget_exhausted': False, 'seconds': 0.0}
        for move, want in ((unknown, 'TRUTH_UNRESOLVED'), (clear, 'TREE_AVOIDED')):
            rechecks.clear()
            with mock.patch('h7c_budget.make_budget_agent', lambda f, p, v, s, d: FakeAgent(move)), \
                    mock.patch.object(dg, 'recheck', recheck):
                row = dg.replay_position((position, 8, 'p', 'v', 'cuda'))
            for r in row['runs']:
                self.assertEqual(r['outcome'], want)  # the manifest label, not the recheck's CLEAR
                self.assertEqual(r['label_source'], {'pre': 'manifest', 'played': 'manifest'})
                self.assertEqual((r['simulation_mode'], r['selected_simulations'], r['root_tactical_score']),
                                 ('tactical', 800, 77))
                self.assertEqual(r['b0_recheck']['pre']['status'], 'VCT2_CLEAR')
            self.assertTrue(all(b == dg.B0 for _, b in rechecks))


# --- the stages on a temporary repository ------------------------------------------------------------------------------

class StageBase(unittest.TestCase):
    def setUp(self):
        from analysis import tb_v3_net
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', root / '.gitignore')
        for rel in (dg.E1_MANIFEST, dg.BUDGET_CONFIRM):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        stand_in(root, dg.MODELS['policy'], 'H3 policy', 'policy')
        stand_in(root, dg.MODELS['value'], 'H6 v3 value', 'value')
        ident = tb_v3_net.model_identity(root / dg.MODELS['policy'], root / dg.MODELS['value'])
        p = mock.patch.dict('analysis.tb_v3.MODELS', ident)
        p.start()
        self.addCleanup(p.stop)
        self.c = dg.contract_body(root, sources=())
        self.write(self.c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.failures = 3

    def write(self, c):
        path = self.root / dg.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, indent=1) + '\n')

    def approve(self, *stages):
        for s in stages:
            self.c['approved'][s] = True
        self.write(self.c)
        git(self.root, 'commit', '-qam', 'approve ' + ' '.join(stages))

    def args(self, **k):
        return argparse.Namespace(contract=None, device='cuda', log_name=None, threads_for_tests=True, **k)

    def commit(self, rel):
        git(self.root, 'add', rel)
        git(self.root, 'commit', '-qm', rel)

    def fake_position(self, task):
        position, factor = task[0], task[1]
        return next(r for r in synthetic_rows(self.failures) if r['key'] == position['key'] and r['factor'] == factor)

    def through_replay(self):
        self.approve('prepare', 'replay')
        dg.prepare(self.args(), self.root)
        dg.verify(argparse.Namespace(contract=None, stage='prepare'), self.root)
        self.commit(self.c['output']['prepare'])
        with mock.patch.object(dg, 'replay_position', self.fake_position):
            rec = dg.replay(self.args(), self.root, log=QUIET)
        dg.verify(argparse.Namespace(contract=None, stage='replay'), self.root)
        self.commit(self.c['output']['replay'])
        return rec


class StageTest(StageBase):
    def test_prepare_replay_deepen(self):
        rec = self.through_replay()
        self.assertEqual((rec['judgement'], rec['outcomes']), ('AGGREGATE_REPRODUCED', dg.EXPECTED))
        self.approve('deepen')
        calls = []
        answers = {'B1': 'UNKNOWN', 'B2': 'PROVEN_LOSS', 'B3': 'PROVEN_LOSS'}

        def runner(item, budget, seconds):
            calls.append((item['position'], budget, seconds))
            return {'status': answers[budget], 'lost_depth': 2, 'nodes_used': 5, 'vcf_calls': 3, 'vcf_cut_calls': 0,
                    'budget_exhausted': False}, 60.0
        out = dg.deepen(self.args(), self.root, log=QUIET, runner=runner)
        self.assertEqual([b for _, b, _ in calls], ['B1', 'B2', 'B3'])  # one UNKNOWN move, three budgets
        item = out['items'][0]
        self.assertEqual((item['resolved_by'], item['final'], item['b0']['vcf_cut_calls']), ('B2 only', 'PROVEN_LOSS', 1))
        self.assertEqual(set(item['results']), set(dg.BUDGETS))
        self.assertEqual(out['report']['failure_final'], {'PROVEN_LOSS': 1})
        self.assertTrue(out['report']['tactical_800_with_proven_loss'])
        dg.verify(argparse.Namespace(contract=None, stage='deepen'), self.root)

    def test_deepen_after_not_reproduced(self):
        self.failures = 2
        rec = self.through_replay()
        self.assertEqual(rec['judgement'], 'NOT_REPRODUCED')
        self.approve('deepen')
        out = dg.deepen(self.args(), self.root, log=QUIET, runner=lambda i, b, s: (
            {'status': 'VCT2_CLEAR', 'lost_depth': None, 'nodes_used': 1, 'vcf_calls': 1, 'vcf_cut_calls': 0,
             'budget_exhausted': False}, 1.0))
        self.assertEqual((out['replay_judgement'], out['report']['failure_final']), ('NOT_REPRODUCED',
                                                                                      {'VCT2_CLEAR': 1}))

    def test_timeouts_and_the_total_cap(self):
        self.through_replay()
        self.approve('deepen')
        limits = []
        unknown = {'status': 'UNKNOWN', 'lost_depth': None, 'nodes_used': 9, 'vcf_calls': 9, 'vcf_cut_calls': 3,
                   'budget_exhausted': True}

        def runner(item, budget, seconds):
            limits.append(seconds)
            return (unknown, 4000.0) if budget == 'B1' else ('TIMEOUT', float(seconds))
        out = dg.deepen(self.args(), self.root, log=QUIET, runner=runner)
        self.assertEqual(limits, [1800, 1800, 1400])  # B3 waits only for what the 2 hour total leaves
        res = out['items'][0]['results']
        self.assertEqual([res[b]['status'] for b in dg.BUDGETS], ['UNKNOWN', 'TIMEOUT', 'TIMEOUT'])
        self.assertEqual((res['B2']['reason'], res['B3']['reason']),
                         ('killed at the task limit', 'killed at the remaining total'))
        self.assertEqual((out['items'][0]['resolved_by'], out['items'][0]['final']), ('incomplete', 'INCOMPLETE'))
        self.assertEqual(out['report']['failure_final'], {'INCOMPLETE': 1})

    def test_not_run_past_the_total(self):
        self.through_replay()
        self.approve('deepen')
        calls = []

        def runner(item, budget, seconds):
            calls.append(budget)
            return {'status': 'UNKNOWN', 'lost_depth': None, 'nodes_used': 1, 'vcf_calls': 1, 'vcf_cut_calls': 1,
                    'budget_exhausted': False}, float(dg.TOTAL_SECONDS)
        out = dg.deepen(self.args(), self.root, log=QUIET, runner=runner)
        res = out['items'][0]['results']
        self.assertEqual((calls, [res[b]['status'] for b in dg.BUDGETS]), (['B1'], ['UNKNOWN', 'NOT_RUN', 'NOT_RUN']))
        # a rerun of the same stage (the record removed) tries nothing more: the journal holds the used time
        (self.root / self.c['output']['deepen']).unlink()
        calls.clear()
        again = dg.deepen(self.args(), self.root, log=QUIET, runner=runner)
        self.assertEqual((calls, again['items'][0]['results']['B2']['status']), ([], 'NOT_RUN'))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'the prepare stage is not approved'):
            dg.prepare(self.args(), self.root)
        self.approve('prepare', 'replay')
        dg.prepare(self.args(), self.root)
        self.commit(self.c['output']['prepare'])
        args = self.args()
        args.device = 'cpu'
        with self.assertRaisesRegex(ValueError, "runs on cuda only"):
            dg.replay(args, self.root, log=QUIET)
        path = self.root / self.c['output']['prepare']
        rec = json.loads(path.read_text())
        path.write_text(json.dumps({k: v for k, v in rec.items() if k != 'passed'}))
        with self.assertRaisesRegex(ValueError, 'not a passed prepare'):
            dg.verify(argparse.Namespace(contract=None, stage='prepare'), self.root)

    def test_the_contract(self):
        self.write({**self.c, 'deepen': {**self.c['deepen'], 'total_seconds': 99}})
        git(self.root, 'commit', '-qam', 'cap changed')
        with self.assertRaisesRegex(ValueError, 'the budgets, limits or expectation differ'):
            dg.load_contract(None, self.root)


class FixedContractTest(unittest.TestCase):
    def test_the_committed_contract(self):
        c, _, eid = dg.load_contract(None, ROOT)
        self.assertEqual(list(c['approved']), list(dg.STAGES))
        self.assertEqual({k: v for k, v in c.items() if k not in ('sources', 'approved')},
                         {k: v for k, v in dg.contract_body(ROOT).items() if k not in ('sources', 'approved')})
        self.assertEqual([s['path'] for s in c['sources']], list(dg.SOURCES))
        self.assertEqual(eid, '74c8b77a0f96372d91c5fe43307a3d54e4f44219909522540d842d6eae550f64')


# --- the desktop blocks ----------------------------------------------------------------------------------------------

def blocks(work):
    doc = (ROOT / 'docs/h7r/budget/TB-V3-BUDGET-DIAG-1-draft.md').read_text()
    found = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
    assert len(found) == 3
    return dict(zip(('d-P', 'd-R', 'd-D'), (x.replace('~/renju-h6', Path(work).as_posix()) for x in found)))


class BlockTest(unittest.TestCase):
    def setUp(self):
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
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', 'scripts/h7r_attempts.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        (w / 'scripts/h7r_budget_diag.py').write_text(B_STUB.replace('docs/h7r/budget/TB-V3-BUDGET-1',
                                                                     dg.D).replace('h7r-budget-', 'h7r-budget-diag-'))
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        (w / dg.D).mkdir(parents=True)
        (w / dg.CONTRACT).write_text('{}')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        self.blocks = blocks(w)

    def bash(self, name, **env):
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True,
                           env={**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}, start_new_session=True)
        return r.returncode, r.stdout + r.stderr

    def subjects(self):
        return git(self.work, 'log', '--format=%s').splitlines()

    def test_the_three_stages_and_a_resumed_failure(self):
        code, out = self.bash('d-R')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        self.assertEqual(self.bash('d-P')[0], 0)
        code, out = self.bash('d-R', B_FAIL='replay')
        self.assertEqual(code, 1, out)
        self.assertIn('__FAILED__ 1', out)
        head = git(self.work, 'rev-parse', 'HEAD')
        code, out = self.bash('d-R')
        self.assertEqual(code, 0, out)
        self.assertIn('resuming replay at ' + head, out)
        self.assertEqual(self.bash('d-D')[0], 0)
        self.assertEqual(self.subjects()[:3], [f'H7-R BUDGET DIAG: {n} exit 0 (records only)' for n in
                                               ('deepen', 'replay', 'prepare')])
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))


class RealPrepareBlockTest(unittest.TestCase):
    """d-P with the real tool and record tool: only the frozen checker and the model files are stand-ins."""

    def test_d_p_records_a_verified_prepare(self):
        from analysis import tb_v3_net
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        origin, w = base / 'origin.git', base / 'work'
        git(base, 'init', '-q', '--bare', str(origin))
        git(base, 'init', '-q', '-b', 'feat/h7-c1', str(w))
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', w / '.gitignore')
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
        for rel in ('src', 'scripts', 'docs/mcts-v8-results/e1'):
            shutil.copytree(ROOT / rel, w / rel, ignore=ignore)
        (w / dg.BUDGET_CONFIRM).parent.mkdir(parents=True)
        shutil.copy(ROOT / dg.BUDGET_CONFIRM, w / dg.BUDGET_CONFIRM)
        (w / 'src/sitecustomize.py').write_text(SITE)
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        stand_in(w, dg.MODELS['policy'], 'H3 policy', 'policy')
        stand_in(w, dg.MODELS['value'], 'H6 v3 value', 'value')
        ident = tb_v3_net.model_identity(w / dg.MODELS['policy'], w / dg.MODELS['value'])
        with mock.patch.dict('analysis.tb_v3.MODELS', ident):
            c = dg.contract_body(w)
        c['approved']['prepare'] = True
        (w / dg.D).mkdir(parents=True)
        (w / dg.CONTRACT).write_text(json.dumps(c, indent=1) + chr(10))
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'the tool and the approved contract')
        git(w, 'remote', 'add', 'origin', str(origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        env = {k: v for k, v in os.environ.items() if k not in ('CUBLAS_WORKSPACE_CONFIG', 'PYTHONPATH')}
        r = subprocess.run(['bash', '-c', blocks(w)['d-P']], cwd=w, capture_output=True, text=True,
                           start_new_session=True,
                           env={**env, 'PYTHONDONTWRITEBYTECODE': '1', 'B_TEST_MODELS': json.dumps(ident)})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        rec = json.loads((w / dg.D / 'prepare.json').read_text())
        self.assertEqual((rec['passed'], len(rec['positions'])), (True, 5))
        self.assertEqual(git(w, 'log', '-1', '--format=%s'), 'H7-R BUDGET DIAG: prepare exit 0 (records only)')
