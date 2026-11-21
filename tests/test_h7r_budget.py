"""TB-V3-BUDGET-1 (``scripts/h7r_budget.py``): the selection rule (order, the 5.5 boundary, p95, none); every move
timed and attributed to its side and budget across the colour-swapped pair; the E1 gate (a pre-named E1 metric that
gets worse fails it even when ``h7c_arena.tactical_judge`` passes); the confirmation sample of the checkout; and on a
temporary repository: prepare, timing, confirm and verify, the fixed sample, resuming on that sample only; the
desktop blocks under real bash. The engine games and E1 runs are stand-ins; the records, checks and rules are the
tool's."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7c_arena as ar
import h7r_budget as b
import h7r_eval as cli
from hybrid import h7r_openings as op
from renju import BLACK
from tests.test_h7r_s400_train import QUIET, git

ROOT = Path(__file__).resolve().parents[1]
PLAY_GAME = b.play_game  # the tool's game loop (the stage tests patch the module's name)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeAgent:
    """Plays the moves of a fixed finished game; each move takes ``cost(factor, ply)`` on the fake clock."""

    def __init__(self, factor, clock, moves, cost):
        self.factor, self.clock, self.moves, self.cost = factor, clock, moves, cost

    def select_move(self, game):
        n = len(game.history)
        self.clock.now += self.cost(self.factor, n)
        return tuple(self.moves[n])


def finished_moves(opening, winner):
    from tests.h7r_helpers import winning_moves
    return winning_moves(opening, winner)


def default_cost(scale):
    return lambda factor, ply: 0.01 * (scale.get(factor, factor) if factor != 1 else 1) * (1 + (ply % 5) / 10)


def fake_play(cost, wins=lambda task: True):
    """``h7r_budget.play_game`` with fake agents and clock: the real loop and records."""
    def play(task, make_agent=None):
        clock = Clock()
        tested = task['tested_color']
        winner = tested if wins(task) else -tested
        moves = finished_moves(task['opening'], winner)
        with mock.patch.object(b, 'perf_counter', clock):
            return PLAY_GAME(task, make_agent=lambda factor, seed: FakeAgent(factor, clock, moves, cost))
    return play


def e1_metrics(rescue=(8, 10), veto=(3, 4), false_veto=1, unsound=0):
    r = lambda n, d: {'value': round(n / d, 3) if d else None, 'n': f'{n}/{d}'}  # noqa: E731
    return {'metrics': {'rescue_rate': r(*rescue), 'veto_recall': r(*veto), 'false_veto': false_veto,
                        'unsound_witness': unsound, 'rescue_pool_coverage': r(9, 10), 'inconsistent_runs': 0}}


# --- rules ---------------------------------------------------------------------------------------------------------

class RuleTest(unittest.TestCase):
    def t(self, mean, p95):
        return {'ratio': {'mean': mean, 'p95': p95}}

    def test_the_selection_rule(self):
        self.assertEqual(b.select({8: self.t(5.5, 5.5)}), 8)  # 5.5 itself is within the cap
        self.assertEqual(b.select({8: self.t(4.9, 6.2), 4: self.t(2.9, 3.4)}), 4)  # p95 alone fails S400
        self.assertEqual(b.select({8: self.t(5.6, 5.0), 4: self.t(5.51, 2.0), 2: self.t(1.6, 1.9)}), 2)
        self.assertIsNone(b.select({8: self.t(6, 6), 4: self.t(6, 6), 2: self.t(6, 6)}))  # END_NO_BUDGET
        self.assertIsNone(b.select({8: self.t(6, 6)}))  # never a factor that was not measured

    def test_percentile_and_ratios(self):
        self.assertEqual(b.percentile([float(i) for i in range(1, 101)], 0.95), 95.0)
        self.assertEqual(b.percentile([1.0, 2.0], 0.95), 2.0)
        games = [{'move_times': [{'side': 'tested', 'seconds': 0.5}, {'side': 'opponent', 'seconds': 0.1},
                                 {'side': 'tested', 'seconds': 0.5}, {'side': 'opponent', 'seconds': 0.1}]}]
        t = b.times_of(games)
        self.assertEqual(t['ratio'], {'mean': 5.0, 'p95': 5.0})
        self.assertEqual((t['tested']['mean_seconds'], t['opponent_s50']['mean_seconds']), (0.5, 0.1))


class GameTest(unittest.TestCase):
    """Every move timed, and attributed to the side and budget that played it, in both games of a pair."""

    def setUp(self):
        body = json.loads((ROOT / b.TIMING_SUITE[1]).read_text())
        self.openings = op.pick(body, 2, 9112)
        self.arm = b.arm_of('timing', 8, 9112)
        self.tasks = cli.build_tasks(self.arm, self.openings)
        self.ctx = {'policy': 'p', 'value': 'v', 'factor': 8, 'device': 'cuda', 'signature': 's' * 64,
                    'arm': self.arm['name']}

    def play(self, task):
        return fake_play(default_cost({8: 5.0}))({**task, 'context': self.ctx})

    def test_both_colours(self):
        for task in self.tasks[:2]:  # one opening, the tested side black then white
            g = self.play(task)
            b.check_game(g, self.tasks, self.arm['name'], 's' * 64, 8)
            parity = 0 if task['tested_color'] == BLACK else 1
            self.assertEqual([t['ply'] for t in g['move_times']], list(range(len(task['opening']), g['length'])))
            for t in g['move_times']:
                tested = t['ply'] % 2 == parity
                self.assertEqual((t['side'], t['factor']), ('tested', 8) if tested else ('opponent', 1))
                want = 0.01 * (5.0 if tested else 1) * (1 + (t['ply'] % 5) / 10)
                self.assertAlmostEqual(t['seconds'], want, places=5)
        self.assertAlmostEqual(b.times_of([self.play(t) for t in self.tasks])['ratio']['mean'], 5.0, delta=0.2)

    def test_a_missing_or_misattributed_time_is_refused(self):
        g = self.play(self.tasks[1])
        for change, message in ((lambda x: x['move_times'].pop(3), 'a move without its time'),
                                (lambda x: x['move_times'][0].update(side='opponent', factor=1), "is not the tested side's"),
                                (lambda x: x['move_times'][1].update(factor=8), "is not the opponent side's"),
                                (lambda x: x['move_times'][2].update(seconds=float('nan')), 'seconds')):
            with self.subTest(message):
                bad = json.loads(json.dumps(g, allow_nan=True))
                change(bad)
                with self.assertRaisesRegex(ValueError, message):
                    b.check_game(bad, self.tasks, self.arm['name'], 's' * 64, 8)


class TacticalTest(unittest.TestCase):
    def legacy_side(self, e1):
        fresh = {'file_sha256': 'f', 'pass': True,
                 **{k: {'correct': 20, 'probes': 20, 'accuracy': 1.0, 'mean_value': 0.0} for k in ('win', 'loss')}}
        return {'e1': e1, 'must_block': {'hits': 40, 'anchor': 38},
                'value_probes': {'device': 'cuda', 'inference': 'd4_average', 'fresh': fresh, 'p93_p94': {}}}

    def test_a_worse_pre_named_metric_fails_even_when_tactical_judge_passes(self):
        budget, s50 = e1_metrics(rescue=(7, 10)), e1_metrics(rescue=(8, 10))
        legacy = ar.tactical_judge(self.legacy_side(budget), self.legacy_side(s50))
        self.assertEqual((legacy['verdict'], legacy['regressions_reported']), ('PASS', ['rescue_rate']))
        gate = b.tactical_gate(budget, s50)
        self.assertEqual(gate['verdict'], 'FAIL')
        self.assertFalse(gate['items']['rescue_rate >= S50'])

    def test_the_gate(self):
        self.assertEqual(b.tactical_gate(e1_metrics(), e1_metrics())['verdict'], 'PASS')  # equal is no regression
        self.assertEqual(b.tactical_gate(e1_metrics(rescue=(9, 10), false_veto=0), e1_metrics())['verdict'], 'PASS')
        self.assertEqual(b.tactical_gate(e1_metrics(false_veto=2), e1_metrics())['verdict'], 'FAIL')
        self.assertEqual(b.tactical_gate(e1_metrics(unsound=1), e1_metrics(unsound=1))['verdict'], 'FAIL')
        self.assertEqual(b.tactical_gate(e1_metrics(veto=(6, 8)), e1_metrics(veto=(3, 4)))['verdict'], 'PASS')  # 0.75
        both = b.tactical_gate(e1_metrics(veto=(0, 0)), e1_metrics(veto=(0, 0)))
        self.assertTrue(both['items']['veto_recall: no data on either side'])
        one = b.tactical_gate(e1_metrics(veto=(0, 0)), e1_metrics())
        self.assertEqual(one['verdict'], 'FAIL')
        self.assertIn('veto_recall: not comparable (a denominator on one side only)', one['items'])


class SampleTest(unittest.TestCase):
    def test_the_confirmation_sample_of_the_checkout(self):
        c = b.contract_body(ROOT, sources=())
        s = b.confirmation_sample(c, ROOT)
        self.assertEqual((len(s['opening_ids']), s['distinct_classes']), (300, 299))
        self.assertEqual(len(s['shared_classes_with']['O1-confirm-anchor']), 6)
        self.assertEqual({k: v for k, v in s['shared_classes_with'].items() if k.startswith('O1-dev')},
                         {'O1-dev-a': 0, 'O1-dev-b': 0, 'O1-dev-c': 0, 'O1-dev-d': 0})


# --- the stages on a temporary repository ------------------------------------------------------------------------------

def stand_in(root, rel, text, meta):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    (path.with_suffix('.json') if meta == 'policy' else path.with_name('best.json')).write_text(text + ' metadata')


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
        for rel in ('docs/h7r/suites', 'docs/h7r/contracts', 'docs/h7r/contracts-v2'):
            shutil.copytree(ROOT / rel, root / rel)
        stand_in(root, b.MODELS['policy'], 'H3 policy', 'policy')
        stand_in(root, b.MODELS['value'], 'H6 v3 value', 'value')
        ident = tb_v3_net.model_identity(root / b.MODELS['policy'], root / b.MODELS['value'])
        for p in (mock.patch.dict('analysis.tb_v3.MODELS', ident),):
            p.start()
            self.addCleanup(p.stop)
        self.c = b.contract_body(root, sources=())
        self.write(self.c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.cost = default_cost({8: 5.0})
        self.e1 = {8: e1_metrics(), 4: e1_metrics(), 2: e1_metrics(), 1: e1_metrics()}
        self.played = []

    def write(self, c):
        path = self.root / b.CONTRACT
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

    def run_stage(self, name, commit=True, stop_after=None):
        state = {'commit': git(self.root, 'rev-parse', 'HEAD'), 'dirty': False}
        play = fake_play(self.cost, lambda task: task['tested_color'] == BLACK or task['pair'] % 3 != 0)
        count = {'n': 0}

        def counted(task, make_agent=None):
            if stop_after is not None and count['n'] >= stop_after:
                raise KeyboardInterrupt('the block died')
            count['n'] += 1
            self.played.append(task['key'])
            return play(task)

        def e1(factor, root, device, log=print, workers=6, threads_for_tests=False):
            return {'factor': factor, 'manifest_sha256': b.file_sha256(ar.E1_MANIFEST), 'positions': 3,
                    **self.e1[factor], 'outcomes': {}}
        with mock.patch.object(b, 'play_game', counted), mock.patch.object(cli, '_git', lambda _: dict(state)), \
                mock.patch.object(b, 'e1_suite', e1):
            rec = {'prepare': b.prepare, 'timing': b.timing, 'confirm': b.confirm}[name](self.args(), self.root,
                                                                                       **({} if name == 'prepare'
                                                                                          else {'log': QUIET}))
        b.verify(argparse.Namespace(contract=None, stage=name), self.root)
        if commit:
            self.commit(self.c['output'][name])
        return rec

    def through_timing(self):
        self.approve('prepare', 'timing')
        self.run_stage('prepare')
        return self.run_stage('timing')


class StageTest(StageBase):
    def test_prepare_timing_confirm(self):
        self.through_timing()
        trec = json.loads((self.root / self.c['output']['timing']).read_text())
        self.assertEqual((trec['decision'], trec['selected'], list(trec['measured'])), ('SELECTED', 'S400', ['S400']))
        self.assertTrue(trec['measured']['S400']['within_cap'])
        self.assertAlmostEqual(trec['measured']['S400']['ratio']['mean'], 5.0, delta=0.2)
        self.assertEqual(trec['confirmation']['opening_ids'],
                         json.loads((self.root / self.c['output']['prepare']).read_text())['confirmation_sample']
                         ['opening_ids'])
        self.assertEqual(len(self.played), 2 * b.TIMING_PAIRS)
        self.approve('confirm')
        self.played.clear()
        rec = self.run_stage('confirm')
        self.assertEqual(len(self.played), 2 * b.PAIRS)
        self.assertEqual((rec['pairs'], rec['reading'], rec['verdict'], rec['failed']), (300, 'STRONGER', 'PASS', []))
        self.assertEqual(rec['tactical']['gate']['verdict'], 'PASS')
        self.assertEqual(set(rec['times']), {'tested', 'opponent_s50', 'ratio'})
        self.assertGreater(rec['times']['tested']['mean_seconds'], rec['times']['opponent_s50']['mean_seconds'])

    def test_s400_over_the_cap_selects_s200(self):
        self.cost = default_cost({8: 6.0, 4: 3.0})
        trec = self.through_timing()
        self.assertEqual((trec['selected'], list(trec['measured'])), ('S200', ['S400', 'S200']))
        self.assertFalse(trec['measured']['S400']['within_cap'])

    def test_no_budget_within_the_cap(self):
        self.cost = default_cost({8: 9.0, 4: 6.0, 2: 5.6})
        trec = self.through_timing()
        self.assertEqual((trec['decision'], trec['selected'], trec['confirmation']), ('END_NO_BUDGET', None, None))
        self.approve('confirm')
        with self.assertRaisesRegex(ValueError, 'the timing stage ended END_NO_BUDGET'):
            self.run_stage('confirm', commit=False)

    def test_a_worse_e1_metric_ends_the_confirmation(self):
        self.through_timing()
        self.approve('confirm')
        self.e1[8] = e1_metrics(rescue=(7, 10))
        rec = self.run_stage('confirm')
        self.assertEqual((rec['verdict'], rec['failed']), ('END', ['E1 gate']))

    def test_the_confirmation_resumes_on_the_fixed_sample_only(self):
        self.through_timing()
        self.approve('confirm')
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage('confirm', commit=False, stop_after=50)
        self.assertFalse((self.root / self.c['output']['confirm']).exists())
        jsonl = self.root / b.RUNS / 'confirm.jsonl'
        kept = {json.loads(line)['key'] for line in jsonl.read_text().splitlines() if line.strip()}
        self.assertTrue(0 < len(kept) <= 50)
        self.played.clear()
        rec = self.run_stage('confirm')
        self.assertEqual((len(self.played), rec['pairs']), (2 * b.PAIRS - len(kept), 300))  # only the missing games
        self.assertFalse(kept & set(self.played))
        # a changed fixed sample is refused before any game
        path = self.root / self.c['output']['timing']
        t = json.loads(path.read_text())
        t['confirmation']['opening_ids'][0] = t['confirmation']['opening_ids'][1]
        with self.assertRaisesRegex(ValueError, 'not the one the timing record fixed'):
            b.confirm_openings(self.c, t, self.root)

    def test_verify_recomputes_the_records(self):
        self.through_timing()
        jsonl = self.root / b.RUNS / 'timing-S400.jsonl'
        lines = jsonl.read_text().splitlines()
        g = json.loads(lines[0])
        g['move_times'][0]['seconds'] += 1.0
        jsonl.write_text('\n'.join([json.dumps(g, separators=(',', ':')), *lines[1:]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'timing-S400.jsonl: not the recorded games'):
            b.verify(argparse.Namespace(contract=None, stage='timing'), self.root)

    def test_prepare_refusals(self):
        with self.assertRaisesRegex(ValueError, 'the prepare stage is not approved'):
            b.prepare(self.args(), self.root)
        self.approve('prepare')
        other = {'format': 'h7r-eval-contract-v2', 'contract_id': 'OTHER', 'arms': [{'seed': 9111}]}
        path = self.root / 'docs/h7r/contracts-v2/OTHER.json'
        path.write_text(json.dumps(other))
        self.commit('docs/h7r/contracts-v2/OTHER.json')
        with self.assertRaisesRegex(ValueError, r'the contract seeds are already used: \{9111'):
            b.prepare(self.args(), self.root)

    def test_the_contract(self):
        self.write({**self.c, 'timing': {**self.c['timing'], 'cap': 6.0}})
        git(self.root, 'commit', '-qam', 'cap changed')
        with self.assertRaisesRegex(ValueError, 'the budgets, cap, samples, seeds or rules differ'):
            b.load_contract(None, self.root)
        self.write({**self.c, 'approved': {'prepare': False, 'timing': True, 'confirm': False}})
        git(self.root, 'commit', '-qam', 'out of order')
        with self.assertRaisesRegex(ValueError, 'timing approved before'):
            b.load_contract(None, self.root)


class FixedContractTest(unittest.TestCase):
    def test_the_committed_contract(self):
        c, _, eid = b.load_contract(None, ROOT)
        self.assertEqual(list(c['approved']), list(b.STAGES))
        self.assertEqual({k: v for k, v in c.items() if k not in ('sources', 'approved')},
                         {k: v for k, v in b.contract_body(ROOT).items() if k not in ('sources', 'approved')})
        self.assertEqual([s['path'] for s in c['sources']], list(b.SOURCES))
        self.assertEqual(eid, 'd18016933e9816d03982090dfa6cfc7d167089e815815105af043e5cb7df948d')


# --- the desktop blocks ----------------------------------------------------------------------------------------------

B_STUB = r'''import json, os, sys
from pathlib import Path
D = 'docs/h7r/budget/TB-V3-BUDGET-1'
cmd = sys.argv[1]
arg = lambda n: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else None
if cmd == 'verify':
    s = arg('--stage')
    r = json.loads(Path(D, s + '.json').read_text())
    if os.environ.get('B_BAD_VERIFY') == s:
        print('STOP: ValueError: not what its games give'); sys.exit(1)
    print(json.dumps({'stage': s, 'log_name': r['log_name'], 'exit_code': 0, 'stage_files': [D + '/' + s + '.json']}))
    sys.exit(0)
if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
    print('STOP: CUBLAS'); sys.exit(7)
if os.environ.get('B_FORBID') == cmd:
    print('STOP: must not run again'); sys.exit(9)
if os.environ.get('B_FAIL') == cmd:
    print('STOP: ValueError: stand-in failure'); sys.exit(1)
Path(D, cmd + '.json').write_text(json.dumps({'format': 'h7r-budget-' + cmd + '-v1', 'passed': True,
                                              'log_name': arg('--log-name')}))
print(cmd); sys.exit(0)
'''


class BlockTest(unittest.TestCase):
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
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', 'scripts/h7r_attempts.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        (w / 'scripts/h7r_budget.py').write_text(B_STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        (w / b.D).mkdir(parents=True)
        (w / b.CONTRACT).write_text('{}')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/budget/TB-V3-BUDGET-1-draft.md').read_text()
        blocks = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 3)
        self.blocks = dict(zip(('b-P', 'b-T', 'b-C'), (x.replace('~/renju-h6', w.as_posix()) for x in blocks)))

    def bash(self, name, **env):
        import os
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True,
                           env={**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}, start_new_session=True)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def subjects(self):
        return git(self.work, 'log', '--format=%s').splitlines()

    def test_the_three_stages_in_order(self):
        code, out = self.bash('b-T')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        for name in ('b-P', 'b-T', 'b-C'):
            code, out = self.bash(name, CUBLAS_WORKSPACE_CONFIG=':4096:8')  # the block unsets it
            self.assertEqual(code, 0, f'{name}: {out}')
            self.assertIn('__RECORDED__ 0', out)
        self.assertEqual(self.subjects()[:3], [f'H7-R BUDGET: {n} exit 0 (records only)' for n in
                                               ('confirm', 'timing', 'prepare')])
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        code, out = self.bash('b-C')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_a_failure_commits_nothing_and_resumes_at_its_head(self):
        self.assertEqual(self.bash('b-P')[0], 0)
        code, out = self.bash('b-T', B_FAIL='timing')
        self.assertEqual(code, 1, out)
        self.assertIn('__FAILED__ 1', out)
        head = git(self.work, 'rev-parse', 'HEAD')
        self.assertEqual(self.subjects()[0], 'H7-R BUDGET: prepare exit 0 (records only)')
        self.assertEqual((self.work / b.RUNS / 'timing.HEAD').read_text().strip(), head)
        code, out = self.bash('b-T')
        self.assertEqual(code, 0, out)
        self.assertIn('resuming timing at ' + head, out)
        self.assertEqual(len([f for f in self.files(f'{b.D}/stage-logs') if '/timing-' in f and f.endswith('.log')]),
                         2)  # the failed attempt's log goes with the record
        self.assertFalse((self.work / b.RUNS / 'timing.HEAD').exists())

    def test_a_moved_head_stops_the_resume(self):
        self.assertEqual(self.bash('b-P')[0], 0)
        self.assertEqual(self.bash('b-T', B_FAIL='timing')[0], 1)
        (self.work / 'note.txt').write_text('x')
        git(self.work, 'add', 'note.txt')
        git(self.work, 'commit', '-qm', 'note')
        code, out = self.bash('b-T')
        self.assertEqual(code, 1, out)
        self.assertIn('STOP: HEAD moved since timing started', out)

    def test_a_record_that_does_not_verify_is_not_committed(self):
        self.assertEqual(self.bash('b-P')[0], 0)
        code, out = self.bash('b-T', B_BAD_VERIFY='timing')
        self.assertEqual(code, 1, out)
        self.assertIn('the record does not check against its files; nothing committed', out)
        self.assertEqual(self.subjects()[0], 'H7-R BUDGET: prepare exit 0 (records only)')
        self.assertEqual(git(self.work, 'diff', '--cached', '--name-only'), '')

    def test_an_uncommitted_result_is_recorded_without_a_new_run(self):
        self.assertEqual(self.bash('b-P')[0], 0)
        L = self.work / b.RUNS / 'logs'
        name = 'timing-AbC123xy.log'
        (L / name).write_text('timing' + chr(10))  # killed before its exit line
        (self.work / b.D / 'timing.json').write_text(json.dumps({'format': 'h7r-budget-timing-v1', 'log_name': name}))
        code, out = self.bash('b-T', B_FORBID='timing')
        self.assertEqual(code, 0, out)
        self.assertIn('recording it (no new run)', out)
        self.assertIn('Recovered exit code', (self.work / b.D / 'stage-logs' / name).read_text())
