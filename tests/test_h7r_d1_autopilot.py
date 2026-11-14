"""scripts/h7r_d1_autopilot.py: the delegation's progression as code. Every step in order, the stops (H-dev label,
time gate FAIL, entry checks, deadline), the one D4 resume for a crash and the record-preserving block for a failure,
and the blocks extracted from the D1 draft."""
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock

import h7r_d1_autopilot as ap
import h7r_d1_d4 as dd
from tests.test_h7r_d1_d4 import as_fixed, write_records

ROOT = Path(__file__).resolve().parents[1]
NOW = ap.START + timedelta(hours=8)


def put(root, rel, body):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body) if not isinstance(body, str) else body)


class Sim:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / ap.CONTRACT).parent.mkdir(parents=True)
        self.c = as_fixed(json.loads((ROOT / ap.CONTRACT).read_text()))  # the delegation starts unapproved
        (self.root / ap.CONTRACT).write_text(json.dumps(self.c, indent=2, ensure_ascii=False) + '\n')
        write_records(root, self.c)

    def decide(self, now=NOW, alive=False):
        return ap.decide(self.root, now, alive=lambda: alive)

    def launch(self, step, kind='run', text=None, started=None):
        n = len(ap.ledger(self.root)) + 1
        out = f'{ap.LEDGER_DIR}/{n:02d}-{step}-{kind}.out'
        with open(self.root / ap.LEDGER_DIR / 'launches.jsonl' if (self.root / ap.LEDGER_DIR).exists() else
                  self._mk(), 'a') as f:
            f.write(json.dumps({'n': n, 'step': step, 'kind': kind, 'out': out,
                                'started': started or time.time() - 5}) + '\n')
        if text is not None:
            put(self.root, out, text)
        return out

    def _mk(self):
        (self.root / ap.LEDGER_DIR).mkdir(parents=True)
        return self.root / ap.LEDGER_DIR / 'launches.jsonl'

    def hdev(self, label='DEVELOPMENT_SIGNAL'):
        put(self.root, self.c['entry']['fit'], {'comparable': True, 'reading': {'category': 'WIDTH_EFFECT'}})
        put(self.root, ap.DONE['hdev'], {'hdev_reading': {'label': label}})

    def timing(self, passed=True, pair=('narrow31001', 'wide31001')):
        put(self.root, ap.DONE['timing'], {
            'result': {'passed': passed, 'mean_ratio': 1.02, 'p95_ratio': 1.3},
            'reference': {'models': {'value_checkpoint_sha256': pair[0].ljust(64, 'c')}},
            'candidate': {'models': {'value_checkpoint_sha256': pair[1].ljust(64, 'c')}}})

    def approve(self, mode):
        self.c['approved'][mode] = True
        (self.root / ap.CONTRACT).write_text(json.dumps(self.c))

    def summary(self, mode, reading, reproduced=True):
        put(self.root, ap.DONE[mode], {'identity': {'experiment_sha256': dd.experiment_sha256(self.c)},
                                       'reading': reading, 'actor_check': {'reproduced': reproduced}})


def patched():
    yes = mock.patch.object(ap, 'committed', return_value=True)
    yes_dd = mock.patch.object(dd, 'committed', return_value=True)
    return yes, yes_dd


class DecideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.sim = Sim(self.tmp.name)
        for p in patched():
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_the_steps_in_order(self):
        s = self.sim
        self.assertEqual(s.decide(), {'action': 'run', 'step': 'hdev', 'block': 'd1-full-3.sh', 'args': []})
        out = s.launch('hdev', text='working\n')
        self.assertEqual(s.decide(alive=True), {'action': 'wait', 'step': 'hdev', 'out': out})
        out2 = s.decide(alive=False)  # the session died without BLOCK EXIT: a failure, its records preserved
        self.assertEqual((out2['action'], out2['block']), ('recover', 'd1-full-3F.sh'))
        put(s.root, out, 'x\n__RECORDED__ 0\nBLOCK EXIT 0\n')
        s.hdev()
        self.assertEqual(s.decide()['block'], 'd1-v2-1.sh')
        s.launch('timing', text='__RECORDED__ 0\nBLOCK EXIT 0\n')
        s.timing()
        self.assertEqual(s.decide()['block'], 'd1-d4-1.sh')
        s.launch('eligible', text='__RECORDED__ 0\nBLOCK EXIT 0\n')
        put(s.root, s.c['output']['sample'], {'games': []})
        self.assertEqual(s.decide(), {'action': 'approve', 'step': 'smoke', 'mode': 'smoke'})
        s.approve('smoke')
        self.assertEqual(s.decide(), {'action': 'run', 'step': 'smoke', 'block': 'd1-d4-run.sh', 'args': ['smoke']})
        s.launch('smoke', text='__RECORDED__ smoke SMOKE\nBLOCK EXIT 0\n')
        s.summary('smoke', 'SMOKE')
        self.assertEqual(s.decide(), {'action': 'approve', 'step': 'full', 'mode': 'full'})
        self.assertEqual(s.decide(now=ap.DEADLINE - timedelta(hours=5))['action'], 'stop')  # too little time left
        s.approve('full')
        self.assertEqual(s.decide()['args'], ['full'])
        s.launch('full', text='__RECORDED__ full ALL_THREE_BETTER\nBLOCK EXIT 0\n')
        s.summary('full', 'ALL_THREE_BETTER')
        out = s.decide()
        self.assertEqual((out['action'], out['final'], out['reading']), ('stop', True, 'ALL_THREE_BETTER'))
        self.assertIn('P5', out['reason'])

    def test_readings_that_end_the_delegation(self):
        s = self.sim
        s.hdev('NOT_ESTABLISHED')
        out = s.decide()
        self.assertEqual((out['action'], out['final']), ('stop', True))
        self.assertIn('NOT_ESTABLISHED', out['reason'])
        s.hdev()
        s.timing(passed=False)
        self.assertIn('time gate FAIL', s.decide()['reason'])
        s.timing(pair=('narrow31002', 'wide31001'))
        put(s.root, s.c['output']['sample'], {'games': []})
        out = s.decide()
        self.assertEqual(out['action'], 'stop')
        self.assertIn('not measured on the linked pair', out['reason'])
        s.timing()
        s.approve('smoke')
        s.summary('smoke', 'SMOKE', reproduced=False)
        self.assertIn('not a reproduced smoke', s.decide()['reason'])
        s.summary('smoke', 'SMOKE')
        s.approve('full')
        s.summary('full', 'NOT_ESTABLISHED')
        out = s.decide()
        self.assertEqual((out['reading'], out['final']), ('NOT_ESTABLISHED', True))
        self.assertIn('delegation ends', out['reason'])

    def test_failures_retry_and_recovery(self):
        s = self.sim
        s.hdev()
        s.timing()
        put(s.root, s.c['output']['sample'], {'games': []})
        s.approve('smoke')
        started = time.time() - 5
        s.launch('smoke', text='__NOT_RECORDED__ 1: kept\nBLOCK EXIT 1\n', started=started)
        put(s.root, f'{ap.D4_LOGS}/smoke-1.log', 'Traceback\nRuntimeError: CUDA error\n__EXIT__ 1\n')
        out = s.decide()
        self.assertEqual((out['action'], out.get('retry')), ('run', True))  # a crash: one resume
        s.launch('smoke', text='__NOT_RECORDED__ 1: kept\nBLOCK EXIT 1\n')
        out = s.decide()
        self.assertEqual((out['action'], out['block'], out['args']), ('recover', 'd1-d4-F.sh', ['smoke']))
        s.launch('smoke', kind='recover', text='__EXIT__ 0\nBLOCK EXIT 0\n')
        out = s.decide()
        self.assertEqual((out['action'], out['final']), ('stop', False))
        # a tool refusal is never retried
        os.remove(s.root / ap.LEDGER_DIR / 'launches.jsonl')
        put(s.root, f'{ap.D4_LOGS}/smoke-1.log', 'STOP: Rapfi identity differs\n__EXIT__ 1\n')
        s.launch('smoke', text='__NOT_RECORDED__ 1: kept\nBLOCK EXIT 1\n', started=started)
        self.assertEqual(s.decide()['action'], 'recover')

    def test_recorded_failures_stop_and_unrecorded_hdev_is_preserved(self):
        s = self.sim
        s.launch('hdev', text='STOP: x\n__RECORDED__ 1\nBLOCK EXIT 1\n')
        out = s.decide()
        self.assertEqual(out['action'], 'stop')
        self.assertIn('records committed', out['reason'])
        os.remove(s.root / ap.LEDGER_DIR / 'launches.jsonl')
        s.launch('hdev', text='Killed\nBLOCK EXIT 137\n')
        self.assertEqual(s.decide()['block'], 'd1-full-3F.sh')
        s.launch('timing')
        s.hdev()
        os.remove(s.root / ap.LEDGER_DIR / 'launches.jsonl')
        s.launch('timing', text='__RECORDED__ 1\nBLOCK EXIT 1\n')  # no timing record: STOP of the tool
        self.assertEqual(s.decide()['action'], 'stop')

    def test_an_authorized_retry_runs_once(self):
        s = self.sim
        s.hdev()
        s.launch('hdev', text='__RECORDED__ 0\nBLOCK EXIT 0\n')
        s.launch('timing', text='STOP: ValueError: x\nBLOCK EXIT 1\n')
        self.assertEqual(s.decide()['action'], 'stop')
        body = {'format': 'h7r-autopilot-retries-v1',
                'retries': [{'step': 'timing', 'failed_launch': 2, 'cause': 'tool defect', 'fix': 'abc1234'}]}
        put(s.root, ap.RETRIES, body)
        out = s.decide()
        self.assertEqual((out['action'], out['block'], out['retry']), ('run', 'd1-v2-1.sh', True))
        self.assertIn('tool defect', out['reason'])
        s.launch('timing', text='STOP: again\nBLOCK EXIT 1\n')
        self.assertEqual(s.decide()['action'], 'stop')  # the authorization names launch 2 only
        put(s.root, ap.RETRIES, {**body, 'retries': [{**body['retries'][0], 'step': 'full'}]})
        with self.assertRaisesRegex(ValueError, 'not an authorizable retry'):
            s.decide()

    def test_amendment_a2_reruns_eligible_and_reapproves_smoke(self):
        """The real ledger after the first smoke attempt: eligible done under the old sample path (launch 4), smoke
        refused at preflight (launch 5) and preserved (launch 6); A2 resets the approvals and moves the sample."""
        s = self.sim
        s.hdev()
        s.timing()
        for step, text in (('hdev', '__RECORDED__ 0\nBLOCK EXIT 0\n'), ('timing', 'BLOCK EXIT 1\n'),
                           ('timing', '__RECORDED__ 0\nBLOCK EXIT 0\n'), ('eligible', '__RECORDED__ 0\nBLOCK EXIT 0\n'),
                           ('smoke', 'STOP: x\n__NOT_RECORDED__ 1\nBLOCK EXIT 1\n')):
            s.launch(step, text=text)
        s.launch('smoke', kind='recover', text='__EXIT__ 0\nBLOCK EXIT 0\n')
        put(s.root, f'{ap.D}/d4/eligible.json', {'games': []})  # the superseded sample: not the contract's
        self.assertEqual(s.decide()['action'], 'stop')
        put(s.root, ap.RETRIES, {'format': 'h7r-autopilot-retries-v1', 'retries': [
            {'step': 'timing', 'failed_launch': 2, 'cause': 'a', 'fix': 'b'},
            {'step': 'eligible', 'failed_launch': 4, 'cause': 'superseded sample', 'fix': 'A2'},
            {'step': 'smoke', 'failed_launch': 5, 'cause': 'manifest binding', 'fix': 'A2'}]})
        out = s.decide()
        self.assertEqual((out['action'], out['block'], out['retry']), ('run', 'd1-d4-1.sh', True))
        s.launch('eligible', text='__RECORDED__ 0\nBLOCK EXIT 0\n')
        put(s.root, s.c['output']['sample'], {'games': [1]})
        self.assertIn('other games / rows', s.decide()['reason'])  # A2 may change only the binding
        put(s.root, s.c['output']['sample'], {'games': [], 'contract_sha256': 'new'})
        self.assertEqual(s.decide(), {'action': 'approve', 'step': 'smoke', 'mode': 'smoke'})
        s.approve('smoke')
        out = s.decide()
        self.assertEqual((out['action'], out['args'], out['retry']), ('run', ['smoke'], True))
        s.launch('smoke', text='__NOT_RECORDED__ 1\nBLOCK EXIT 1\n')
        self.assertEqual(s.decide()['action'], 'recover')  # the authorization named launch 5 only

    def test_the_deadline(self):
        out = self.sim.decide(now=ap.DEADLINE)
        self.assertEqual((out['action'], out['final']), ('stop', True))
        self.assertEqual(ap.DEADLINE.isoformat(), '2026-11-10T15:24:27+09:00')


    def test_the_limit_applies_to_every_launch_and_to_a_running_block(self):
        s = self.sim
        s.hdev()
        s.timing()
        put(s.root, s.c['output']['sample'], {'games': []})
        s.approve('smoke')
        s.summary('smoke', 'SMOKE')
        s.approve('full')  # approved early: the time check still applies at launch
        out = s.decide(now=ap.DEADLINE - timedelta(minutes=1))
        self.assertEqual((out['action'], out['final']), ('stop', True))
        self.assertIn('full needs 6.0 h', out['reason'])
        self.assertEqual(s.decide(now=ap.DEADLINE - timedelta(hours=7))['action'], 'run')
        run = s.launch('full', text='[d1-d4-replay] 10/396\n')
        self.assertEqual(s.decide(now=ap.DEADLINE - timedelta(hours=1), alive=True)['action'], 'wait')
        self.assertEqual(s.decide(now=ap.DEADLINE, alive=True)['action'], 'interrupt')
        with open(s.root / ap.LEDGER_DIR / 'launches.jsonl', 'a') as f:
            f.write(json.dumps({'n': 1, 'mark': 'interrupted'}) + '\n')
        self.assertEqual(s.decide(now=ap.DEADLINE, alive=True)['action'], 'wait')  # interrupted once, not twice
        put(s.root, f'{ap.D4_LOGS}/full-1.log', 'KeyboardInterrupt\n')  # no tool refusal, yet never resumed
        out = s.decide(now=ap.DEADLINE + timedelta(minutes=1), alive=False)  # ended without BLOCK EXIT
        self.assertEqual((out['action'], out['block'], out['args']), ('recover', 'd1-d4-F.sh', ['full']))
        s.launch('full', kind='recover', text='__EXIT__ 0\nBLOCK EXIT 0\n')
        out = s.decide(now=ap.DEADLINE + timedelta(minutes=2))
        self.assertEqual((out['action'], out['final']), ('stop', True))
        self.assertIn('limit', out['reason'])
        self.assertTrue(run)

    def test_a_failed_preservation_is_not_reported_as_preserved(self):
        s = self.sim
        s.launch('hdev', text='Killed\nBLOCK EXIT 137\n')
        self.assertEqual(s.decide()['block'], 'd1-full-3F.sh')
        s.launch('hdev', kind='recover', text='STOP: a runner is still alive\nBLOCK EXIT 1\n')
        out = s.decide()
        self.assertEqual(out['action'], 'stop')
        self.assertIn('record preservation failed', out['reason'])

    def test_a_resume_needs_time_too(self):
        s = self.sim
        s.hdev()
        s.timing()
        put(s.root, s.c['output']['sample'], {'games': []})
        s.approve('smoke')
        started = time.time() - 5
        s.launch('smoke', text='__NOT_RECORDED__ 1\nBLOCK EXIT 1\n', started=started)
        put(s.root, f'{ap.D4_LOGS}/smoke-1.log', 'RuntimeError: CUDA\n')
        self.assertTrue(s.decide()['retry'])
        out = s.decide(now=ap.DEADLINE - timedelta(minutes=10))  # no time for a resume: preserve the records
        self.assertEqual(out['action'], 'recover')


class BlocksTest(unittest.TestCase):
    def test_blocks_come_from_the_draft(self):
        blocks = ap.extract_blocks(ROOT)
        self.assertEqual(sorted(blocks), sorted({'d1-full-3.sh', 'd1-full-3R.sh', 'd1-full-3F.sh', 'd1-v2-1.sh',
                                                 'd1-d4-1.sh', 'd1-d4-run.sh', 'd1-d4-F.sh'}))
        self.assertIn('h7r_d1_full.py evaluate', blocks['d1-full-3.sh'])
        self.assertIn('h7r_factory_v2_timing.py', blocks['d1-v2-1.sh'])
        self.assertIn('h7r_d1_d4.py prepare', blocks['d1-d4-1.sh'])
        self.assertIn('h7r_d1_d4.py run', blocks['d1-d4-run.sh'])
        for step, (block, _) in {**ap.BLOCKS, **ap.RECOVER}.items():
            self.assertIn(block, blocks, step)


if __name__ == '__main__':
    unittest.main()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


@unittest.skipUnless(shutil.which('tmux') and shutil.which('git'), 'tmux and git needed')
class GitTmuxIntegrationTest(unittest.TestCase):
    """The driver's commands against a real temporary git remote and a real tmux session (blocks replaced by short
    scripts); the clock is pinned before the limit."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.origin, self.work, self.other = base / 'origin.git', base / 'work', base / 'other'
        subprocess.run(['git', 'init', '-q', '--bare', str(self.origin)], check=True)
        subprocess.run(['git', 'init', '-q', '-b', 'feat/h7-c1', str(self.work)], check=True)
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(self.work, 'config', k, v)
        sim = Sim(self.work)
        put(self.work, '.gitignore', 'runs/\n')
        git(self.work, 'add', '-A')
        git(self.work, 'commit', '-qm', 'init')
        git(self.work, 'remote', 'add', 'origin', str(self.origin))
        git(self.work, 'push', '-q', 'origin', 'feat/h7-c1')
        subprocess.run(['git', 'clone', '-q', '-b', 'feat/h7-c1', str(self.origin), str(self.other)], check=True,
                       capture_output=True)
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(self.other, 'config', k, v)
        self.c = sim.c
        self.home = base / 'home'
        self.home.mkdir()
        self.session = f'd1test{os.getpid()}'
        for p in (mock.patch.object(ap, 'TMUX', self.session), mock.patch.object(ap, 'now_kst', return_value=NOW)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: subprocess.run(['tmux', 'kill-session', '-t', self.session], capture_output=True))

    def args(self, **kw):
        import argparse
        return argparse.Namespace(root=str(self.work), home=str(self.home), minutes=0.5, **kw)

    def remote_commit(self, name='x.txt'):
        put(self.other, name, 'new')
        git(self.other, 'add', name)
        git(self.other, 'commit', '-qm', 'remote change')
        git(self.other, 'push', '-q', 'origin', 'feat/h7-c1')

    def blocks(self, hdev_script):
        names = ('d1-full-3.sh', 'd1-full-3R.sh', 'd1-full-3F.sh', 'd1-v2-1.sh', 'd1-d4-1.sh', 'd1-d4-run.sh',
                 'd1-d4-F.sh')
        return mock.patch.object(ap, 'extract_blocks', return_value={n: hdev_script if n == 'd1-full-3.sh' else
                                                                     'exit 9\n' for n in names})

    def test_no_pull_while_a_block_runs_and_a_real_launch(self):
        head = git(self.work, 'rev-parse', 'HEAD')
        with self.blocks('touch "$PWD/started"; sleep 4; echo __RECORDED__ 0\n'):
            launched = ap.cmd_launch(self.args())
        self.assertEqual((launched['step'], launched['action']), ('hdev', 'launched'))
        self.remote_commit()
        for _ in range(50):
            if (self.work / 'started').exists():
                break
            time.sleep(0.1)
        out = ap.cmd_next(self.args())
        self.assertEqual((out['action'], out['pulled']), ('wait', False))
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)  # the running block's HEAD did not move
        self.assertEqual(ap.cmd_wait(self.args()), 0)
        self.assertEqual(ap.ledger(self.work)[-1]['exit'], 0)
        out = ap.cmd_next(self.args())  # finished: now it pulls
        self.assertTrue(out['pulled'])
        self.assertNotEqual(git(self.work, 'rev-parse', 'HEAD'), head)

    def test_staged_records_allow_only_recovery(self):
        ap.append_ledger(self.work, {'n': 1, 'step': 'hdev', 'kind': 'run', 'block': 'd1-full-3.sh', 'args': [],
                                     'out': f'{ap.LEDGER_DIR}/01-hdev-run.out', 'started': time.time()})
        put(self.work, f'{ap.LEDGER_DIR}/01-hdev-run.out', 'copy-logs\nBLOCK EXIT 1\n')
        put(self.work, f'{ap.D}/full/evaluate-logs/evaluate-1.log', 'log')
        git(self.work, 'add', f'{ap.D}/full/evaluate-logs/evaluate-1.log')
        self.remote_commit()
        head = git(self.work, 'rev-parse', 'HEAD')
        out = ap.cmd_next(self.args())
        self.assertEqual((out['action'], out['block'], out['tree'], out['pulled']),
                         ('recover', 'd1-full-3F.sh', 'records', False))
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)
        put(self.work, 'scripts_x.py', 'x')
        git(self.work, 'add', 'scripts_x.py')
        with self.assertRaisesRegex(ValueError, 'not staged records'):
            ap.cmd_next(self.args())
        git(self.work, 'reset', '-q', '--', 'scripts_x.py')
        put(self.work, ap.CONTRACT, self.c | {'purpose': 'edited'})  # a modified tracked file is never allowed
        with self.assertRaisesRegex(ValueError, 'not staged records'):
            ap.cmd_next(self.args())

    def test_a_tmux_start_failure_is_not_an_attempt(self):
        real = subprocess.run

        def fail_tmux(cmd, *a, **k):
            if cmd[:2] == ['tmux', 'new-session']:
                raise subprocess.CalledProcessError(1, cmd, stderr='no server')
            return real(cmd, *a, **k)
        with self.blocks('echo __RECORDED__ 0\n'), mock.patch.object(ap.subprocess, 'run', side_effect=fail_tmux):
            with self.assertRaisesRegex(ValueError, 'start_failed'):
                ap.cmd_launch(self.args())
        self.assertEqual(ap.ledger(self.work), [])
        self.assertEqual([e['started_ok'] for e in ap.ledger(self.work, all_entries=True)], [False])
        out = ap.cmd_next(self.args())
        self.assertEqual((out['action'], out['step']), ('run', 'hdev'))  # not stuck on the failed start
        with self.blocks('echo __RECORDED__ 0\n'):
            launched = ap.cmd_launch(self.args())
        self.assertEqual(launched['n'], 2)
        ap.cmd_wait(self.args())
        self.assertEqual([e['exit'] for e in ap.ledger(self.work)], [0])
