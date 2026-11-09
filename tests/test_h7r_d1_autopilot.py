"""scripts/h7r_d1_autopilot.py: the delegation's progression as code. Every step in order, the stops (H-dev label,
time gate FAIL, entry checks, deadline), the one D4 resume for a crash and the record-preserving block for a failure,
and the blocks extracted from the D1 draft."""
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock

import h7r_d1_autopilot as ap
import h7r_d1_d4 as dd
from tests.test_h7r_d1_d4 import write_records

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
        shutil.copy(ROOT / ap.CONTRACT, self.root / ap.CONTRACT)
        self.c = json.loads((self.root / ap.CONTRACT).read_text())
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
        self.assertEqual(s.decide(alive=False)['action'], 'stop')  # the session died without BLOCK EXIT
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
