"""TB-RT30-v4 as the web's legacy opponent: its same-move gate, its status and its lazily started worker."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from time import monotonic, sleep
import unittest

import scripts.run_web_play as web
from rt30 import legacy
from scripts import check_rt30_v4_gate as gate

ROOT = Path(__file__).resolve().parents[1]


def replay(arms, keys, commit, deadline=None, manifest=None, move=(7, 8)):
    rows = []
    for key in keys:
        row = {'key': key}
        for arm in arms:
            row[arm] = {'move': list(move), 'seconds': 1.0, 'route': 'tree', 'phases': {},
                        'nodes': {'attack': 1, 'stage_vct': 0, 'root': 2, 'vct2': 3},
                        'checked': {'stage_vct': [], 'root': [[list(move), 'SAFE']], 'vct2': []},
                        'tree_move': list(move), 'rt': {'timeouts': [], 'attack_helper_seconds': 0.1}}
        rows.append(row)
    return {'format': 'rt30-replay-v1', 'engine': 'TB-Engine-v1', 'arms': list(arms), 'native_scan': True,
            'manifest': manifest, 'deadline': deadline, 'root_rescue': True, 'timing': {'final_margin': 1.5},
            'select': 'sample', 'sample': None, 'seed': 42, 'git_commit': commit,
            'machine': {'platform': 'Linux-SER5', 'python': '3.14.4'}, 'rows': rows}


class GateTest(unittest.TestCase):
    V3_LOCK = {'src/rt30/native/build.py': 'old', 'src/rt30/engine_v3.py': 'e'}
    V4_LOCK = {'src/rt30/native/build.py': 'new', 'src/rt30/engine_v3.py': 'e', 'src/rt30/tb_rt30_v4.py': 'v4'}

    @classmethod
    def blob(cls, commit, path):
        return {'v3-commit': cls.V3_LOCK, 'v4-commit': cls.V4_LOCK}.get(commit, {}).get(path)

    def run_gate(self, v4_eq=None, pairs=None):
        keys = ['a/0/1', 'a/0/2']
        record = replay(gate.EQUIVALENCE_ARMS, keys, 'v3-freeze')
        v4_eq = v4_eq or replay(gate.EQUIVALENCE_ARMS, keys, 'v4-commit')
        manifest = {'path': 'docs/rt30-eval/slow-stratified-v1.json', 'version': 'slow-stratified-v1', 'keys': 2}
        if pairs is None:
            pairs = [(replay(['rt3'], keys, 'v3-commit', manifest=manifest),
                      replay(['rt3'], keys, 'v4-commit', manifest=manifest))]
        return gate.evaluate(v4_eq, pairs, v3_record=record, v3_lock=self.V3_LOCK, v4_lock=self.V4_LOCK,
                             blob=self.blob)

    def test_pass(self):
        result = self.run_gate()
        self.assertEqual((result['verdict'], result['mismatched']), ('V4_SAME_MOVE_PASS', 0))
        self.assertEqual(set(result['parts']), {'equivalence', 'slow-stratified-v1'})

    def test_a_different_move_or_record_fails(self):
        keys = ['a/0/1', 'a/0/2']
        v4_eq = replay(gate.EQUIVALENCE_ARMS, keys, 'v4-commit')
        v4_eq['rows'][1]['rt3']['nodes']['vct2'] = 4
        result = self.run_gate(v4_eq=v4_eq)
        self.assertEqual(result['verdict'], 'V4_SAME_MOVE_FAIL')
        self.assertEqual(result['parts']['equivalence']['rt3']['mismatches'][0]['fields'], ['nodes'])
        manifest = {'version': 'critical-regressions-v1'}
        moved = [(replay(['rt3'], keys, 'v3-commit', manifest=manifest),
                  replay(['rt3'], keys, 'v4-commit', manifest=manifest, move=(6, 6)))]
        self.assertEqual(self.run_gate(pairs=moved)['parts']['critical-regressions-v1']['mismatches'][0]['fields'],
                         ['move', 'checked', 'tree_move'])

    def test_refusals(self):
        keys = ['a/0/1', 'a/0/2']
        manifest = {'version': 'slow-stratified-v1'}
        good = (replay(['rt3'], keys, 'v3-commit', manifest=manifest), replay(['rt3'], keys, 'v4-commit',
                                                                             manifest=manifest))
        bad = []
        swapped = copy.deepcopy(good)
        bad.append([(swapped[1], swapped[0])])                       # v3 run on the v4 commit
        timed = copy.deepcopy(good)
        timed[1]['deadline'] = 27.5
        bad.append([timed])                                           # a deadline puts time in decisions
        machine = copy.deepcopy(good)
        machine[1]['machine'] = {'platform': 'Windows', 'python': '3.13'}
        bad.append([machine])
        seed = copy.deepcopy(good)
        seed[1]['seed'] = 7
        bad.append([seed])
        bad.append([])                                                # no representative set at all
        for pairs in bad:
            with self.assertRaises(ValueError, msg=str(pairs)[:80]):
                self.run_gate(pairs=pairs)
        with self.assertRaises(ValueError):                           # the equivalence run is not v4
            self.run_gate(v4_eq=replay(gate.EQUIVALENCE_ARMS, keys, 'v3-commit'))

    def test_real_locks_differ_by_the_builder_only(self):
        v3, v4 = gate.lock_entries(gate.V3_LOCK), gate.lock_entries(gate.V4_LOCK)
        self.assertEqual({p for p in v3 if v3[p] != v4[p]}, {'src/rt30/native/build.py'})
        self.assertEqual(set(v4) - set(v3), {'src/rt30/tb_rt30_v4.py'})


class StatusTest(unittest.TestCase):
    def test_status_follows_the_gate_record(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v4_same_move.json'
            self.assertEqual(legacy.status(path)[0], 'candidate')  # no record
            record = {'format': legacy.FORMAT, 'verdict': 'V4_SAME_MOVE_PASS', 'v4_lock': legacy.lock_entries(),
                      'compared': 90}
            path.write_text(json.dumps(record), encoding='utf-8')
            self.assertEqual(legacy.status(path)[0], 'legacy-certified')
            path.write_text(json.dumps({**record, 'verdict': 'V4_SAME_MOVE_FAIL'}), encoding='utf-8')
            self.assertEqual(legacy.status(path)[0], 'candidate')
            other = dict(record['v4_lock'], **{'src/rt30/tb_rt30_v4.py': 'f' * 64})
            path.write_text(json.dumps({**record, 'v4_lock': other}), encoding='utf-8')
            self.assertEqual(legacy.status(path)[0], 'candidate')  # made on another v4

    def test_committed_status(self):
        status, reason = legacy.status()
        self.assertIn(status, ('candidate', 'legacy-certified'))
        if legacy.GATE_RECORD.is_file():  # once committed, the record must certify this checkout's v4
            self.assertEqual(status, 'legacy-certified', reason)


class FakeEngineSupervisor:
    made = 0

    def __init__(self):
        type(self).made += 1
        self.closed = False
        self.received = []
        self.restarts = self.watchdogs = 0

    def select(self, *, game_key, seed, history, received_at=None):
        self.received.append(received_at)
        return 'move'

    def spare_ready(self):
        return True

    def close(self):
        self.closed = True


class LazyTest(unittest.TestCase):
    def test_starts_on_first_use_and_closes_when_idle(self):
        made = []

        def make():
            sleep(0.2)  # a cold start
            made.append(FakeEngineSupervisor())
            return made[-1]

        lazy = legacy.LazySupervisor(make, idle_seconds=0.5, log=lambda *a, **k: None)
        self.addCleanup(lazy.close)
        self.assertFalse(lazy.loaded)
        self.assertEqual(lazy.stats()['loaded'], False)
        before = monotonic()
        self.assertEqual(lazy.select(game_key='g', seed=1, history=[], received_at=before), 'move')
        self.assertGreater(made[0].received[0], before)  # the deadline counts from the ready worker
        lazy.select(game_key='g', seed=1, history=[], received_at=5.0)
        self.assertEqual((len(made), made[0].received[1]), (1, 5.0))  # warm: as given
        self.assertTrue(lazy.loaded)
        deadline = monotonic() + 5
        while lazy.loaded and monotonic() < deadline:
            sleep(0.05)
        self.assertFalse(lazy.loaded)
        self.assertTrue(made[0].closed)
        lazy.select(game_key='g', seed=1, history=[])
        self.assertEqual((len(made), lazy.starts, lazy.closed_idle), (2, 2, 1))

    def test_not_closed_while_searching(self):
        gate_open = threading.Event()

        class Busy(FakeEngineSupervisor):
            def select(self, **kw):
                gate_open.wait(5)
                return 'move'

        lazy = legacy.LazySupervisor(Busy, idle_seconds=0.1, log=lambda *a, **k: None)
        self.addCleanup(lazy.close)
        thread = threading.Thread(target=lazy.select, kwargs={'game_key': 'g', 'seed': 1, 'history': []})
        thread.start()
        sleep(0.3)
        lazy._close_if_idle()
        self.assertTrue(lazy.loaded)
        gate_open.set()
        thread.join(5)


class EnginesOverviewTest(unittest.TestCase):
    def test_featured_order_and_overview(self):
        saved = dict(web._TBV3), dict(web._RT30), dict(web.VERSION_LABELS)

        def restore():
            for target, value in zip((web._TBV3, web._RT30, web.VERSION_LABELS), saved):
                target.clear()
                target.update(value)
        self.addCleanup(restore)
        web._TBV3.clear()
        web._RT30.clear()
        self.assertEqual(web.featured_agents(), [])
        lazy = legacy.LazySupervisor(FakeEngineSupervisor, log=lambda *a, **k: None)
        self.addCleanup(lazy.close)
        web.register_rt30([lazy], f'{legacy.LABEL} (실시간 27.5초)',
                          {'engine': 'TB-RT30-v4', 'engine_config_hash': 'ab' * 32, 'status': 'legacy-certified',
                           'status_reason': 'gate', 'deadline_seconds': 27.5})
        from tests.test_web_tbv3 import INFO
        web.register_tbv3([FakeSupervisorWithStats()], '최신 인증 모델 - TB-V3 Resolve / fast5', INFO, 30.0)
        self.assertEqual(web.featured_agents(), [web.TBV3_KEY, web.RT30_KEY])
        self.assertEqual(web.default_agent_key(), web.TBV3_KEY)  # the legacy engine is never the default
        overview = web.engines_overview()
        self.assertEqual((overview['default'], overview['available']['rt30']['status'],
                          overview['available']['tbv3']['status']), ('tbv3', 'legacy-certified', 'latest-certified'))
        self.assertEqual(overview['available']['rt30']['workers'][0]['loaded'], False)  # lazy: not started yet
        agent = web.create_agent(web.RT30_KEY, seed=1)
        self.assertIsInstance(agent, web.SupervisedAgent)
        self.assertNotIsInstance(agent, web.CertifiedSupervisedAgent)


class FakeSupervisorWithStats:
    restarts = watchdogs = 0

    class _worker:
        pid = 123

        @staticmethod
        def alive():
            return True

    def spare_ready(self):
        return True


if __name__ == '__main__':
    unittest.main()
