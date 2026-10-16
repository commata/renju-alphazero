"""TB-RT30-v3 in the web engine worker: the V8-B helper beside the worker (rt30.web_slot, docs/web-ser5.md §9.11)."""
import os
from pathlib import Path
from time import monotonic
import unittest

from rt30 import fast3_windows
from rt30.supervisor import AgentSpec, EngineSupervisor

from test_rt30_engine import POSITIONS
from test_rt30_fast_windows import recorded_position

ROOT = Path(__file__).resolve().parents[1]
PATHS = [str(ROOT / 'src'), str(ROOT / 'tests')]


def v3_supervisor(*, attack_helper=True, watchdog=40.0, deadline=27.5):
    return EngineSupervisor(AgentSpec('rt30.supervisor:make_tb_rt30_v3_agent', {'uniform_prior': True}),
                            engine_deadline=deadline, watchdog=watchdog, spare=False, src_paths=PATHS,
                            attack_helper=attack_helper, preload=True, log=lambda *a, **k: None)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@unittest.skipUnless(fast3_windows.available(), 'native scan not built (python -m rt30.native.build)')
class WebSlotTest(unittest.TestCase):
    def test_v3_searches_with_its_helper_in_the_daemonic_worker(self):
        sup = v3_supervisor()
        self.addCleanup(sup.close)
        game = recorded_position(*POSITIONS['own_vct'][0])
        result = sup.select(game_key='g', seed=7, history=list(game.history), received_at=monotonic())
        self.assertEqual(result.source, 'engine')
        self.assertTrue(result.rt['attack_parallel'])
        self.assertEqual(result.rt['attack_helper_failed'], '')
        self.assertEqual(result.diagnostics['v8_route'], 'own_vct')  # the helper's V8-B win
        self.assertLess(result.wall_seconds, 27.5)

    def test_without_the_helper_v3_refuses_to_start(self):
        with self.assertRaisesRegex(RuntimeError, 'V8-B helper'):
            v3_supervisor(attack_helper=False)

    def test_replacing_the_worker_replaces_its_helper(self):
        sup = v3_supervisor()
        self.addCleanup(sup.close)
        old = sup._worker
        helper_pid = old.helper.pid
        sup._replace_worker()
        self.assertTrue(sup._worker.wait_ready(120))
        self.assertIsNot(sup._worker, old)
        end = monotonic() + 5
        while alive(helper_pid) and monotonic() < end:
            pass
        self.assertFalse(alive(helper_pid))
        self.assertTrue(sup._worker.helper.is_alive())

    def test_a_dead_helper_is_replaced_before_the_next_search(self):
        sup = v3_supervisor()
        self.addCleanup(sup.close)
        sup._worker.helper.kill()
        sup._worker.helper.join(5)
        game = recorded_position(*POSITIONS['own_vct'][0])
        result = sup.select(game_key='g', seed=7, history=list(game.history), received_at=monotonic())
        self.assertEqual(sup.restarts, 1)
        self.assertEqual(result.rt['attack_helper_failed'], '')
        self.assertTrue(result.rt['attack_parallel'])


if __name__ == '__main__':
    unittest.main()
