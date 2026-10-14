"""rt30.supervisor: the answer comes by the watchdog, from the engine or its best-so-far (docs/web-ser5.md §3.6)."""
from pathlib import Path
from time import monotonic, sleep
import unittest

from rt30.supervisor import AgentSpec, EngineSupervisor, fallback_move

ROOT = Path(__file__).resolve().parents[1]
PATHS = [str(ROOT / 'src'), str(ROOT / 'tests')]
HISTORY = [(7, 7), (7, 8), (8, 8)]


def supervisor(behaviour, *, deadline=1.0, watchdog=2.0, spare=True, **kwargs):
    sup = EngineSupervisor(AgentSpec('rt30_fake_agents:make', {'behaviour': behaviour, **kwargs}),
                           engine_deadline=deadline, watchdog=watchdog, spare=spare, src_paths=PATHS,
                           log=lambda *a, **k: None)
    return sup


def wait_spare(sup, timeout=30.0):
    end = monotonic() + timeout
    while not sup.spare_ready() and monotonic() < end:
        sleep(0.05)
    return sup.spare_ready()


class SupervisorTest(unittest.TestCase):
    def run_select(self, sup, received_at=None, game_key='g1'):
        return sup.select(game_key=game_key, seed=5, history=HISTORY, received_at=received_at)

    def test_normal_move_carries_times_and_the_absolute_deadline(self):
        sup = supervisor('quick')
        self.addCleanup(sup.close)
        received = monotonic() - 0.3  # the request arrived 0.3 s ago
        result = self.run_select(sup, received)
        self.assertEqual(result.source, 'engine')
        self.assertEqual(result.move, (0, 0))
        self.assertGreaterEqual(result.queue_seconds, 0.3)
        self.assertAlmostEqual(result.diagnostics['remaining'], 1.0 - result.queue_seconds, delta=0.2)
        self.assertGreaterEqual(result.wall_seconds, result.queue_seconds + result.engine_seconds - 1e-3)
        self.assertEqual(result.best, [('emergency', (0, 0))])

    def test_agent_per_game_keeps_its_state(self):
        sup = supervisor('quick')
        self.addCleanup(sup.close)
        self.assertEqual(self.run_select(sup).diagnostics['calls'], 1)
        self.assertEqual(self.run_select(sup).diagnostics['calls'], 2)
        self.assertEqual(self.run_select(sup, game_key='g2').diagnostics['calls'], 1)

    def test_engine_deadline_answer_is_before_the_watchdog(self):
        sup = supervisor('deadline', deadline=1.0, watchdog=2.0)
        self.addCleanup(sup.close)
        result = self.run_select(sup)
        self.assertEqual(result.source, 'engine')
        self.assertLess(result.wall_seconds, 1.5)
        self.assertFalse(result.watchdog)

    def test_hung_worker_answers_best_so_far_at_the_watchdog_and_is_replaced(self):
        sup = supervisor('hang', deadline=0.5, watchdog=1.5)
        self.addCleanup(sup.close)
        self.assertTrue(wait_spare(sup))
        first_pid = sup._worker.pid
        result = self.run_select(sup)
        self.assertEqual((result.source, result.watchdog, result.worker_restarted), ('watchdog', True, True))
        self.assertEqual(result.move, (14, 14))  # the last progress move, not the emergency one
        self.assertLess(result.wall_seconds, 1.5 + 0.5)
        self.assertNotEqual(sup._worker.pid, first_pid)  # the warm spare took over
        # The next request is served again before its watchdog (the new worker also hangs here).
        again = self.run_select(sup)
        self.assertEqual(again.source, 'watchdog')
        self.assertEqual((sup.watchdogs, sup.restarts), (2, 2))

    def test_silent_hang_falls_back_to_the_rule_move(self):
        sup = supervisor('hang_silent', deadline=0.5, watchdog=1.0, spare=False)
        self.addCleanup(sup.close)
        result = self.run_select(sup)
        self.assertEqual(result.source, 'watchdog')  # the emergency progress came before the hang
        self.assertEqual(result.move, (0, 0))

    def test_worker_error_plays_best_so_far_and_keeps_the_worker(self):
        sup = supervisor('crash')
        self.addCleanup(sup.close)
        pid = sup._worker.pid
        result = self.run_select(sup)
        self.assertEqual(result.source, 'worker_error')
        self.assertIn('boom', result.error)
        self.assertEqual(result.move, (0, 0))
        self.assertEqual(sup._worker.pid, pid)

    def test_dead_worker_is_replaced(self):
        sup = supervisor('exit', deadline=0.5, watchdog=3.0)
        self.addCleanup(sup.close)
        self.assertTrue(wait_spare(sup))
        result = self.run_select(sup)
        self.assertIn(result.source, ('worker_error', 'watchdog'))
        self.assertTrue(result.worker_restarted)
        self.assertLess(result.wall_seconds, 3.5)

    def test_queue_wait_counts_against_the_same_deadline(self):
        sup = supervisor('slow', deadline=1.0, watchdog=1.5, seconds=0.0)
        self.addCleanup(sup.close)
        # A request that already waited 1.2 s: only 0.3 s are left before its watchdog.
        result = self.run_select(sup, received_at=monotonic() - 1.2)
        self.assertLess(result.wall_seconds, 1.5 + 0.3)

    def test_fallback_rule(self):
        # White to move, black has an open four: block (the lower point).
        self.assertEqual(fallback_move([(7, 7), (0, 0), (7, 8), (0, 1), (7, 9), (0, 2), (7, 10)]), (7, 6))
        # Black to move with its own four: win first.
        self.assertEqual(fallback_move([(7, 7), (0, 0), (7, 8), (0, 1), (7, 9), (0, 2), (7, 10), (14, 14)]), (7, 6))
        # White to move, white has four and black has four: white wins.
        self.assertEqual(fallback_move([(7, 7), (0, 0), (7, 8), (0, 1), (7, 9), (0, 2), (7, 10), (0, 3), (14, 14)]),
                         (0, 4))
        self.assertIn(fallback_move([(7, 7)]), [(6, 6), (6, 7), (6, 8), (7, 6), (7, 8), (8, 6), (8, 7), (8, 8)])

    def test_bad_times_rejected(self):
        with self.assertRaises(ValueError):
            EngineSupervisor(AgentSpec('rt30_fake_agents:make'), engine_deadline=30, watchdog=29)


if __name__ == '__main__':
    unittest.main()


class WebSessionTest(unittest.TestCase):
    """run_web_play with the "rt30" opponent: the move comes from the worker; the DB has the times."""

    def setUp(self):
        import json
        from tempfile import TemporaryDirectory

        import scripts.run_web_play as web
        from weblog.store import LogStore, open_readonly

        self.web, self.json, self.open_readonly = web, json, open_readonly
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.sup = supervisor('hang', deadline=0.5, watchdog=1.5)
        self.addCleanup(self.sup.close)
        web.register_rt30(self.sup, 'TB-RT30 test', {'engine': 'TB-RT30 test', 'deadline_seconds': 0.5})
        self.addCleanup(lambda: (web.VERSION_LABELS.pop(web.RT30_KEY, None), web._RT30.clear()))
        self.store = LogStore(self.dir / 'web.db', log=lambda *a, **k: None)
        self.addCleanup(self.store.close)

    def test_watchdog_move_is_played_and_logged(self):
        web = self.web
        session = web.PlaySession(log_root=self.dir / 'logs', store=self.store)
        state = session.reset(agent_key=web.RT30_KEY, human_color=web.WHITE, seed=3, received_at=monotonic())
        self.assertEqual(state['move_count'], 1)
        state = session.play_human(7, 8, received_at=monotonic())
        self.assertEqual(state['move_count'], 3)
        self.assertEqual(session.move_records[-1]['actor'], 'TB-RT30 test')
        self.assertEqual((session.move_records[-1]['row0'], session.move_records[-1]['col0']), (14, 14))
        self.store.flush()
        conn = self.open_readonly(self.dir / 'web.db')
        self.addCleanup(conn.close)
        row = dict(conn.execute("SELECT * FROM moves WHERE actor = 'TB-RT30 test'").fetchone())
        rt = self.json.loads(row['rt_json'])
        self.assertEqual((rt['source'], rt['watchdog'], row['timed_out'], row['emergency_used']), ('watchdog', True, 1, 1))
        self.assertLess(row['wall_seconds'], 2.0)
        self.assertEqual(row['deadline_seconds'], 0.5)
        game = dict(conn.execute('SELECT engine FROM games').fetchone())
        self.assertEqual(game['engine'], 'TB-RT30 test')
