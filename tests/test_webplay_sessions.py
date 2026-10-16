"""Multi-session web play (docs/web-ser5.md §8): sessions, the AI job queue, the async session flow, HTTP."""
import http.client
import json
import sqlite3
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import scripts.run_web_play as web
from renju import BLACK, WHITE
from webplay.jobs import AIJob, AIJobQueue, QueueFull
from webplay.sessions import (COOKIE, SessionLimit, SessionManager, cookie_header, session_hash,
                              session_id_from_cookie)


class GateAgent:
    """Searches until ``gate`` opens (or ``fail``); plays the smallest legal move."""
    gate = threading.Event()
    fail = False
    calls = 0

    def __init__(self, seed):
        self.name, self.diagnostics = 'gate', {}

    def select_move(self, game):
        type(self).calls += 1
        if not type(self).gate.wait(10):
            raise TimeoutError('gate never opened')
        if type(self).fail:
            raise RuntimeError('boom')
        return min(game.legal_moves())


web.register_agent('gate', 'Gate (test)', GateAgent)


def wait_for(predicate, timeout=10.0):
    end = monotonic() + timeout
    while monotonic() < end:
        if predicate():
            return True
        sleep(0.01)
    return False


class _Gate(unittest.TestCase):
    def setUp(self):
        GateAgent.gate = threading.Event()
        GateAgent.fail = False
        GateAgent.calls = 0
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: GateAgent.gate.set())  # never leave a slot thread waiting


class JobQueueTest(unittest.TestCase):
    def queue(self, **kwargs):
        jobs = AIJobQueue(**kwargs)
        self.addCleanup(jobs.close)
        return jobs

    def test_room_is_reserved_first_and_given_back(self):
        jobs = self.queue(max_queue=2)
        hold = threading.Event()
        jobs.submit(jobs.reserve(), AIJob(run=lambda slot: hold.wait(5)))  # occupies the only slot
        self.assertTrue(wait_for(lambda: jobs.stats()['running'] == 1))
        a, b = jobs.reserve(), jobs.reserve()
        with self.assertRaises(QueueFull):
            jobs.reserve()
        jobs.release(a)
        jobs.release(a)  # twice is harmless
        c = jobs.reserve()
        jobs.release(b), jobs.release(c)
        hold.set()

    def test_fifo_cancel_and_position(self):
        jobs = self.queue(max_queue=4)
        hold, order = threading.Event(), []
        jobs.submit(jobs.reserve(), AIJob(run=lambda slot: hold.wait(5)))
        self.assertTrue(wait_for(lambda: jobs.stats()['running'] == 1))
        queued = [AIJob(run=lambda slot, n=n: order.append(n)) for n in range(3)]
        for job in queued:
            jobs.submit(jobs.reserve(), job)
        self.assertEqual([jobs.position(j) for j in queued], [1, 2, 3])
        self.assertTrue(jobs.cancel(queued[1]))
        self.assertFalse(jobs.cancel(queued[1]))
        hold.set()
        self.assertTrue(wait_for(lambda: jobs.stats()['done'] == 3))
        self.assertEqual(order, [0, 2])

    def test_slot_affinity_and_a_failing_job(self):
        jobs = self.queue(max_queue=8, slots=2, log=lambda *a, **k: None)
        seen = {}
        jobs.submit(jobs.reserve(), AIJob(run=lambda slot: (_ for _ in ()).throw(RuntimeError('x'))))
        for n in range(4):
            jobs.submit(jobs.reserve(), AIJob(run=lambda slot, n=n: seen.__setitem__(n, slot), slot=1))
        self.assertTrue(wait_for(lambda: len(seen) == 4))
        self.assertEqual(set(seen.values()), {1})
        self.assertTrue(wait_for(lambda: jobs.stats()['failed'] == 1))


    def test_a_job_pinned_to_a_busy_slot_moves_to_an_idle_one(self):
        jobs = self.queue(max_queue=4, slots=2, spill_after=0.2)
        hold, ran = threading.Event(), {}
        jobs.submit(jobs.reserve(), AIJob(run=lambda slot: hold.wait(5), slot=0))  # slot 0 busy
        self.assertTrue(wait_for(lambda: jobs.stats()['running'] == 1))
        pinned = AIJob(run=lambda slot: ran.setdefault('slot', slot), slot=0)
        jobs.submit(jobs.reserve(), pinned)
        sleep(0.1)
        self.assertEqual(ran, {})                     # still waiting for its own slot
        self.assertTrue(wait_for(lambda: ran.get('slot') == 1, timeout=3))
        self.assertEqual((pinned.moved_from, jobs.stats()['moved']), (0, 1))
        hold.set()

    def test_a_pinned_job_waits_for_its_free_slot(self):
        jobs = self.queue(max_queue=4, slots=2, spill_after=0.0)
        ran = {}
        jobs.submit(jobs.reserve(), AIJob(run=lambda slot: ran.setdefault('slot', slot), slot=1))
        self.assertTrue(wait_for(lambda: 'slot' in ran))
        self.assertEqual(ran['slot'], 1)              # its slot was idle: no move
        self.assertEqual(jobs.stats()['moved'], 0)


class SessionManagerTest(unittest.TestCase):
    class Session:
        def __init__(self, digest):
            self.digest, self.closed, self.is_busy = digest, False, False

        def busy(self):
            return self.is_busy

        def close(self):
            self.closed = True

    def test_create_reuse_limit_and_ttl(self):
        now = [0.0]
        manager = SessionManager(self.Session, max_sessions=2, ttl_seconds=10, clock=lambda: now[0])
        sid, first, created = manager.get_or_create(None)
        self.assertTrue(created)
        self.assertEqual(first.digest, session_hash(sid))
        self.assertGreaterEqual(len(sid), 40)
        self.assertEqual(manager.get_or_create(sid)[1:], (first, False))
        self.assertTrue(manager.get_or_create('forged')[2])  # an unknown id gets a new session
        with self.assertRaises(SessionLimit):
            manager.get_or_create(None)
        first.is_busy = True
        now[0] = 11.0
        self.assertIs(manager.get(sid), first)       # busy: kept, and seen again now
        first.is_busy = False
        now[0] = 22.0
        self.assertIsNone(manager.get(sid))          # idle past the TTL: closed
        self.assertTrue(first.closed)
        self.assertEqual(manager.expired, 2)

    def test_cookie(self):
        header = cookie_header('abc', secure=True)
        for part in (f'{COOKIE}=abc', 'HttpOnly', 'SameSite=Lax', 'Path=/', 'Secure'):
            self.assertIn(part, header)
        self.assertNotIn('Secure', cookie_header('abc', secure=False))
        self.assertEqual(session_id_from_cookie(f'x=1; {COOKIE}=abc; y=2'), 'abc')
        self.assertIsNone(session_id_from_cookie('x=1'))


class AsyncSessionTest(_Gate):
    def setUp(self):
        super().setUp()
        self.jobs = AIJobQueue(max_queue=2)
        self.addCleanup(self.jobs.close)

    def session(self, **kwargs):
        return web.PlaySession(log_root=Path(self.tmp.name), **kwargs)

    def test_human_move_queues_the_reply_and_polling_never_waits(self):
        session = self.session()
        state = session.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        self.assertEqual(state['status'], 'READY')   # human white: the human moves after the opening
        state = session.play_human_async(6, 6, jobs=self.jobs)
        self.assertIn(state['status'], ('QUEUED', 'THINKING'))
        self.assertTrue(wait_for(lambda: session.state()['status'] == 'THINKING'))
        started = monotonic()
        polled = session.state()                      # the search holds no session lock
        self.assertLess(monotonic() - started, 0.5)
        self.assertEqual(polled['move_count'], 2)
        with self.assertRaises(web.SessionBusy):
            session.play_human_async(5, 5, jobs=self.jobs)
        with self.assertRaises(web.SessionBusy):
            session.undo()
        with self.assertRaises(web.SessionBusy):
            session.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        GateAgent.gate.set()
        self.assertTrue(wait_for(lambda: session.state()['status'] == 'READY'))
        state = session.state()
        self.assertEqual(state['move_count'], 3)
        self.assertIsNotNone(state['last_ai_move'])
        self.assertIsNotNone(session.move_records[-1]['seconds'])

    def test_ai_first_when_human_black(self):
        session = self.session()
        GateAgent.gate.set()
        state = session.new_game_async(agent_key='gate', human_color=BLACK, seed=1, jobs=self.jobs)
        self.assertIn(state['status'], ('QUEUED', 'THINKING', 'READY'))
        self.assertTrue(wait_for(lambda: session.state()['move_count'] == 2 and not session.busy()))

    def test_new_game_cancels_a_queued_reply(self):
        blocker, session = self.session(), self.session()
        blocker.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        blocker.play_human_async(6, 6, jobs=self.jobs)           # occupies the slot
        self.assertTrue(wait_for(lambda: blocker.state()['status'] == 'THINKING'))
        session.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        state = session.play_human_async(6, 6, jobs=self.jobs)
        self.assertEqual((state['status'], state['queue_position']), ('QUEUED', 1))
        state = session.new_game_async(agent_key='gate', human_color=WHITE, seed=2, jobs=self.jobs)
        self.assertEqual((state['status'], state['move_count']), ('READY', 1))
        GateAgent.gate.set()
        self.assertTrue(wait_for(lambda: not blocker.busy()))
        sleep(0.1)
        self.assertEqual(GateAgent.calls, 1)                       # the cancelled reply never ran
        self.assertEqual(session.state()['move_count'], 1)

    def test_full_queue_refuses_before_changing_the_game(self):
        blocker = self.session()
        blocker.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        blocker.play_human_async(6, 6, jobs=self.jobs)
        self.assertTrue(wait_for(lambda: blocker.state()['status'] == 'THINKING'))
        waiting = [self.session() for _ in range(2)]
        for s in waiting:
            s.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
            s.play_human_async(6, 6, jobs=self.jobs)
        late = self.session()
        late.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        with self.assertRaises(QueueFull):
            late.play_human_async(6, 6, jobs=self.jobs)
        self.assertEqual((late.state()['move_count'], late.state()['status']), (1, 'READY'))
        with self.assertRaises(QueueFull):  # human black: the AI's first reply needs room too
            late.new_game_async(agent_key='gate', human_color=BLACK, seed=1, jobs=self.jobs)
        GateAgent.gate.set()

    def test_search_error_is_reported_and_the_session_stays_usable(self):
        session = self.session()
        session.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        GateAgent.fail = True
        GateAgent.gate.set()
        session.play_human_async(6, 6, jobs=self.jobs)
        self.assertTrue(wait_for(lambda: session.state()['last_error'] is not None))
        state = session.state()
        self.assertEqual((state['status'], state['move_count']), ('READY', 2))
        self.assertIn('boom', state['last_error'])
        session.undo()                                            # back to the human's turn
        self.assertEqual(session.state()['move_count'], 1)

    def test_reply_for_an_older_game_is_dropped(self):
        session = self.session()
        session.new_game_async(agent_key='gate', human_color=WHITE, seed=1, jobs=self.jobs)
        session.play_human_async(6, 6, jobs=self.jobs)
        self.assertTrue(wait_for(lambda: session.state()['status'] == 'THINKING'))
        with session.lock:
            session.epoch += 1                                     # as if the game had changed
        GateAgent.gate.set()
        self.assertTrue(wait_for(lambda: session.state()['status'] == 'READY'))  # not stuck in THINKING
        self.assertEqual(session.state()['move_count'], 2)
        self.assertEqual(self.jobs.stats()['reserved'], 0)

    def test_close_cancels_and_marks_the_game_abandoned(self):
        from weblog.store import LogStore

        db = Path(self.tmp.name) / 'web.db'
        store = LogStore(db)
        session = self.session(store=store, session_hash='h' * 16)
        session.new_game_async(agent_key='v5', human_color=WHITE, seed=1, jobs=self.jobs)
        session.close()
        store.close()
        conn = sqlite3.connect(db)
        self.assertEqual(conn.execute('SELECT status, session_hash FROM games').fetchall(), [('abandoned', 'h' * 16)])
        conn.close()


class HttpTest(_Gate):
    def setUp(self):
        super().setUp()
        from weblog.store import LogStore

        self.db = Path(self.tmp.name) / 'web.db'
        self.store = LogStore(self.db)
        self.app = web.WebApp(store=self.store, log_root=Path(self.tmp.name), max_sessions=3, max_queue=1)
        previous, web.APP = web.APP, self.app
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            self.app.close()
            self.store.close()
            web.APP = previous
        self.addCleanup(stop)

    def call(self, method, path, body=None, cookie=None, content_type='application/json'):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        headers = {}
        if cookie:
            headers['Cookie'] = f'{COOKIE}={cookie}'
        if body is not None:
            headers['Content-Type'] = content_type
        conn.request(method, path, None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        payload = json.loads(response.read() or b'{}')
        set_cookie = response.getheader('Set-Cookie')
        retry = response.getheader('Retry-After')
        conn.close()
        return response.status, payload, session_id_from_cookie(set_cookie), set_cookie, retry

    def test_two_browsers_two_games(self):
        status, _, a, header, _ = self.call('GET', '/api/state')
        self.assertEqual(status, 200)
        self.assertIn('HttpOnly', header)
        self.assertIn('SameSite=Lax', header)
        _, _, b, _, _ = self.call('GET', '/api/state')
        self.assertNotEqual(a, b)
        self.call('POST', '/api/new', {'agent': 'gate', 'human_color': 'WHITE'}, cookie=a)
        status, state, again, _, _ = self.call('POST', '/api/move', {'row': 6, 'col': 6}, cookie=a)
        self.assertEqual((status, again), (202, None))            # queued; no new cookie for a known session
        self.assertIn(state['status'], ('QUEUED', 'THINKING'))
        self.assertEqual(self.call('GET', '/api/state', cookie=b)[1]['move_count'], 0)  # b's own game
        self.assertEqual(self.call('POST', '/api/move', {'row': 5, 'col': 5}, cookie=a)[0], 409)
        GateAgent.gate.set()
        self.assertTrue(wait_for(lambda: self.call('GET', '/api/state', cookie=a)[1]['status'] == 'READY'))
        self.assertEqual(self.call('GET', '/api/state', cookie=a)[1]['move_count'], 3)
        self.store.flush()
        conn = sqlite3.connect(self.db)
        hashes = [row[0] for row in conn.execute('SELECT session_hash FROM games')]
        conn.close()
        self.assertEqual(hashes, [session_hash(a)])                # the hash, never the id
        self.assertNotIn(a, hashes)

    def test_queue_full_is_503_and_changes_nothing(self):
        a = self.call('GET', '/api/state')[2]
        b = self.call('GET', '/api/state')[2]
        c = self.call('GET', '/api/state')[2]
        for sid in (a, b, c):
            self.call('POST', '/api/new', {'agent': 'gate', 'human_color': 'WHITE'}, cookie=sid)
        self.call('POST', '/api/move', {'row': 6, 'col': 6}, cookie=a)
        self.assertTrue(wait_for(lambda: self.call('GET', '/api/state', cookie=a)[1]['status'] == 'THINKING'))
        self.assertEqual(self.call('POST', '/api/move', {'row': 6, 'col': 6}, cookie=b)[0], 202)  # the 1 place
        status, payload, _, _, retry = self.call('POST', '/api/move', {'row': 6, 'col': 6}, cookie=c)
        self.assertEqual((status, retry), (503, '10'))
        self.assertEqual(self.call('GET', '/api/state', cookie=c)[1]['move_count'], 1)
        status, _, _, _, retry = self.call('GET', '/api/state')   # a 4th browser: session limit
        self.assertEqual((status, retry), (503, '60'))
        GateAgent.gate.set()

    def test_works_under_a_proxy_path_prefix(self):
        self.assertEqual(web._route('/renju/api/state?x=1'), '/api/state')   # Caddy "handle /renju/*"
        self.assertEqual(web._route('/api/state'), '/api/state')             # "handle_path" or the root
        self.assertEqual(web._route('/renju/'), '/')
        sid = self.call('GET', '/renju/api/state')[2]
        self.assertEqual(self.call('POST', '/renju/api/new', {'agent': 'v5', 'human_color': 'WHITE'}, cookie=sid)[0], 200)
        self.assertEqual(self.call('GET', '/renju/api/state', cookie=sid)[1]['move_count'], 1)
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        conn.request('GET', '/renju/')
        response = conn.getresponse()
        self.assertEqual((response.status, b'api/state' in response.read()), (200, True))
        conn.close()

    def test_only_json_posts(self):
        sid = self.call('GET', '/api/state')[2]
        status = self.call('POST', '/api/new', {'agent': 'v5'}, cookie=sid, content_type='text/plain')[0]
        self.assertEqual(status, 415)

    def test_health(self):
        status, payload, _, _, _ = self.call('GET', '/api/health')
        self.assertEqual(status, 200)
        self.assertEqual(payload['ai']['slots'], 1)


if __name__ == '__main__':
    unittest.main()
