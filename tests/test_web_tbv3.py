"""The web's latest-certified opponent TB-V3 / fast5 (rt30.tb_v3_slot): fail-closed start, no fallback, provenance."""
import unittest
from unittest import mock

import scripts.run_web_play as web
from renju import WHITE
from rt30 import tb_v3_slot
from rt30.supervisor import MoveResult


class FakeSupervisor:
    """Answers like ``EngineSupervisor.select`` with a fixed source and engine time."""

    def __init__(self, source='engine', seconds=1.0):
        self.source, self.seconds, self.calls = source, seconds, 0

    def select(self, *, game_key, seed, history, received_at=None):
        from renju import Game

        self.calls += 1
        game = Game()
        for move in history:
            game.play(*move)
        move = min(game.legal_moves())
        return MoveResult(move, self.source, diagnostics={'v8_route': 'tree'}, engine_seconds=self.seconds,
                          wall_seconds=self.seconds, deadline_seconds=tb_v3_slot.HANG_SECONDS - 1,
                          error=None if self.source == 'engine' else 'hung')


INFO = {'engine': 'TB-V3 Resolve/fast5', 'engine_id': 'TB-V3', 'engine_config_hash': 'd0ef' * 16, 'runtime': 'fast5',
        'runtime_hash': '669c' * 16, 'fast_mode': 'fast5', 'value_device': 'cpu', 'model': 'latest-certified',
        'policy_checkpoint_sha256': 'a' * 64, 'policy_metadata_sha256': 'b' * 64,
        'value_checkpoint_sha256': 'c' * 64, 'value_metadata_sha256': 'd' * 64, 'evidence': {}, 'evidence_summary': {}}


class Registered(unittest.TestCase):
    def register(self, supervisor):
        saved = dict(web._TBV3), web.VERSION_LABELS.get(web.TBV3_KEY)
        self.addCleanup(self._restore, saved)
        web.register_tbv3([supervisor], tb_v3_slot.LABEL, INFO, tb_v3_slot.SOFT_LIMIT_SECONDS)

    @staticmethod
    def _restore(saved):
        web._TBV3.clear()
        web._TBV3.update(saved[0])
        if saved[1] is None:
            web.VERSION_LABELS.pop(web.TBV3_KEY, None)
        else:
            web.VERSION_LABELS[web.TBV3_KEY] = saved[1]


class AgentTest(Registered):
    def test_engine_move_and_soft_monitor(self):
        from renju import Game

        self.register(FakeSupervisor(seconds=31.0))
        agent = web.create_agent(web.TBV3_KEY, seed=3)
        self.assertIsInstance(agent, web.CertifiedSupervisedAgent)
        game = Game()
        game.play(7, 7)
        move = agent.select_move(game)
        self.assertIn(move, game.legal_moves())
        self.assertTrue(agent.last.rt['soft_limit_exceeded'])  # played as searched, flagged
        self.assertIsNone(agent.last.deadline_seconds)

    def test_no_fallback_move(self):
        from renju import Game

        for source in ('watchdog', 'fallback', 'worker_error', 'no_worker'):
            self.register(FakeSupervisor(source=source))
            game = Game()
            game.play(7, 7)
            with self.assertRaises(RuntimeError, msg=source):
                web.create_agent(web.TBV3_KEY, seed=3).select_move(game)

    def test_failed_search_plays_nothing(self):
        self.register(FakeSupervisor(source='watchdog'))
        session = web.PlaySession()
        session.reset(agent_key='v6', human_color=WHITE, seed=7)
        session.agent_key, session.agent = web.TBV3_KEY, web.create_agent(web.TBV3_KEY, seed=7)
        session.game.play(0, 0)  # white (human) moved: black (TB-V3) to move
        history = list(session.game.history)
        with session.lock, self.assertRaises(RuntimeError):
            session._play_ai_locked()
        self.assertEqual(session.game.history, history)

    def test_default_and_provenance(self):
        self.register(FakeSupervisor())
        self.assertEqual(web.default_agent_key(), web.TBV3_KEY)
        self.assertEqual(web.engine_summary(), {'engine': 'TB-V3', 'runtime': 'fast5', 'value_device': 'cpu',
                                                'model': 'latest-certified', 'engine_hash': 'd0ef' * 3,
                                                'runtime_hash': '669c' * 3})
        details = web.engine_details()
        self.assertEqual(details['value_checkpoint_sha256'], 'c' * 64)
        self.assertNotIn('policy_checkpoint', details)  # no paths in the public diagnostics
        session = web.PlaySession()
        session.agent_key = web.TBV3_KEY
        info = session._s3_info()
        self.assertEqual((info['engine_config_hash'], info['config']), (INFO['engine_config_hash'],
                                                                        tb_v3_slot.LATEST_CERTIFIED))

    def test_not_served_without_registration(self):
        saved = dict(web._TBV3)
        self.addCleanup(lambda: (web._TBV3.clear(), web._TBV3.update(saved)))
        web._TBV3.clear()
        self.assertNotEqual(web.default_agent_key(), web.TBV3_KEY)
        self.assertIsNone(web.engine_summary())


class SlowSupervisor(FakeSupervisor):
    def select(self, **kw):
        import time
        time.sleep(self.seconds)
        return super().select(**kw)


class SmokeScriptTest(Registered):
    """``scripts/ser5/tbv3_smoke.py`` against a real HTTP server whose TB-V3 worker is a stand-in."""

    def serve(self, secure_cookie=False):
        import threading
        from http.server import ThreadingHTTPServer
        from pathlib import Path
        from tempfile import TemporaryDirectory

        tmp = TemporaryDirectory()
        app = web.WebApp(log_root=Path(tmp.name), max_sessions=4, max_queue=1, secure_cookie=secure_cookie)
        previous, web.APP = web.APP, app
        server = ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def stop():
            server.shutdown()
            server.server_close()
            app.close()
            web.APP = previous
            tmp.cleanup()
        self.addCleanup(stop)
        return f'http://127.0.0.1:{server.server_address[1]}'

    def smoke(self, base, *args):
        import io
        from contextlib import redirect_stdout

        from scripts.ser5 import tbv3_smoke

        out = io.StringIO()
        with redirect_stdout(out):
            code = tbv3_smoke.main(['--base', base, *args])
        return code, out.getvalue()

    def test_smoke_http_with_normal_cookie_passes(self):
        self.register(FakeSupervisor(seconds=0.5))
        base = self.serve()
        code, out = self.smoke(base, '--moves', '3', '--timeout', '20')
        self.assertEqual(code, 0, out)
        self.assertIn('PASS  session: cookie not Secure', out)
        web._TBV3['supervisors'][0].source = 'watchdog'  # the worker stops answering: the script fails
        code, out = self.smoke(base, '--moves', '2', '--timeout', '20')
        self.assertEqual(code, 1)
        self.assertIn('no engine move', out)

    def test_smoke_http_with_secure_cookie_refuses_immediately(self):
        """The production server (--secure-cookie) over http://: refused at the session check, not a timeout."""
        import time

        self.register(FakeSupervisor(seconds=0.5))
        base = self.serve(secure_cookie=True)
        started = time.monotonic()
        code, out = self.smoke(base, '--moves', '3', '--timeout', '60')
        self.assertEqual(code, 1, out)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIn('Secure session cookie', out)
        self.assertIn('secure_cookie=True', out)  # the health snapshot shows the server's setting

    def test_smoke_https_secure_cookie_preserves_session(self):
        """Over https:// a Secure cookie comes back: the continuity check passes (and fails on another game)."""
        import http.cookiejar

        from scripts.ser5 import tbv3_smoke

        class Stub(tbv3_smoke.Client):
            def __init__(self, seed):
                super().__init__('https://example.invalid')
                self.seed = seed
                self.jar.set_cookie(http.cookiejar.Cookie(
                    0, tbv3_smoke.COOKIE, 'sid', None, False, 'example.invalid', False, False, '/', True, True,
                    None, False, None, None, {}))

            def call(self, path, body=None):
                return {'seed': self.seed, 'agent_key': 'tbv3'}, []

        state = {'seed': 5, 'agent_key': 'tbv3'}
        self.assertEqual(tbv3_smoke.check_session(Stub(5), [], state), 'Secure')
        with self.assertRaises(tbv3_smoke.SmokeFailure):
            tbv3_smoke.check_session(Stub(6), [], state)

    def test_timeout_is_reported_with_the_queue(self):
        self.register(SlowSupervisor(seconds=3.0))
        code, out = self.smoke(self.serve(), '--moves', '1', '--timeout', '1')
        self.assertEqual(code, 1)
        self.assertIn('smoke performance timeout, not an engine failure', out)
        self.assertIn('queue: running=', out)


class DeploymentCheckTest(unittest.TestCase):
    def test_fails_closed(self):
        from analysis import canonical, tb_v3

        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):  # no certified model files
            tb_v3_slot.check_deployment('missing/best.pt', 'missing/value/best.pt')
        models = (lambda path, policy=None: {k: v for k, v in tb_v3.MODELS.items() if k.startswith('policy')},
                  lambda path: {k: v for k, v in tb_v3.MODELS.items() if k.startswith('value')})
        with mock.patch('analysis.track_b_guard.check_policy_checkpoint', models[0]), \
                mock.patch('analysis.track_b_guard.check_value_checkpoint', models[1]):
            info = tb_v3_slot.check_deployment('p.pt', 'v.pt')  # evidence, identity and builds of this checkout
            from analysis import tb_v3_fast5
            self.assertEqual((info['runtime_hash'], info['value_device'], info['model']),
                             (tb_v3_fast5.runtime_hash('cpu'), 'cpu', 'latest-certified'))
            self.assertEqual(info['evidence_summary']['cpu_cert']['verdict'], 'PASS')
            self.assertIn('cpu_cert', info['evidence'])
            with mock.patch.object(canonical, 'CANONICAL_RUNTIME', 'fast4'), self.assertRaises(RuntimeError):
                tb_v3_slot.check_deployment('p.pt', 'v.pt')
            with mock.patch.object(canonical, '_git_blob_sha256', lambda commit, path: 'f' * 64), \
                    self.assertRaises(RuntimeError):  # evidence no longer matches this checkout
                tb_v3_slot.check_deployment('p.pt', 'v.pt')
        wrong = (lambda path, policy=None: {'policy_checkpoint_sha256': 'f' * 64, 'policy_metadata_sha256': 'f' * 64},
                 models[1])
        with mock.patch('analysis.track_b_guard.check_policy_checkpoint', wrong[0]), \
                mock.patch('analysis.track_b_guard.check_value_checkpoint', wrong[1]), \
                self.assertRaises(RuntimeError):
            tb_v3_slot.check_deployment('p.pt', 'v.pt')


if __name__ == '__main__':
    unittest.main()
