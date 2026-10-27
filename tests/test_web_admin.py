"""Admin stages 1/2: real read-only SQLite, HTTP auth, session lifecycle and CSRF."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
import json
import os
import socket
import time
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from webadmin.auth import Auth, IDLE, LIFETIME, LOCKOUT, hash_password
from webadmin.inspect import inspect_database
from webadmin.server import AdminServer, game_status, validate_origin
from weblog.store import initialize, open_readonly

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


class DatabaseFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'games.db'
        initialize(self.db)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            games = [
                ('clean', 'finished', 'AI_WIN', 0, 'tbv3'),
                ('undo', 'finished', 'HUMAN_WIN', 2, 'tbv3'),
                ('draw', 'finished', 'DRAW', 0, 'rt30'),
                ('unknown', 'finished', None, 0, 'rt30'),
                ('stale', 'in_progress', None, 0, 'tbv3'),
                ('active', 'in_progress', None, 0, 'tbv3'),
                ('left', 'abandoned', 'UNFINISHED', 0, 'rt30'),
                ('unfinished', 'unfinished', 'UNFINISHED', 0, 'rt30'),
            ]
            for gid, status, result, undo, agent in games:
                conn.execute('''INSERT INTO games (game_id, started_at, status, result, undo_count,
                    agent_key, agent_name, engine_config_hash, human_color, session_hash)
                    VALUES (?, '2026-10-26T00:00:00+00:00', ?, ?, ?, ?, ?, 'engine', 'WHITE', 'PRIVATE')''',
                             (gid, status, result, undo, agent, agent))
            for gid, seq, ply, stamp, undone in [
                ('undo', 0, 1, '2026-10-26T01:00:00+00:00', None),
                ('undo', 1, 2, '2026-10-26T01:01:00+00:00', '2026-10-26T01:02:00+00:00'),
                ('undo', 2, 2, '2026-10-26T01:03:00+00:00', None),
                ('active', 0, 1, '2026-10-27T11:45:00+00:00', None),
            ]:
                conn.execute('''INSERT INTO moves (game_id, seq, ply, player, actor, row0, col0, recorded_at,
                    undone_at) VALUES (?, ?, ?, 'BLACK', 'OPENING_RULE', 7, 7, ?, ?)''',
                             (gid, seq, ply, stamp, undone))
            conn.execute('''INSERT INTO veto_positions (game_id, ply, recorded_at, reason, payload_json)
                VALUES ('undo', 2, '2026-10-26T01:00:00+00:00', 'VCT2_PROVEN_LOSS', '{}')''')


class InspectionTests(DatabaseFixture):
    def test_real_values_undo_included_and_no_database_write(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        report = inspect_database(self.db, now=NOW)
        self.assertEqual(report['counts'], {'games': 8, 'moves': 4, 'veto_positions': 1})
        self.assertEqual(report['in_progress'], 2)
        self.assertEqual(report['suspected_stopped'], 1)
        self.assertEqual(report['finished_summary']['finished'], 4)
        self.assertEqual(report['finished_summary']['ai_win_rate'], .25)
        self.assertEqual(report['finished_summary']['unknown_results'], 1)
        self.assertEqual(report['undo_games']['human_wins'], 1)
        self.assertEqual(report['undo_games']['games'], 1)
        variants = {row['agent_key']: row for row in report['engine_veto_counts']}
        self.assertEqual(variants['tbv3']['games'], 4)
        self.assertEqual(variants['tbv3']['veto_rows'], 1)
        self.assertEqual(variants['rt30']['veto_rows'], 0)
        self.assertNotIn('PRIVATE', json.dumps(report))
        self.assertTrue(report['policy']['include_undo_games'])
        self.assertEqual(report['reader_pragmas']['journal_mode'], 'wal')
        self.assertIn('games_started', report['indexes']['games'])
        self.assertEqual(before, hashlib.sha256(self.db.read_bytes()).hexdigest())
        with closing(open_readonly(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT status FROM games WHERE game_id='stale'").fetchone()[0],
                             'in_progress')
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM games")

    def test_missing_schema_and_missing_file_fail_without_creation(self):
        missing = self.db.parent / 'missing.db'
        with self.assertRaises(sqlite3.Error):
            inspect_database(missing)
        self.assertFalse(missing.exists())
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute('PRAGMA user_version=3')
        with self.assertRaisesRegex(ValueError, 'expected schema 2, found 3'):
            inspect_database(self.db)

    def test_missing_column_is_reported_not_migrated(self):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute('ALTER TABLE games RENAME COLUMN undo_count TO old_undo_count')
        with self.assertRaisesRegex(ValueError, 'undo_count'):
            inspect_database(self.db)

    def test_empty_database_rate_is_unknown(self):
        empty = self.db.parent / 'empty.db'
        initialize(empty)
        report = inspect_database(empty, now=NOW)
        self.assertIsNone(report['finished_summary']['ai_win_rate'])
        self.assertIsNone(report['latest_game'])
        self.assertEqual(report['undo_games']['games'], 0)

    def test_recent_undo_counts_as_activity_and_bad_dates_are_visible(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE moves SET undone_at='2026-10-27T11:59:00+00:00' WHERE game_id='active'")
            conn.execute("UPDATE moves SET recorded_at='2026-10-26T00:00:00+00:00' WHERE game_id='active'")
            conn.execute("UPDATE games SET started_at='bad-date' WHERE game_id='stale'")
        report = inspect_database(self.db, now=NOW)
        self.assertEqual(report['suspected_stopped'], 0)
        self.assertEqual(report['unclassified_activity_time'], 1)

    def test_inspection_cli_and_no_engine_imports(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/ser5/admin_inspect.py'), '--db', str(self.db)],
                                text=True, capture_output=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(len(report['checkout_commit']), 40)
        self.assertNotIn('PRIVATE', result.stdout)
        program = """import sys
import webadmin.server, webadmin.inspect
assert not any(x in sys.modules for x in ('torch', 'agents', 'analysis', 'rt30', 'scripts.run_web_play'))
"""
        subprocess.run([sys.executable, '-c', program], check=True)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/ser5/admin_inspect.py'),
                                 '--db', str(self.db.parent / 'absent.db')], capture_output=True)
        self.assertEqual(result.returncode, 1)


class EntrypointTests(DatabaseFixture):
    def test_missing_credentials_fail_closed(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith('ADMIN_')}
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_web_admin.py'), '--db', str(self.db)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('ADMIN_USER', result.stderr)

    def test_real_process_local_prefix_and_login(self):
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1', 0))
            port = reserved.getsockname()[1]
        origin = f'http://127.0.0.1:{port}'
        env = {**os.environ, 'ADMIN_USER': 'admin', 'ADMIN_PASSWORD_HASH': hash_password('entry-test'),
               'ADMIN_ORIGIN': origin}
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_web_admin.py'), '--db', str(self.db),
                                    '--port', str(port)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    self.fail('admin process exited before listening')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.05)
            conn = HTTPConnection('127.0.0.1', port, timeout=5)
            self.addCleanup(conn.close)
            conn.request('GET', '/renju/admin/login')
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn('로그인', response.read().decode())
            body = json.dumps({'username': 'admin', 'password': 'entry-test'})
            conn.request('POST', '/renju/admin/login', body, {'Origin': origin, 'Content-Type': 'application/json'})
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            cookie = response.getheader('Set-Cookie')
            self.assertNotIn('; Secure', cookie)  # local HTTP only; HTTPS cookie flags tested separately
            conn.request('GET', '/renju/admin/', headers={'Cookie': cookie.split(';')[0]})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn('읽기 전용', response.read().decode())
        finally:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        self.assertEqual(process.returncode, 0)


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_password('test-pass')

    def setUp(self):
        self.now = 0
        self.auth = Auth('관리자', self.password_hash, clock=lambda: self.now)

    def test_wrong_credentials_lockout_cannot_be_bypassed_with_correct_password(self):
        for _ in range(4):
            self.assertEqual(self.auth.login('ip', 'wrong', 'test-pass'), (None, False))
        self.assertEqual(self.auth.login('ip', '관리자', 'wrong'), (None, True))
        self.assertEqual(self.auth.login('ip', '관리자', 'test-pass'), (None, True))
        self.now += LOCKOUT
        token, locked = self.auth.login('ip', '관리자', 'test-pass')
        self.assertTrue(token)
        self.assertFalse(locked)
        self.assertTrue(self.auth.check(token))

    def test_idle_and_absolute_expiry_and_logout(self):
        token, _ = self.auth.login('ip', '관리자', 'test-pass')
        self.now = IDLE
        self.assertFalse(self.auth.check(token))
        token, _ = self.auth.login('ip', '관리자', 'test-pass')
        start = self.now
        for elapsed in range(600, LIFETIME, 600):
            self.now = start + elapsed
            self.assertTrue(self.auth.check(token))
        self.now = start + LIFETIME
        self.assertFalse(self.auth.check(token))
        token, _ = self.auth.login('ip', '관리자', 'test-pass')
        self.auth.logout(token)
        self.assertFalse(self.auth.check(token))
        self.assertFalse(Auth('관리자', self.password_hash).check(token))

    def test_config_rejected_and_random_salts(self):
        for value in ('', 'plain', self.password_hash.replace('$16384$', '$999999999$')):
            with self.assertRaises(ValueError):
                Auth('admin', value)
        with self.assertRaises(ValueError):
            Auth('', self.password_hash)
        self.assertNotEqual(self.password_hash, hash_password('test-pass'))
        for origin in ('', 'https://example.org/', 'http://example.org', 'https://u:p@host',
                       'https://example.org:999999', 'https://host/#x'):
            with self.assertRaises(ValueError):
                validate_origin(origin)
        self.assertEqual(validate_origin('https://example.org'), 'https://example.org')
        self.assertEqual(validate_origin('http://127.0.0.1:8010'), 'http://127.0.0.1:8010')


class HTTPTests(DatabaseFixture):
    def setUp(self):
        super().setUp()
        self.now = 0
        self.auth = Auth('admin', hash_password('test-password'), clock=lambda: self.now)
        self.origin = 'https://example.org'
        self.server = AdminServer(('127.0.0.1', 0), db=self.db, origin=self.origin,
                                  auth=self.auth, assets=ROOT / 'web/admin')
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        self.status_mock = patch('webadmin.server.game_status', return_value={'health': {'ok': True}})
        self.mock_status = self.status_mock.start()
        self.addCleanup(self.status_mock.stop)

    def call(self, path='/', method='GET', body=None, cookie=None, headers=None):
        headers = dict(headers or {})
        if cookie:
            headers['Cookie'] = cookie
        if body is not None:
            body = json.dumps(body)
            headers.setdefault('Content-Type', 'application/json')
        conn = HTTPConnection(*self.server.server_address, timeout=5)
        try:
            conn.request(method, path, body, headers)
            response = conn.getresponse()
            data, returned = response.read(), dict(response.getheaders())
            self.assertEqual(returned.get('Cache-Control'), 'no-store')
            return response.status, data, returned
        finally:
            conn.close()

    def login(self, **kwargs):
        return self.call('/login', 'POST', {'username': 'admin', 'password': 'test-password'},
                         headers={'Origin': self.origin, **kwargs})

    def test_public_login_and_all_other_paths_protected(self):
        status, _, headers = self.call()
        self.assertEqual(status, 302)
        self.assertEqual(headers['Location'], '/renju/admin/login')
        for path in ('/login', '/login.css', '/login.js', '/renju/admin/login'):
            self.assertEqual(self.call(path)[0], 200)
        for path in ('/data/status', '/data/games', '/data/exports/test.tar.gz', '/admin.js',
                     '/api/health', '/renju/admin/data/status', '/../../deploy/renju-admin.env'):
            self.assertEqual(self.call(path)[0], 401, path)
        self.assertEqual(self.call('/data/backup', 'POST', {})[0], 401)
        self.assertEqual(self.call('/data/status', 'HEAD')[0], 401)
        self.assertFalse(self.mock_status.called)

    def test_login_readonly_status_rotation_and_logout(self):
        status, _, headers = self.login()
        self.assertEqual(status, 200)
        issued = headers['Set-Cookie']
        for flag in ('Path=/renju/admin', 'HttpOnly', 'Secure', 'SameSite=Strict'):
            self.assertIn(flag, issued)
        cookie = issued.split(';')[0]
        status, body, _ = self.call('/renju/admin/data/status', cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['database']['finished_summary']['ai_win_rate'], .25)
        self.assertNotIn(b'PRIVATE', body)
        self.assertEqual(self.call('/', cookie=cookie)[0], 200)
        self.assertEqual(self.call('/admin.js', cookie=cookie)[0], 200)
        status, _, headers = self.call('/login', 'POST', {'username': 'admin', 'password': 'test-password'},
                                      cookie=cookie, headers={'Origin': self.origin})
        self.assertEqual(status, 200)
        self.assertEqual(self.call('/data/status', cookie=cookie)[0], 401)
        cookie = headers['Set-Cookie'].split(';')[0]
        self.assertEqual(self.call('/logout', 'POST', {}, cookie=cookie)[0], 403)
        status, _, headers = self.call('/logout', 'POST', {}, cookie=cookie, headers={'Origin': self.origin})
        self.assertEqual(status, 200)
        self.assertIn('Max-Age=0', headers['Set-Cookie'])
        self.assertEqual(self.call('/data/status', cookie=cookie)[0], 401)

    def test_post_origin_and_input_validation(self):
        body = {'username': 'admin', 'password': 'test-password'}
        for headers in ({}, {'Origin': 'https://attacker.org'}, {'Origin': 'null'},
                        {'Origin': self.origin + '/', 'X-Forwarded-Host': 'example.org'}):
            self.assertEqual(self.call('/login', 'POST', body, headers=headers)[0], 403)
        for body in ({}, {'username': [], 'password': 'x'}, {'username': 'x', 'password': 'a' * 1025}):
            self.assertEqual(self.call('/login', 'POST', body, headers={'Origin': self.origin})[0], 400)
        self.assertEqual(self.call('/login', 'POST', {},
                                  headers={'Origin': self.origin, 'Content-Type': 'text/plain'})[0], 415)

    def test_rate_limit_uses_last_forwarded_address(self):
        for i in range(5):
            status, _, _ = self.call('/login', 'POST', {'username': 'admin', 'password': 'wrong'},
                                     headers={'Origin': self.origin,
                                              'X-Forwarded-For': f'192.0.2.{i}, 198.51.100.1'})
        self.assertEqual(status, 429)
        self.assertEqual(self.login(**{'X-Forwarded-For': '192.0.2.44, 198.51.100.1'})[0], 429)
        self.assertEqual(self.login(**{'X-Forwarded-For': '198.51.100.2'})[0], 200)
        self.now = LOCKOUT
        self.assertEqual(self.login(**{'X-Forwarded-For': '198.51.100.1'})[0], 200)

    def test_session_expiry_and_bad_cookie(self):
        _, _, headers = self.login()
        cookie = headers['Set-Cookie'].split(';')[0]
        self.now = IDLE
        self.assertEqual(self.call('/data/status', cookie=cookie)[0], 401)
        self.assertEqual(self.call('/data/status', cookie='renju_admin=madeup')[0], 401)

    def test_schema_failure_is_503_not_create_or_migrate(self):
        _, _, headers = self.login()
        cookie = headers['Set-Cookie'].split(';')[0]
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute('PRAGMA user_version=3')
        self.assertEqual(self.call('/data/status', cookie=cookie)[0], 503)
        with closing(open_readonly(self.db)) as conn:
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 3)

    def test_health_client_fixed_loopback_endpoints_no_proxies(self):
        self.status_mock.stop()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return b'{"ok":true}'
        with patch('webadmin.server.build_opener') as build:
            build.return_value.open.side_effect = [Response(), OSError('offline')]
            status = game_status()
            urls = [call.args[0] for call in build.return_value.open.call_args_list]
        self.assertEqual(urls, ['http://127.0.0.1:8000/api/health', 'http://127.0.0.1:8000/api/engines'])
        self.assertEqual(status['engines'], {'available': False})


if __name__ == '__main__':
    unittest.main()
