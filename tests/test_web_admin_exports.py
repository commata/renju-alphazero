"""Verify snapshot/dump/CSV equality, hashes, real background exports, locks and authenticated downloads."""
from contextlib import closing
import csv
import fcntl
import hashlib
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from tests.test_web_admin_games import build_database
from webadmin.auth import Auth, hash_password
from webadmin.exports import ExportBusy, ExportManager, TABLE_ORDER, build_export
from webadmin.server import AdminServer
from weblog.backup import snapshot_database
from weblog.store import LogStore

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.db, self.ids = build_database(self.folder)
        self.exports = self.folder / 'exports'
        self.exports.mkdir()
        self.manager = ExportManager(self.db, self.exports, repo=ROOT, log_root=self.folder / 'logs', keep=2)
        self.job = 'a' * 32

    def build(self, raw=False):
        return build_export(self.db, self.exports, self.job, repo=ROOT, log_root=self.folder / 'logs', include_raw_logs=raw)

    def validate_archive(self, path):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            self.assertFalse(any(name.startswith('/') or '..' in Path(name).parts for name in names))
            files = {name: archive.extractfile(name).read() for name in names}
        manifest = json.loads(files['manifest.json'])
        self.assertEqual(manifest['schema_version'], 2)
        for name, metadata in manifest['files'].items():
            self.assertEqual(metadata['sha256'], hashlib.sha256(files[name]).hexdigest())
            self.assertEqual(metadata['bytes'], len(files[name]))
        copy = self.folder / 'copy.db'
        copy.write_bytes(files['renju_web_full.db'])
        with closing(sqlite3.connect(copy)) as db, closing(sqlite3.connect(':memory:')) as restored:
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0], 'delete')
            restored.executescript(files['renju_web_full.sql'].decode('utf-8'))
            self.assertEqual(restored.execute('PRAGMA user_version').fetchone()[0], 2)
            self.assertEqual(restored.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(restored.execute('PRAGMA foreign_key_check').fetchall(), [])
            for table, count in manifest['counts'].items():
                records = db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall()
                self.assertEqual(records, restored.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall())
                csv_rows = list(csv.reader(io.StringIO(files[table + '.csv'].decode('utf-8-sig'))))
                self.assertEqual(len(csv_rows) - 1, count)
                self.assertEqual(len(records), count)
                expected_csv = [[str(value) if value is not None else '' for value in row]
                                for row in db.execute(f'SELECT * FROM {table} ORDER BY {TABLE_ORDER[table]}')]
                self.assertEqual(csv_rows[1:], expected_csv)
        return manifest, files

    def test_all_files_share_snapshot_and_raw_logs_are_optional(self):
        metadata = self.build()
        manifest, files = self.validate_archive(self.exports / f'{self.job}.tar.gz')
        self.assertEqual(len(files), 6)
        self.assertEqual(metadata['counts'], manifest['counts'])
        self.assertEqual(manifest['counts'], {'games': 3, 'moves': 23, 'veto_positions': 0})
        self.assertFalse(manifest['raw_logs_requested'])
        self.assertEqual(len(manifest['server_commit']), 40)
        self.assertFalse((self.exports / f'{self.job}.tmp').exists())

    def test_optional_logs_reject_symlinks_and_arbitrary_database_paths(self):
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("UPDATE games SET raw_log_dir='/etc' WHERE game_id=?", (self.ids['win'],))
        with closing(sqlite3.connect(self.db)) as db:
            raw = Path(db.execute('SELECT raw_log_dir FROM games WHERE game_id=?', (self.ids['undo'],)).fetchone()[0])
        (raw / 'moves.csv').unlink()
        secret = self.folder / 'secret.env'
        secret.write_text('ADMIN_PASSWORD_HASH=do-not-export')
        (raw / 'moves.csv').symlink_to(secret)
        self.build(raw=True)
        manifest, files = self.validate_archive(self.exports / f'{self.job}.tar.gz')
        self.assertGreaterEqual(manifest['raw_logs_skipped'], 2)
        self.assertEqual(manifest['raw_logs_included'], 1)
        self.assertFalse(any(b'do-not-export' in data for data in files.values()))

    def test_snapshot_does_not_overwrite_and_disk_failure_cleans_partial_files(self):
        target = self.folder / 'existing.db'
        target.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            snapshot_database(self.db, target)
        self.assertEqual(target.read_bytes(), b'preserve')
        with patch('webadmin.exports.shutil.disk_usage') as usage:
            usage.return_value.free = 0
            with self.assertRaisesRegex(OSError, 'disk space'):
                self.build()
        self.assertFalse((self.exports / f'{self.job}.tar.gz').exists())
        self.assertFalse((self.exports / f'{self.job}.tmp').exists())

    def wait(self, job_id):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            jobs = self.manager.list_jobs()['jobs']
            job = next(job for job in jobs if job['id'] == job_id)
            if job['state'] != 'creating':
                return job
            time.sleep(.03)
        self.fail('background export did not complete')

    def test_real_worker_retention_and_download_across_manager_restart(self):
        ids = []
        for _ in range(3):
            job = self.manager.start()
            ids.append(job['id'])
            self.assertEqual(self.wait(job['id'])['state'], 'ready')
        self.assertFalse((self.exports / f'{ids[0]}.tar.gz').exists())
        self.assertEqual(len(self.manager.list_jobs()['jobs']), 2)
        restarted = ExportManager(self.db, self.exports, repo=ROOT, log_root=self.folder / 'logs')
        handle, job = restarted.open_download(ids[-1])
        with handle:
            self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), job['sha256'])
        self.validate_archive(self.exports / f'{ids[-1]}.tar.gz')
        for name in ('../etc/passwd', 'bad', ids[0]):
            with self.assertRaises((ValueError, LookupError)):
                restarted.open_download(name)

    def test_worker_lock_survives_parent_descriptor_close_and_stale_jobs_recover(self):
        inherited = []
        class FakeChild:
            def wait(self): return None
        def spawn(*args, **kwargs):
            inherited.append(os.dup(kwargs['pass_fds'][0]))
            return FakeChild()
        with patch('webadmin.exports.subprocess.Popen', side_effect=spawn):
            job = self.manager.start()
        try:
            other = ExportManager(self.db, self.exports, repo=ROOT, log_root=self.folder / 'logs')
            with self.assertRaises(ExportBusy):
                other.start()
            self.assertTrue(other.list_jobs()['busy'])
        finally:
            os.close(inherited.pop())
        self.assertEqual(self.manager.list_jobs()['jobs'][0]['state'], 'failed')
        self.assertFalse(self.manager.list_jobs()['busy'])

    def test_worker_failure_is_reported_and_periodic_backup_cli_still_works(self):
        broken = ExportManager(self.folder / 'absent.db', self.exports, repo=ROOT, log_root=self.folder / 'logs')
        job = broken.start()
        result = self.wait(job['id'])
        self.assertEqual(result['state'], 'failed')
        self.assertFalse((self.exports / f'{job["id"]}.tar.gz').exists())
        self.assertFalse((self.folder / 'absent.db').exists())
        directory = self.folder / 'daily'
        subprocess.run([sys.executable, str(ROOT / 'deploy/backup_db.py'), '--db', str(self.db),
                        '--dir', str(directory), '--keep', '14'], check=True, capture_output=True)
        copies = list(directory.glob('*.db'))
        self.assertEqual(len(copies), 1)
        with closing(sqlite3.connect(copies[0])) as conn:
            self.assertEqual(conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 2)

    def test_consistency_while_writer_adds_games(self):
        store = LogStore(self.db)
        stopped = threading.Event()
        def write():
            while not stopped.wait(.002):
                store.start_game(agent_key='tbv3', agent_name='TB-V3', human_color='BLACK')
        thread = threading.Thread(target=write)
        thread.start()
        try:
            self.build()
        finally:
            stopped.set()
            thread.join()
            self.assertTrue(store.flush())
            store.close()
        manifest, _ = self.validate_archive(self.exports / f'{self.job}.tar.gz')
        self.assertGreaterEqual(manifest['counts']['games'], 3)
        self.assertEqual(store.failures, 0)


class ExportHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        db, _ = build_database(folder)
        self.manager = ExportManager(db, folder / 'exports', repo=ROOT, log_root=folder / 'logs')
        self.origin = 'https://example.org'
        self.server = AdminServer(('127.0.0.1', 0), db=db, origin=self.origin,
                                  auth=Auth('admin', hash_password('pw')), assets=ROOT / 'web/admin', exports=self.manager)
        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def call(self, path, cookie=None, method='GET', data=None, origin=True):
        headers = {'Content-Type': 'application/json'}
        if cookie: headers['Cookie'] = cookie
        if origin: headers['Origin'] = self.origin
        conn = HTTPConnection(*self.server.server_address, timeout=10)
        try:
            conn.request(method, path, json.dumps(data) if data is not None else None, headers)
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def login(self):
        status, _, headers = self.call('/login', method='POST', data={'username': 'admin', 'password': 'pw'})
        self.assertEqual(status, 200)
        return headers['Set-Cookie'].split(';')[0]

    def test_auth_csrf_validation_and_streaming_download(self):
        for path in ('/data/statistics', '/data/exports', '/data/exports/' + 'a'*32 + '/download', '/statistics.js', '/exports.js'):
            self.assertEqual(self.call(path)[0], 401)
        self.assertEqual(self.call('/data/exports', method='POST', data={})[0], 401)
        cookie = self.login()
        self.assertEqual(self.call('/data/exports', cookie, 'POST', {}, origin=False)[0], 403)
        self.assertEqual(self.call('/data/exports', cookie, 'POST', {'include_raw_logs': 'yes'})[0], 400)
        self.assertEqual(self.call('/data/exports/../download', cookie)[0], 400)
        self.assertEqual(self.call('/data/exports/' + 'a'*32 + '/download', cookie)[0], 404)
        self.assertEqual(self.call('/data/statistics?undo=no', cookie)[0], 200)
        self.assertEqual(self.call('/data/statistics?undo=x', cookie)[0], 400)
        status, body, _ = self.call('/data/exports', cookie, 'POST', {})
        self.assertEqual(status, 202)
        job_id = json.loads(body)['id']
        for _ in range(200):
            jobs = json.loads(self.call('/data/exports', cookie)[1])['jobs']
            if jobs[0]['state'] != 'creating': break
            time.sleep(.03)
        self.assertEqual(jobs[0]['state'], 'ready')
        status, body, headers = self.call(f'/renju/admin/data/exports/{job_id}/download', cookie)
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/gzip')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(int(headers['Content-Length']), len(body))
        self.assertIn('attachment', headers['Content-Disposition'])
        with tarfile.open(fileobj=io.BytesIO(body)) as archive:
            self.assertIn('manifest.json', archive.getnames())
        self.assertEqual(self.call('/data/statistics?undo=yes&undo=no', cookie)[0], 400)


if __name__ == '__main__':
    unittest.main()
