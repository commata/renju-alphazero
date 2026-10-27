"""Stage 6: consistent exports in a separate, low-priority process.

A filesystem flock spans the worker lifetime, including administrator restarts.
Only generated UUID names can be downloaded or removed by retention. A completed
archive is published atomically; no partially generated file is ever served.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import stat
import sys
import tarfile
import threading
import time
import uuid

from weblog.backup import snapshot_database
from weblog.store import SCHEMA_VERSION, open_readonly
from .inspect import checkout_commit

JOB_ID = re.compile(r'[0-9a-f]{32}')
TABLE_ORDER = {'games': 'started_at, game_id', 'moves': 'game_id, seq', 'veto_positions': 'game_id, ply, id'}
MARGIN = 64 * 1024 * 1024


class ExportBusy(RuntimeError):
    pass


def _write_json(path: Path, data: dict) -> None:
    pending = path.with_suffix('.json.part')
    with pending.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    pending.replace(path)


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def _raw_files(conn: sqlite3.Connection, log_root: Path) -> tuple[list[tuple[Path, Path]], int]:
    """Only completed snapshot games' JSON/CSV logs, strictly below the configured log root.

    Never follow symlinks, arbitrary DB paths or include config/credential files.
    Raw logs are optional ancillary files; only the DB/SQL/CSV share the snapshot.
    """
    files, skipped = {}, 0
    root = log_root.resolve()
    for (raw,) in conn.execute("SELECT raw_log_dir FROM games WHERE status='finished' AND raw_log_dir IS NOT NULL"):
        path = Path(raw)
        if not path.is_absolute():
            try:
                path = log_root / path.relative_to('logs/web_play')
            except ValueError:
                skipped += 1
                continue
        try:
            relative = path.relative_to(log_root)
            if not relative.parts or '..' in relative.parts:
                raise ValueError('invalid raw directory')
            chain = [log_root.joinpath(*relative.parts[:i]) for i in range(len(relative.parts) + 1)]
            if any(p.is_symlink() for p in chain) or not path.is_dir() or not path.resolve().is_relative_to(root):
                raise ValueError('unsafe or missing raw directory')
            for name in ('game.json', 'moves.csv'):
                source = path / name
                if source.is_file() and not source.is_symlink():
                    files[Path('logs/web_play') / relative / name] = source
                else:
                    skipped += 1
        except (OSError, ValueError):
            skipped += 1
    return [(source, relative) for relative, source in sorted(files.items())], skipped


def build_export(db: Path, root: Path, job_id: str, *, repo: Path, log_root: Path,
                 include_raw_logs: bool = False) -> dict:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError('invalid export id')
    work = root / f'{job_id}.tmp'
    work.mkdir(mode=0o700)
    archive = root / f'{job_id}.tar.gz'
    created = datetime.now(timezone.utc).isoformat()
    try:
        # SQL + CSV + raw copies + archive can be several times the database size.
        size = db.stat().st_size + (Path(str(db) + '-wal').stat().st_size if Path(str(db) + '-wal').exists() else 0)
        if shutil.disk_usage(root).free < size * 8 + MARGIN:
            raise OSError('insufficient disk space for export')
        snapshot = work / 'renju_web_full.db'
        snapshot_database(db, snapshot)
        snapshot_completed = datetime.now(timezone.utc).isoformat()
        with closing(open_readonly(snapshot)) as conn:
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f'expected schema {SCHEMA_VERSION}, found {version}')
            counts = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in TABLE_ORDER}
            with (work / 'renju_web_full.sql').open('w', encoding='utf-8') as handle:
                for line in conn.iterdump():
                    handle.write(line + '\n')
                # iterdump does not preserve this application schema version by itself.
                handle.write(f'PRAGMA user_version={version};\n')
            for table, order in TABLE_ORDER.items():
                cursor = conn.execute(f'SELECT * FROM {table} ORDER BY {order}')
                with (work / f'{table}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow([column[0] for column in cursor.description])
                    written = 0
                    for row in cursor:
                        writer.writerow(row)
                        written += 1
                if written != counts[table]:
                    raise ValueError(f'CSV row count mismatch: {table}')
            raw_files, skipped = _raw_files(conn, log_root) if include_raw_logs else ([], 0)
        raw_size = sum(source.stat().st_size for source, _ in raw_files)
        if shutil.disk_usage(root).free < raw_size * 2 + snapshot.stat().st_size * 4 + MARGIN:
            raise OSError('insufficient disk space for raw logs and archive')
        for source, relative in raw_files:
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Recheck immediately before copying; O_NOFOLLOW also protects the leaf.
            with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        files = sorted(path for path in work.rglob('*') if path.is_file())
        manifest = {'format': 'renju-admin-export-v1', 'created_at': created,
                    'snapshot_completed_at': snapshot_completed,
                    'schema_version': version, 'counts': counts, 'server_commit': checkout_commit(repo),
                    'raw_logs_requested': include_raw_logs, 'raw_logs_included': len(raw_files),
                    'raw_logs_skipped': skipped,
                    'files': {p.relative_to(work).as_posix(): _fingerprint(p) for p in files}}
        _write_json(work / 'manifest.json', manifest)
        # Atomic rename on the same filesystem; no symlinks or host paths in the archive.
        pending = work / 'archive.part'
        with tarfile.open(pending, 'w:gz', compresslevel=1) as tar:
            for path in [*files, work / 'manifest.json']:
                tar.add(path, arcname=path.relative_to(work).as_posix(), recursive=False)
        metadata = {'id': job_id, 'state': 'ready', 'created_at': created,
                    'include_raw_logs': include_raw_logs, 'counts': counts, 'schema_version': version,
                    'server_commit': manifest['server_commit'], 'raw_logs_included': len(raw_files),
                    'raw_logs_skipped': skipped, **_fingerprint(pending)}
        pending.replace(archive)
        return metadata
    finally:
        shutil.rmtree(work, ignore_errors=True)


class ExportManager:
    def __init__(self, db: Path, root: Path, *, repo: Path, log_root: Path, keep: int = 5):
        if keep < 1:
            raise ValueError('export keep must be positive')
        self.db, self.root = db.resolve(), root.resolve()
        self.repo, self.log_root, self.keep = repo.resolve(), log_root.resolve(), keep
        self.lock = threading.Lock()

    def _prepare(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _lock_file(self):
        fd = os.open(self.root / '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise ExportBusy('백업 생성이 이미 진행 중입니다.') from None
        return fd

    def _read(self) -> list[dict]:
        jobs = []
        for path in self.root.glob('*.json'):
            if not JOB_ID.fullmatch(path.stem) or path.is_symlink():
                continue
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if (job.get('id') == path.stem and job.get('state') in ('creating', 'ready', 'failed')
                        and isinstance(job.get('created_at'), str)):
                    jobs.append(job)
            except (ValueError, OSError, AttributeError):
                continue
        return sorted(jobs, key=lambda job: job.get('created_at', ''), reverse=True)

    def _cleanup(self):
        """Called only with the filesystem lock: no worker's temporary files can be removed."""
        for path in self.root.glob('*.tmp'):
            if JOB_ID.fullmatch(path.stem) and not path.is_symlink() and time.time() - path.stat().st_mtime > 86400:
                shutil.rmtree(path)
        for path in self.root.glob('*.json.part'):
            if JOB_ID.fullmatch(path.name[:-len('.json.part')]) and time.time() - path.stat().st_mtime > 86400:
                path.unlink(missing_ok=True)
        ready, failed = 0, 0
        for job in self._read():
            path = self.root / f'{job["id"]}.json'
            if job['state'] == 'creating':
                job.update(state='failed', error='백업 프로세스가 중단되었습니다. 다시 생성하세요.')
                _write_json(path, job)
            if job['state'] == 'ready':
                ready += 1
                remove = ready > self.keep
            else:
                failed += 1
                remove = failed > self.keep
            if remove:
                (self.root / f'{job["id"]}.tar.gz').unlink(missing_ok=True)
                path.unlink(missing_ok=True)

    def list_jobs(self) -> dict:
        with self.lock:
            self._prepare()
            try:
                fd = self._lock_file()
            except ExportBusy:
                busy = True
            else:
                try:
                    self._cleanup()
                finally:
                    os.close(fd)
                busy = False
            return {'jobs': self._read(), 'busy': busy, 'keep': self.keep}

    def start(self, *, include_raw_logs: bool = False) -> dict:
        with self.lock:
            self._prepare()
            fd = self._lock_file()
            job_id = uuid.uuid4().hex
            job = {'id': job_id, 'state': 'creating', 'created_at': datetime.now(timezone.utc).isoformat(),
                   'include_raw_logs': include_raw_logs}
            try:
                self._cleanup()
                _write_json(self.root / f'{job_id}.json', job)
                env = {key: value for key, value in os.environ.items() if not key.startswith('ADMIN_')}
                env['PYTHONPATH'] = str(self.repo / 'src')
                command = [sys.executable, '-m', 'webadmin.exports', '--db', str(self.db), '--root', str(self.root),
                           '--repo', str(self.repo), '--log-root', str(self.log_root), '--id', job_id,
                           '--lock-fd', str(fd)]
                if include_raw_logs:
                    command.append('--raw-logs')
                child = subprocess.Popen(command, pass_fds=(fd,), cwd=self.repo, env=env, stdout=subprocess.DEVNULL)
                threading.Thread(target=child.wait, name='admin-export-reaper', daemon=True).start()
            except Exception:
                job.update(state='failed', error='백업 프로세스를 시작하지 못했습니다.')
                _write_json(self.root / f'{job_id}.json', job)
                raise
            finally:
                os.close(fd)  # worker keeps its inherited descriptor until completion
            return job

    def open_download(self, job_id: str):
        if not JOB_ID.fullmatch(job_id):
            raise ValueError('invalid export id')
        path = self.root / f'{job_id}.json'
        try:
            if path.is_symlink():
                raise FileNotFoundError
            job = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(job, dict) or job.get('id') != job_id or job.get('state') != 'ready':
                raise FileNotFoundError
            fd = os.open(self.root / f'{job_id}.tar.gz', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise LookupError('백업 파일이 올바르지 않습니다.')
            return os.fdopen(fd, 'rb'), job
        except (FileNotFoundError, ValueError):
            raise LookupError('완료된 백업을 찾을 수 없습니다.') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('db', 'root', 'repo', 'log-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    parser.add_argument('--raw-logs', action='store_true')
    args = parser.parse_args()
    if not JOB_ID.fullmatch(args.id):
        parser.error('invalid export id')
    os.umask(0o077)
    try:
        os.nice(10)
        job = build_export(args.db, args.root, args.id, repo=args.repo, log_root=args.log_root,
                           include_raw_logs=args.raw_logs)
        _write_json(args.root / f'{args.id}.json', job)
        return 0
    except Exception as exc:
        _write_json(args.root / f'{args.id}.json', {'id': args.id, 'state': 'failed',
                    'created_at': datetime.now(timezone.utc).isoformat(), 'include_raw_logs': args.raw_logs,
                    'error': f'백업 생성 실패 ({type(exc).__name__}). 디스크 공간과 서비스 로그를 확인하세요.'})
        print(f'admin export {args.id}: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        os.close(args.lock_fd)


if __name__ == '__main__':
    sys.exit(main())
