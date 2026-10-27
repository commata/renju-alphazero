"""Shared online SQLite snapshot used by the timer and administrator exports."""
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from .store import open_readonly


def snapshot_database(source: Path, target: Path, *, timeout: float = 120) -> None:
    """Create a new, verified standalone DB. Never overwrite a file or modify the source."""
    started = time.monotonic()

    def progress(*_):
        if time.monotonic() - started > timeout:
            raise TimeoutError('SQLite backup exceeded its time budget')

    with closing(open_readonly(source)) as reader:
        # Exclusive creation prevents an accidental overwrite, including a symlink target.
        with target.open('xb'):
            pass
        try:
            with closing(sqlite3.connect(target)) as copy:
                reader.backup(copy, pages=256, progress=progress, sleep=.05)
                if [row[0] for row in copy.execute('PRAGMA integrity_check')] != ['ok']:
                    raise ValueError('SQLite snapshot integrity_check failed')
                copy.execute('PRAGMA journal_mode=DELETE')  # archive is usable without WAL sidecars
        except BaseException:
            target.unlink(missing_ok=True)
            raise
