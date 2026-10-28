"""Append-only confirmation bindings, shared across output paths; no model or result mutation.

The files live under runs/h7r/confirmations, outside committed evidence. A process
lock protects the read/compare/create transaction and atomic replacement survives
interruption. Identical recomputation is allowed; changed inputs are not.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os


@contextmanager
def file_lock(path, nonblocking=False):
    """A local cross-process lock; keep basic output commands portable to Windows."""
    with Path(path).open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
            try:
                msvcrt.locking(lock.fileno(), mode, 1)
            except OSError as exc:
                raise ValueError(f'confirmation directory is already running: {path}') from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            except BlockingIOError:
                raise ValueError(f'confirmation directory is already running: {path}') from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def bind(root: Path, category: str, key: str, value: dict) -> None:
    folder = Path(root) / 'runs/h7r/confirmations'
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(f'{category}:{key}'.encode()).hexdigest()
    path = folder / f'{name}.json'
    with file_lock(folder / '.lock'):
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f'{category} already bound to different confirmation inputs: {key}')
            return
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as out:
            json.dump(value, out, sort_keys=True, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)


def bind_contract(root, contract, sha):
    bind(root, 'contract', contract['contract_id'], {'sha256': sha})
    for seed in sorted({a['seed'] for a in contract.get('arms', [])}):
        bind(root, 'seed', str(seed), {'contract_id': contract['contract_id'], 'sha256': sha})


def publish_json(path, value):
    """Exclusive, atomic publication: no partial JSON and no concurrent overwrite."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            json.dump(value, out, indent=1, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
