"""Append-only confirmation bindings, shared across output paths; no model or result mutation.

The files live under runs/h7r/confirmations, outside committed evidence. A process
lock protects the read/compare/create transaction and atomic replacement survives
interruption. Identical recomputation is allowed; changed inputs are not.

A local discipline device: it binds this checkout only. A fresh clone or another machine
starts with an empty ledger, so the folder is kept with the confirmation's source records
(docs/h7-rl-plan.md 9.7) and every verdict names the entries and their SHA-256.
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


REPOSITORY = Path(__file__).resolve().parents[2]
FOLDER = 'runs/h7r/confirmations'


def _folder(root) -> Path:
    return Path(root) / FOLDER


def entry_name(category: str, key: str) -> str:
    return hashlib.sha256(f'{category}:{key}'.encode()).hexdigest()


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')


def guard(root, allow_dirty: bool) -> None:
    """``--allow-dirty`` (tests only) never writes the repository's own ledger: tests pass a temporary root."""
    if allow_dirty and _folder(root).resolve() == _folder(REPOSITORY).resolve():
        raise ValueError(f'--allow-dirty never writes the confirmation ledger of {REPOSITORY} (tests use a '
                         'temporary root)')


def _check(folder: Path, bindings) -> tuple[list[str], list[tuple]]:
    """Conflicts (an entry bound to another value, or one key twice in ``bindings``) and the entries still unwritten."""
    seen, clash, todo = {}, [], []
    for category, key, value in bindings:
        name = entry_name(category, key)
        if name in seen:
            if seen[name] != value:
                clash.append(f'{category}:{key} twice with different values')
            continue
        seen[name] = value
        path = folder / f'{name}.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != value:
                clash.append(f'{category}:{key}')
        else:
            todo.append((name, category, key, value))
    return clash, todo


def conflicts(root, bindings) -> list[str]:
    """Read-only: the bindings that the ledger already holds with another value (nothing is written)."""
    folder = _folder(root)
    if not folder.exists():
        return []
    with file_lock(folder / '.lock'):
        return _check(folder, bindings)[0]


def bind_all(root, bindings, allow_dirty: bool = False) -> list[dict]:
    """Bind every ``(category, key, value)`` in one lock: all conflicts are checked before any entry is written, so a
    refusal leaves the ledger unchanged. Identical existing entries are kept (audit recomputation). Returns each
    entry's identifier and the SHA-256 of its stored bytes."""
    guard(root, allow_dirty)
    folder = _folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / '.lock'):
        clash, todo = _check(folder, bindings)
        if clash:
            raise ValueError(f'already bound to different confirmation inputs: {", ".join(clash)}')
        for name, _, _, value in todo:
            path = folder / f'{name}.json'
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as out:
                out.write(_encode(value))
                out.flush()
                os.fsync(out.fileno())
            temporary.replace(path)
        entries, seen = [], set()
        for category, key, _ in bindings:
            name = entry_name(category, key)
            if name not in seen:
                seen.add(name)
                entries.append({'category': category, 'key': key, 'entry': f'{FOLDER}/{name}.json',
                                'sha256': hashlib.sha256((folder / f'{name}.json').read_bytes()).hexdigest()})
        return entries


def bind(root: Path, category: str, key: str, value: dict, allow_dirty: bool = False) -> None:
    bind_all(root, [(category, key, value)], allow_dirty)


def contract_bindings(contract, sha) -> list[tuple]:
    out = [('contract', contract['contract_id'], {'sha256': sha})]
    for seed in sorted({a['seed'] for a in contract.get('arms', [])}):
        out.append(('seed', str(seed), {'contract_id': contract['contract_id'], 'sha256': sha}))
    return out


def bind_contract(root, contract, sha, allow_dirty: bool = False):
    return bind_all(root, contract_bindings(contract, sha), allow_dirty)


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
