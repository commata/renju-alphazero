"""Browser sessions of the web server (docs/web-ser5.md §8).

Each browser gets its own game behind the cookie ``renju_sid``: a random id from
``secrets.token_urlsafe(32)``, sent ``HttpOnly; SameSite=Lax; Path=/`` (and ``Secure`` behind
HTTPS). The id is a bearer secret, so it is never stored or logged: the database and the logs see
``session_hash(id)`` only.

Sessions live in memory. Idle ones expire after ``ttl_seconds`` (lazy cleanup on every lookup), but
never while the AI is searching or queued for them; an expired session is closed (``close()``: its
game is marked abandoned). A server restart loses them all (the database keeps the games).
"""
from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable

COOKIE = 'renju_sid'


class SessionLimit(Exception):
    """No room for a new session (every one of ``max_sessions`` is still alive)."""


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:16]


def cookie_header(session_id: str, *, secure: bool, max_age: int | None = None) -> str:
    parts = [f'{COOKIE}={session_id}', 'Path=/', 'HttpOnly', 'SameSite=Lax']
    if max_age is not None:
        parts.append(f'Max-Age={max_age}')
    if secure:
        parts.append('Secure')
    return '; '.join(parts)


def session_id_from_cookie(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(';'):
        name, _, value = part.strip().partition('=')
        if name == COOKIE and value:
            return value
    return None


@dataclass
class _Entry:
    session: Any
    last_seen: float
    hash: str = field(default='')


class SessionManager:
    """``factory(session_hash)`` makes a session; it must have ``busy()`` and ``close()``."""

    def __init__(self, factory: Callable[[str], Any], *, max_sessions: int = 16, ttl_seconds: float = 3600.0,
                 clock: Callable[[], float] = monotonic):
        if max_sessions < 1 or ttl_seconds <= 0:
            raise ValueError('need max_sessions >= 1 and ttl_seconds > 0')
        self.factory, self.max_sessions, self.ttl, self.clock = factory, max_sessions, ttl_seconds, clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self.expired = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, session_id: str | None):
        """The live session of ``session_id`` (and mark it seen), else None."""
        expired = []
        try:
            with self._lock:
                expired = self._cleanup_locked()
                entry = self._entries.get(session_id) if session_id else None
                if entry is None:
                    return None
                entry.last_seen = self.clock()
                return entry.session
        finally:
            self._close(expired)

    def get_or_create(self, session_id: str | None) -> tuple[str, Any, bool]:
        """``(id, session, created)``; a new id when ``session_id`` is unknown or expired."""
        expired = []
        try:
            with self._lock:
                expired = self._cleanup_locked()
                entry = self._entries.get(session_id) if session_id else None
                if entry is not None:
                    entry.last_seen = self.clock()
                    return session_id, entry.session, False
                if len(self._entries) >= self.max_sessions:
                    raise SessionLimit(f'{len(self._entries)} sessions in use')
                new_id = new_session_id()
                digest = session_hash(new_id)
                self._entries[new_id] = _Entry(self.factory(digest), self.clock(), digest)
                return new_id, self._entries[new_id].session, True
        finally:
            self._close(expired)

    def sessions(self) -> list:
        with self._lock:
            return [entry.session for entry in self._entries.values()]

    def close_all(self) -> None:
        with self._lock:
            entries, self._entries = list(self._entries.values()), {}
        self._close([e.session for e in entries])

    def _cleanup_locked(self) -> list:
        now = self.clock()
        stale = [sid for sid, e in self._entries.items() if now - e.last_seen > self.ttl and not e.session.busy()]
        self.expired += len(stale)
        return [self._entries.pop(sid).session for sid in stale]

    @staticmethod
    def _close(sessions) -> None:
        for session in sessions:  # outside the manager lock: closing may write logs
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 - one bad session must not break the others
                print(f'[web] closing an expired session failed: {type(exc).__name__}: {exc}', flush=True)
