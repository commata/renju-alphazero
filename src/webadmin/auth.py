"""One admin, bounded in-memory sessions and login throttling; no game dependencies."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import hmac
import secrets
import threading
import time

IDLE = 30 * 60
LIFETIME = 12 * 60 * 60
LOCKOUT = 15 * 60
MAX_FAILURES = 5
MAX_SESSIONS = 128
MAX_CLIENTS = 4096


def hash_password(password: str) -> str:
    if not password or len(password.encode('utf-8')) > 1024:
        raise ValueError('password must be 1..1024 UTF-8 bytes')
    salt = secrets.token_bytes(16)
    digest = _derive(password, salt)
    return f'scrypt$16384$8$1${salt.hex()}${digest.hex()}'


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode('utf-8'), salt=salt, n=16384, r=8, p=1, dklen=32,
                          maxmem=64 * 1024 * 1024)


def parse_hash(value: str) -> tuple[bytes, bytes]:
    parts = value.split('$')
    if len(parts) != 6 or parts[:4] != ['scrypt', '16384', '8', '1']:
        raise ValueError('invalid admin password hash; generate it with --hash-password')
    try:
        salt, digest = bytes.fromhex(parts[4]), bytes.fromhex(parts[5])
    except ValueError:
        raise ValueError('invalid admin password hash') from None
    if len(salt) != 16 or len(digest) != 32:
        raise ValueError('invalid admin password hash lengths')
    return salt, digest


class Auth:
    def __init__(self, username: str, password_hash: str, *, clock=time.monotonic):
        if not username or len(username.encode('utf-8')) > 256:
            raise ValueError('ADMIN_USER must be 1..256 UTF-8 bytes')
        self.username = username.encode('utf-8')
        self.salt, self.digest = parse_hash(password_hash)
        self.clock = clock
        self.lock = threading.Lock()
        self.sessions: dict[str, tuple[float, float]] = {}
        self.failures: OrderedDict[str, tuple[int, float]] = OrderedDict()

    def _prune(self, now: float) -> None:
        self.sessions = {key: (created, seen) for key, (created, seen) in self.sessions.items()
                         if now - created < LIFETIME and now - seen < IDLE}
        for ip, (_, last) in list(self.failures.items()):
            if now - last >= LOCKOUT:
                del self.failures[ip]

    def login(self, ip: str, username: str, password: str) -> tuple[str | None, bool]:
        """Return (token, locked). Serialize scrypt work so concurrent logins cannot exhaust memory."""
        with self.lock:
            now = self.clock()
            self._prune(now)
            count, _ = self.failures.get(ip, (0, now))
            if count >= MAX_FAILURES or (ip not in self.failures and len(self.failures) >= MAX_CLIENTS):
                return None, True
            # Always derive a hash, even for an unknown user.
            candidate = _derive(password, self.salt)
            valid = hmac.compare_digest(candidate, self.digest)
            valid &= hmac.compare_digest(username.encode('utf-8'), self.username)
            if not valid:
                self.failures[ip] = (count + 1, now)
                return None, count + 1 >= MAX_FAILURES
            self.failures.pop(ip, None)
            if len(self.sessions) >= MAX_SESSIONS:
                del self.sessions[min(self.sessions, key=lambda key: self.sessions[key][1])]
            token = secrets.token_urlsafe(32)
            self.sessions[token] = (now, now)
            return token, False

    def check(self, token: str | None) -> bool:
        with self.lock:
            now = self.clock()
            self._prune(now)
            if not token or token not in self.sessions:
                return False
            created, _ = self.sessions[token]
            self.sessions[token] = (created, now)
            return True

    def logout(self, token: str | None) -> None:
        with self.lock:
            self.sessions.pop(token, None)
