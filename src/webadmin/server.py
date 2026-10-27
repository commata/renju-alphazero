"""Loopback-only admin HTTP server. Caddy strips /renju/admin before proxying.

Only login assets and POST /login are public. All other routes require a session,
including unknown data/export paths. No game module or torch is imported.
"""
from __future__ import annotations

from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from .auth import Auth
from .inspect import inspect_database

PREFIX = '/renju/admin'
COOKIE = 'renju_admin'
PUBLIC_ASSETS = {'/login.css': ('style.css', 'text/css; charset=utf-8'),
                 '/login.js': ('login.js', 'text/javascript; charset=utf-8')}
PRIVATE_ASSETS = {'/admin.js': ('admin.js', 'text/javascript; charset=utf-8')}


def validate_origin(origin: str) -> str:
    parts = urlsplit(origin)
    if (parts.scheme not in ('https', 'http') or not parts.hostname or parts.username or parts.password
            or parts.path or parts.query or parts.fragment or parts.netloc.endswith(':')):
        raise ValueError('ADMIN_ORIGIN must be a canonical origin without a trailing slash or path')
    try:
        port = parts.port
    except ValueError:
        raise ValueError('invalid origin port') from None
    if parts.scheme == 'http' and parts.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('HTTP is permitted only for local testing; use HTTPS in production')
    if port == 0:
        raise ValueError('invalid origin port')
    return origin


class AdminServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, *, db: Path, origin: str, auth: Auth, assets: Path):
        if address[0] != '127.0.0.1':
            raise ValueError('admin must bind to 127.0.0.1')
        self.origin = validate_origin(origin)
        self.secure = self.origin.startswith('https://')
        self.db, self.auth, self.assets = db, auth, assets
        super().__init__(address, Handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(10)
        return request, address


def game_status() -> dict:
    """Only fixed loopback destinations; no redirects, proxies or browser credentials."""
    from urllib.request import HTTPRedirectHandler

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = build_opener(ProxyHandler({}), NoRedirect())
    result = {}
    for name in ('health', 'engines'):
        try:
            with opener.open(f'http://127.0.0.1:8000/api/{name}', timeout=2) as response:
                data = response.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise ValueError('response too large')
                result[name] = json.loads(data)
        except (OSError, ValueError):
            result[name] = {'available': False}
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = 'RenjuAdmin/1.0'

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        super().end_headers()

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler's error pages also get no-store and omit request data.
        self._json({'error': 'request rejected'}, code)

    def _json(self, value, status=200, headers=None):
        self._send(json.dumps(value, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8',
                   status, headers)

    def _send(self, body: bytes, content_type: str, status=200, headers=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _asset(self, filename, content_type):
        # Filename comes only from fixed route maps, never the request path.
        self._send((self.server.assets / filename).read_bytes(), content_type)

    def _token(self):
        try:
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            return cookie[COOKIE].value if COOKIE in cookie else None
        except CookieError:
            return None

    def _cookie(self, value: str, *, clear=False):
        result = f'{COOKIE}={value}; Path={PREFIX}; HttpOnly; SameSite=Strict'
        if self.server.secure:
            result += '; Secure'
        if clear:
            result += '; Max-Age=0'
        return result

    def _authenticated(self):
        return self.server.auth.check(self._token())

    def _client_ip(self):
        # Caddy appends the connecting client's address; do not trust a user-supplied first entry.
        peer = self.client_address[0]
        if peer == '127.0.0.1':
            forwarded = self.headers.get('X-Forwarded-For', '').split(',')[-1].strip()
            try:
                return str(ipaddress.ip_address(forwarded))
            except ValueError:
                pass
        return peer

    def _path(self):
        path = urlsplit(self.path).path
        if path == PREFIX or path.startswith(PREFIX + '/'):
            return path[len(PREFIX):] or '/'
        return path

    def do_GET(self):
        if len(self.path) > 2048:
            self._json({'error': 'request too long'}, 414)
            return
        path = self._path()
        try:
            if path == '/login':
                self._asset('login.html', 'text/html; charset=utf-8')
                return
            if path in PUBLIC_ASSETS:
                self._asset(*PUBLIC_ASSETS[path])
                return
            if not self._authenticated():
                if path == '/':
                    self._send(b'', 'text/plain', 302, {'Location': PREFIX + '/login'})
                else:
                    self._json({'error': 'login required'}, 401)
                return
            if path == '/':
                self._asset('index.html', 'text/html; charset=utf-8')
            elif path in PRIVATE_ASSETS:
                self._asset(*PRIVATE_ASSETS[path])
            elif path == '/data/status':
                self._json({'database': inspect_database(self.server.db), 'game_server': game_status()})
            else:
                self._json({'error': 'not found'}, 404)
        except (OSError, sqlite3.Error, ValueError):
            self._json({'error': '관리자 데이터를 읽을 수 없습니다. DB 경로·스키마·서비스 상태를 확인하세요.'}, 503)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if len(self.path) > 2048:
            self._json({'error': 'request too long'}, 414)
            return
        path = self._path()
        if path != '/login' and not self._authenticated():
            self._json({'error': 'login required'}, 401)
            return
        # Compare against a configured canonical origin, never forwarded Host/Proto.
        if self.headers.get_all('Origin') != [self.server.origin]:
            self._json({'error': 'origin rejected'}, 403)
            return
        if path not in ('/login', '/logout'):
            self._json({'error': 'not found'}, 404)
            return
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length') or []) != 1:
            self._json({'error': 'Content-Length required'}, 400)
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self._json({'error': 'JSON required'}, 415)
            return
        try:
            size = int(self.headers['Content-Length'])
            if not 0 < size <= 4096:
                raise ValueError('body size')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('object required')
            if path == '/logout':
                self.server.auth.logout(self._token())
                self._json({'ok': True}, headers={'Set-Cookie': self._cookie('', clear=True)})
                return
            username, password = payload.get('username'), payload.get('password')
            if (not isinstance(username, str) or not isinstance(password, str)
                    or len(username.encode('utf-8')) > 256 or len(password.encode('utf-8')) > 1024):
                raise ValueError('invalid credentials shape')
        except (ValueError, UnicodeError):
            self._json({'error': 'invalid request'}, 400)
            return
        token, locked = self.server.auth.login(self._client_ip(), username, password)
        if token is None:
            headers = {'Retry-After': '900'} if locked else None
            self._json({'error': '로그인이 제한되었습니다. 15분 후 다시 시도하세요.' if locked
                        else '아이디 또는 비밀번호를 확인하세요.'}, 429 if locked else 401, headers)
            return
        # A fresh login always rotates the caller's existing session.
        self.server.auth.logout(self._token())
        self._json({'ok': True}, headers={'Set-Cookie': self._cookie(token)})

    def log_message(self, fmt, *args):
        # Never log submitted credentials, cookies, query strings or forwarded identities.
        pass
