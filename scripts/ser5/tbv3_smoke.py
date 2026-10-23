"""Smoke test of a running web server's latest-certified opponent (TB-V3 / fast5, CPU value).

    # A. before deploying: a temporary local server WITHOUT --secure-cookie
    python scripts/run_web_play.py --port 8001 --tbv3 --ai-slots 1 &
    python scripts/ser5/tbv3_smoke.py --base http://127.0.0.1:8001
    # B. the production service (--secure-cookie) through Caddy's HTTPS
    python scripts/ser5/tbv3_smoke.py --base https://ser5tack.duckdns.org --user USER --password PASS

Checks, against the server as deployed (stdlib only, changes nothing but one game):
    /api/health   ok, the default opponent is "tbv3", its engine TB-V3 / fast5 / cpu / latest-certified
    /api/engine   the engine, runtime and model identities (full SHA-256) the server checked at start
    session       /api/new sets the session cookie, the cookie can travel over this scheme (a Secure cookie
                  never goes back over http://: refused at once, not after a timeout), and the next request
                  sees the same game (its seed)
    one game      the human plays black's centre (placed by the opening rule) and then the nearest free point;
                  every AI reply comes from the engine (no error), and each reply's time is printed
Exit status 1 on the first failed check, with the session state and the server's AI queue counts.

Times: 30 s is the engine's soft monitor (logged by the server, never cut); ``--timeout`` (120 s) is this
script's limit for one reply, a performance timeout, not an engine failure (the worker may still be searching);
300 s is the server's hang guard (the worker is replaced and the move is an error).
"""
from __future__ import annotations

import argparse
import base64
import http.cookiejar
import json
import random
import sys
import time
import urllib.parse
import urllib.request

COOKIE = 'renju_sid'  # webplay.sessions.COOKIE


class SmokeFailure(Exception):
    pass


class Client:
    def __init__(self, base: str, user: str | None = None, password: str | None = None):
        self.base = base.rstrip('/')
        self.scheme = urllib.parse.urlsplit(self.base).scheme
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.auth = None
        if user is not None:
            self.auth = 'Basic ' + base64.b64encode(f'{user}:{password or ""}'.encode()).decode()

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        if self.auth:
            headers['Authorization'] = self.auth
        request = urllib.request.Request(self.base + path, data=data, method='GET' if body is None else 'POST',
                                         headers=headers)
        with self.opener.open(request, timeout=30) as response:
            return json.loads(response.read()), response.headers.get_all('Set-Cookie') or []

    def session_cookie(self):
        return next((c for c in self.jar if c.name == COOKIE), None)


def check_session(client: Client, set_cookies: list[str], state: dict) -> str:
    """The cookie /api/new set must come back on the next request: same game (seed) seen there."""
    cookie = client.session_cookie()
    header = next((h for h in set_cookies if h.split('=', 1)[0].strip() == COOKIE), None)
    if cookie is None and header is None:
        raise SmokeFailure('session: the server set no session cookie')
    secure = (cookie.secure if cookie is not None else
              any(p.strip().lower() == 'secure' for p in (header or '').split(';')))
    if secure and client.scheme != 'https':
        raise SmokeFailure('session: the server sets a Secure session cookie, which a client never sends back over '
                           f'{client.scheme}:// (every request would open a new session); use the HTTPS address, '
                           'or a local smoke server without --secure-cookie')
    again, _ = client.call('/api/state')
    if (again.get('seed'), again.get('agent_key')) != (state.get('seed'), state.get('agent_key')):
        raise SmokeFailure(f"session: the next request saw another game (seed {again.get('seed')}, expected "
                           f"{state.get('seed')}): the session cookie did not come back")
    return 'Secure' if secure else 'not Secure'


def snapshot(client: Client, state: dict | None) -> str:
    lines = []
    if state is not None:
        lines.append(f"state: {state.get('status')}, move_count={state.get('move_count')}, "
                     f"to_play={state.get('to_play')}, last_error={state.get('last_error')!r}")
    try:
        health, _ = client.call('/api/health')
        ai = health.get('ai', {})
        lines.append(f"queue: running={ai.get('running')} queued={ai.get('queued')} done={ai.get('done')} "
                     f"failed={ai.get('failed')}; sessions={health.get('sessions')}, "
                     f"secure_cookie={health.get('secure_cookie')}")
        if state is not None and ai.get('done') and not ai.get('running') and state.get('status') == 'READY':
            lines.append('hint: the server completed AI jobs that this client did not observe: '
                         'check session-cookie continuity')
    except Exception as exc:  # noqa: BLE001 - the snapshot is best effort
        lines.append(f'health: unavailable ({type(exc).__name__}: {exc})')
    return '\n'.join('      ' + line for line in lines)


def run(args) -> int:
    client = Client(args.base, args.user, args.password)
    state = None
    try:
        health, _ = client.call('/api/health')
        engine = health.get('engine') or {}
        if not health.get('ok') or health.get('default_agent') != 'tbv3':
            raise SmokeFailure(f"health: default opponent {health.get('default_agent')!r}, expected 'tbv3'")
        want = {'engine': 'TB-V3', 'runtime': 'fast5', 'value_device': 'cpu', 'model': 'latest-certified'}
        if {k: engine.get(k) for k in want} != want:
            raise SmokeFailure(f'health: engine {engine}')
        print(f"PASS  health: {engine}, AI slots {health['ai'].get('slots')}, "
              f"secure_cookie {health.get('secure_cookie')}", flush=True)
        details, _ = client.call('/api/engine')
        print(f"PASS  engine: {details['engine']} engine {details['engine_config_hash'][:12]} runtime "
              f"{details['runtime_hash'][:12]} policy {details['policy_checkpoint_sha256'][:12]} "
              f"value {details['value_checkpoint_sha256'][:12]}", flush=True)

        seed = random.SystemRandom().randrange(1, 2 ** 31)
        state, set_cookies = client.call('/api/new', {'agent': 'tbv3', 'human_color': 'BLACK', 'seed': seed})
        kind = check_session(client, set_cookies, state)
        print(f'PASS  session: cookie {kind}, sent back over {client.scheme}://, same game on the next request',
              flush=True)
        times = []
        for _ in range(args.moves):
            count = state['move_count']
            end = time.monotonic() + args.timeout
            while state['to_play'] != state['human_color'] or state['status'] in ('QUEUED', 'THINKING'):
                if state.get('last_error') or state['done'] or time.monotonic() > end:
                    break
                time.sleep(0.5)
                state, _ = client.call('/api/state')
            if state.get('last_error'):
                raise SmokeFailure(f"AI reply: {state['last_error']}")
            if state['done']:
                print(f"PASS  game over after {state['move_count']} moves ({state['winner']})", flush=True)
                break
            if state['move_count'] <= count:
                raise SmokeFailure(f'no AI reply within {args.timeout:g} s: a smoke performance timeout, not an '
                                   f'engine failure (the worker may still be searching; the server replaces a '
                                   f'worker only after its 300 s hang guard)')
            times.append(state['last_ai_seconds'])
            print(f"PASS  AI reply {len(times)}: {state['last_ai_move']} in {state['last_ai_seconds']:.2f} s "
                  f"({state['agent_name']})", flush=True)
            last = state['last_move']
            taken = {tuple(m) for m in state['history']}
            forbidden = {tuple(p[:2]) for p in state.get('forbidden', [])}
            free = sorted(((r, c) for r in range(15) for c in range(15) if (r, c) not in taken | forbidden),
                          key=lambda m: (max(abs(m[0] - last[0]), abs(m[1] - last[1])), m))
            state, _ = client.call('/api/move', {'row': free[0][0], 'col': free[0][1]})
        if times:
            print(f'PASS  {len(times)} AI replies, max {max(times):.2f} s, mean {sum(times) / len(times):.2f} s',
                  flush=True)
        return 0
    except SmokeFailure as exc:
        print(f'FAIL  {exc}\n{snapshot(client, state)}', flush=True)
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='http://127.0.0.1:8001',
                        help='server root (include a /renju prefix); http:// only for a server without --secure-cookie')
    parser.add_argument('--user', help="Caddy Basic Auth user (production through HTTPS)")
    parser.add_argument('--password', help='Caddy Basic Auth password')
    parser.add_argument('--moves', type=int, default=6, help='AI replies to wait for')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='seconds to wait for one AI reply (smoke performance limit; the hang guard is 300 s)')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
