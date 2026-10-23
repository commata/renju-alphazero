"""Smoke test of a running web server's latest-certified opponent (TB-V3 / fast5, CPU value).

    python scripts/ser5/tbv3_smoke.py --base http://127.0.0.1:8000 [--moves 6]

Checks, against the server as deployed (stdlib only, changes nothing but one game):
    /api/health   ok, the default opponent is "tbv3", its engine TB-V3 / fast5 / cpu / latest-certified
    /api/engine   the engine, runtime and model identities (full SHA-256) the server checked at start
    one game      the human plays black's centre (placed by the opening rule) and then the nearest free point;
                  every AI reply comes from the engine (no error), and each reply's time is printed
Exit status 1 on the first failed check.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='http://127.0.0.1:8000', help='server root (include a /renju prefix)')
    parser.add_argument('--moves', type=int, default=6, help='AI replies to wait for')
    parser.add_argument('--timeout', type=float, default=120.0, help='seconds to wait for one AI reply')
    args = parser.parse_args(argv)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    base = args.base.rstrip('/')

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(base + path, data=data, method='GET' if body is None else 'POST',
                                         headers={'Content-Type': 'application/json'} if body is not None else {})
        with opener.open(request, timeout=30) as response:
            return json.loads(response.read())

    def fail(message):
        print(f'FAIL  {message}', flush=True)
        return 1

    health = call('/api/health')
    engine = health.get('engine') or {}
    if not health.get('ok') or health.get('default_agent') != 'tbv3':
        return fail(f"health: default opponent {health.get('default_agent')!r}, expected 'tbv3'")
    want = {'engine': 'TB-V3', 'runtime': 'fast5', 'value_device': 'cpu', 'model': 'latest-certified'}
    if {k: engine.get(k) for k in want} != want:
        return fail(f'health: engine {engine}')
    print(f"PASS  health: {engine}, AI slots {health['ai'].get('slots')}", flush=True)
    details = call('/api/engine')
    print(f"PASS  engine: {details['engine']} engine {details['engine_config_hash'][:12]} runtime "
          f"{details['runtime_hash'][:12]} policy {details['policy_checkpoint_sha256'][:12]} "
          f"value {details['value_checkpoint_sha256'][:12]}", flush=True)

    state = call('/api/new', {'agent': 'tbv3', 'human_color': 'BLACK'})
    times = []
    for _ in range(args.moves):
        count = state['move_count']
        end = time.monotonic() + args.timeout
        while state['to_play'] != state['human_color'] or state['status'] in ('QUEUED', 'THINKING'):
            if state.get('last_error'):
                return fail(f"AI reply: {state['last_error']}")
            if state['done'] or time.monotonic() > end:
                break
            time.sleep(0.5)
            state = call('/api/state')
        if state.get('last_error'):
            return fail(f"AI reply: {state['last_error']}")
        if state['done']:
            print(f"PASS  game over after {state['move_count']} moves ({state['winner']})", flush=True)
            break
        if state['move_count'] <= count:
            return fail(f'no AI reply within {args.timeout:g} s')
        times.append(state['last_ai_seconds'])
        print(f"PASS  AI reply {len(times)}: {state['last_ai_move']} in {state['last_ai_seconds']:.2f} s "
              f"({state['agent_name']})", flush=True)
        last = state['last_move']
        taken = {tuple(m) for m in state['history']}
        free = sorted(((r, c) for r in range(15) for c in range(15) if (r, c) not in taken
                       and [r, c] not in [p[:2] for p in state.get('forbidden', [])]),
                      key=lambda m: (max(abs(m[0] - last[0]), abs(m[1] - last[1])), m))
        state = call('/api/move', {'row': free[0][0], 'col': free[0][1]})
    if times:
        print(f'PASS  {len(times)} AI replies, max {max(times):.2f} s, mean {sum(times) / len(times):.2f} s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
