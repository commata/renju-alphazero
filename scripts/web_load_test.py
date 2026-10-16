"""B3: load test of the web server with simulated browsers (docs/web-ser5.md §8, §9.11).

Each simulated browser is one cookie session that plays TB-RT30 (``--agent rt30``) game after game
for ``--minutes``, alternating its colour. Its "human" moves come from ``--human``: a quick local
engine (``v5``, the default, about 1.4 s a move on one core: positions where TB-RT30 really searches;
it competes with the server for CPU when the client runs on the same machine, reported as
``human_engine_seconds``), or ``random`` (a legal move next to the stones: no CPU, but the AI wins
fast and hardly searches). Before each move the browser waits ``--think`` seconds, like a person. It polls ``/api/state`` every 0.25 s while the AI is queued or
thinking, and records what a person would see: the wait from its move to the AI's reply.

With ``--db`` (the server's SQLite file, readable on the same machine) the report adds the server's
own times for the moves of these sessions (matched by session hash): queue, engine, wall, the
deadline overshoot, watchdog / emergency moves and V8-B helper failures.

    python scripts/web_load_test.py --url http://127.0.0.1:8100 --sessions 4 --minutes 15 \\
        --db runs/b3/web_slots1.db --output runs/b3/s1_n4.json
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import statistics
import sys
import threading
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hybrid.progress import Progress  # noqa: E402
from renju import Game  # noqa: E402

POLL = 0.25


class Browser:
    def __init__(self, index: int, url: str, args, stop_at: float):
        parsed = urlparse(url)
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.index, self.args, self.stop_at = index, args, stop_at
        self.rng = random.Random(args.seed * 1000 + index)
        self.cookie: str | None = None
        self.waits: list[float] = []      # human move -> AI reply seen (s)
        self.events = {'busy_409': 0, 'full_503': 0, 'errors': 0, 'ai_errors': 0, 'games': 0, 'finished': 0}
        self.human_seconds: list[float] = []
        self.agent = None
        if args.human != 'random':
            from scripts.run_web_play import create_agent
            self.agent_factory = lambda seed: create_agent(args.human, seed=seed)

    def call(self, method, path, body=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        if self.cookie:
            headers['Cookie'] = f'renju_sid={self.cookie}'
        conn.request(method, path, None if body is None else json.dumps(body), headers)
        response = conn.getresponse()
        payload = json.loads(response.read() or b'{}')
        cookie = response.getheader('Set-Cookie')
        retry = response.getheader('Retry-After')
        conn.close()
        if cookie:
            self.cookie = cookie.split(';')[0].split('=', 1)[1]
        return response.status, payload, retry

    def send(self, path, body):
        """POST until accepted (503: wait Retry-After); the state, or None when the test time is over."""
        while monotonic() < self.stop_at:
            status, payload, retry = self.call('POST', path, body)
            if status in (200, 202):
                return payload
            if status == 503:
                self.events['full_503'] += 1
                sleep(float(retry or 5))
            elif status == 409:
                self.events['busy_409'] += 1
                sleep(1)
            else:
                self.events['errors'] += 1
                raise RuntimeError(f'{path}: HTTP {status}: {payload.get("error")}')
        return None

    def wait_ai(self, state):
        while state['status'] in ('QUEUED', 'THINKING'):
            sleep(POLL)
            status, state, _ = self.call('GET', '/api/state')
            if status != 200:
                self.events['errors'] += 1
        if state.get('last_error'):
            self.events['ai_errors'] += 1
        return state

    def human_move(self, state):
        game = Game()
        for move in state['history']:
            game.play(*move)
        if self.agent is not None:
            started = monotonic()
            move = self.agent.select_move(game)
            self.human_seconds.append(monotonic() - started)
            return move
        stones = state['history']
        legal = game.legal_moves()
        near = [m for m in legal if any(max(abs(m[0] - r), abs(m[1] - c)) <= 2 for r, c in stones)]
        return self.rng.choice(near or legal)

    def run(self):
        colour = 'BLACK' if self.index % 2 else 'WHITE'
        self.call('GET', '/api/state')  # the session cookie
        while monotonic() < self.stop_at:
            if self.args.human != 'random':
                self.agent = self.agent_factory(self.rng.randrange(1 << 30))
            state = self.send('/api/new', {'agent': self.args.agent, 'human_color': colour,
                                           'seed': self.rng.randrange(1 << 30)})
            if state is None:
                break
            self.events['games'] += 1
            started = monotonic()
            state = self.wait_ai(state)  # the AI's first reply when the human is black
            while not state['done'] and monotonic() < self.stop_at:
                sleep(self.args.think)
                move = self.human_move(state)
                sent = monotonic()
                state = self.send('/api/move', {'row': move[0], 'col': move[1]})
                if state is None:
                    break
                if state['status'] in ('QUEUED', 'THINKING'):
                    state = self.wait_ai(state)
                    self.waits.append(monotonic() - sent)
            if state is not None and state['done']:
                self.events['finished'] += 1
            colour = 'WHITE' if colour == 'BLACK' else 'BLACK'


def quantiles(values):
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    pick = lambda q: round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)  # noqa: E731
    return {'n': len(values), 'median': round(statistics.median(ordered), 3), 'p95': pick(0.95),
            'p99': pick(0.99), 'max': round(ordered[-1], 3)}


def server_side(db: Path, hashes: list[str], deadline: float) -> dict:
    from weblog.store import open_readonly

    conn = open_readonly(db)
    try:
        marks = ','.join('?' * len(hashes))
        rows = [dict(r) for r in conn.execute(
            f'SELECT m.queue_seconds, m.engine_seconds, m.wall_seconds, m.deadline_seconds, m.timed_out, '
            f'm.emergency_used, m.rt_json FROM moves m JOIN games g USING (game_id) '
            f'WHERE g.session_hash IN ({marks}) AND m.actor NOT IN (\'HUMAN\', \'OPENING_RULE\')', hashes)]
    finally:
        conn.close()
    rts = [json.loads(r['rt_json']) if r['rt_json'] else {} for r in rows]
    engine = [r['engine_seconds'] for r in rows if r['engine_seconds'] is not None]
    return {
        'ai_moves': len(rows),
        'queue_seconds': quantiles([r['queue_seconds'] for r in rows if r['queue_seconds'] is not None]),
        'engine_seconds': quantiles(engine),
        'wall_seconds': quantiles([r['wall_seconds'] for r in rows if r['wall_seconds'] is not None]),
        f'engine_over_{deadline:g}s': sum(e > deadline for e in engine),
        'timed_out_moves': sum(bool(r['timed_out']) for r in rows),
        'emergency_moves': sum(bool(r['emergency_used']) for r in rows),
        'watchdog_moves': sum(bool(rt.get('watchdog')) for rt in rts),
        'attack_parallel_moves': sum(bool(rt.get('attack_parallel')) for rt in rts),
        'attack_helper_failed_moves': sum(bool(rt.get('attack_helper_failed')) for rt in rts),
        'slot_moved_moves': sum(rt.get('slot_moved_from') is not None for rt in rts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://127.0.0.1:8100')
    parser.add_argument('--sessions', type=int, required=True)
    parser.add_argument('--minutes', type=float, default=15.0)
    parser.add_argument('--agent', default='rt30', help='the server opponent key')
    parser.add_argument('--human', default='v5', help="a quick engine key (v5, default) or 'random' (no CPU, trivial games)")
    parser.add_argument('--think', type=float, default=2.0, help='seconds before each human move (plus the engine time)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--db', type=Path, help="the server's SQLite file (same machine): server-side times")
    parser.add_argument('--deadline', type=float, default=27.5)
    parser.add_argument('--label', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    started = monotonic()
    stop_at = started + args.minutes * 60
    browsers = [Browser(i, args.url, args, stop_at) for i in range(args.sessions)]
    failures: list[str] = []

    def run(browser):
        try:
            browser.run()
        except Exception as exc:  # noqa: BLE001 - reported in the summary
            failures.append(f'browser {browser.index}: {type(exc).__name__}: {exc}')

    threads = [threading.Thread(target=run, args=(b,), daemon=True) for b in browsers]
    for thread in threads:
        thread.start()
        sleep(0.5)  # stagger the first requests a little
    print(f'B3 load: {args.sessions} sessions x {args.minutes:g} min vs {args.agent} ({args.url}), '
          f'human {args.human}, think {args.think:g} s', flush=True)
    progress = Progress('b3-load', args.minutes * 60, 's')
    while any(t.is_alive() for t in threads):
        sleep(5)
        moves = sum(len(b.waits) for b in browsers)
        elapsed = min(monotonic() - started, args.minutes * 60)
        progress.update(elapsed - progress.done, extra=f'{moves} AI moves')  # a line about once a minute

    waits = [w for b in browsers for w in b.waits]
    events = {k: sum(b.events[k] for b in browsers) for k in browsers[0].events}
    hashes = [hashlib.sha256(b.cookie.encode()).hexdigest()[:16] for b in browsers if b.cookie]
    summary = {'sessions': args.sessions, 'minutes': args.minutes, 'agent': args.agent, 'human': args.human,
               'think_seconds': args.think, 'label': args.label, 'events': events, 'failures': failures,
               'client_wait_seconds': quantiles(waits),
               'per_session_ai_moves': [len(b.waits) for b in browsers],
               'human_engine_seconds': quantiles([s for b in browsers for s in b.human_seconds])}
    if args.db is not None and hashes:
        summary['server'] = server_side(args.db, hashes, args.deadline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
