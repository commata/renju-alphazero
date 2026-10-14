"""SQLite store for web games: one writer thread, events from the game server (docs/web-ser5.md §7).

The web server calls the event methods (``start_game``, ``record_move``, ``mark_undone``,
``record_veto``, ``finish_game``); they only enqueue, so a move is never delayed by the disk and a
database failure never fails a move (it is counted in ``failures`` / ``last_error`` and printed).
One thread owns the write connection and applies the events in order. Engine workers never
touch the database: they hand their results to the server, which logs them.

The raw logs (``logs/web_play/<game>/game.json``, ``moves.csv``, ``veto_positions/*.json``) stay as
they are and are written at the end of a game; the database is written move by move, so a game
in progress survives a server crash. ``games.raw_log_dir`` links the two.

Conventions:
- ``game_id`` is a random hex string made by the server (no round trip to the writer);
- coordinates are 0-indexed (``row0``/``col0``);
- times are UTC ISO-8601 strings; durations are seconds;
- an undone move is kept with ``undone_at`` set; ``seq`` counts every recorded move of a game,
  ``ply`` is unique among the moves still on the board;
- engine details stay in ``diagnostics_json`` / ``rt_json`` (SQLite's JSON functions read them);
  only fields common to every engine are columns.

Readers open the file with ``open_readonly`` (``mode=ro``) and never write.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GAME_STATUSES = ('in_progress', 'finished', 'unfinished', 'abandoned')

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'finished', 'unfinished', 'abandoned')),
    result TEXT,                 -- HUMAN_WIN / AI_WIN / DRAW / UNFINISHED
    winner TEXT,                 -- BLACK / WHITE
    winner_actor TEXT,
    agent_key TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    seed INTEGER,
    human_color TEXT NOT NULL,
    number_of_moves INTEGER,
    ai_total_seconds REAL,
    undo_count INTEGER NOT NULL DEFAULT 0,
    engine TEXT,
    engine_config_hash TEXT,
    engine_commit TEXT,
    git_dirty INTEGER,
    policy_checkpoint_sha256 TEXT,
    policy_metadata_sha256 TEXT,
    agent_info_json TEXT,
    final_board_json TEXT,
    raw_log_dir TEXT
);
CREATE INDEX IF NOT EXISTS games_started ON games (started_at);
CREATE INDEX IF NOT EXISTS games_agent ON games (agent_key, started_at);

CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games (game_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,          -- order of recording in the game, undone moves included
    ply INTEGER NOT NULL,          -- stones on the board after the move
    player TEXT NOT NULL,          -- BLACK / WHITE
    actor TEXT NOT NULL,           -- HUMAN / OPENING_RULE / engine name
    row0 INTEGER NOT NULL,
    col0 INTEGER NOT NULL,
    board_hash TEXT,               -- position before the move
    recorded_at TEXT NOT NULL,
    engine_seconds REAL,           -- select_move
    queue_seconds REAL,            -- request received -> search started (lock / worker queue)
    wall_seconds REAL,             -- request received -> move ready: what the user waited
    deadline_seconds REAL,         -- the move's deadline, from the request (NULL: none)
    route TEXT,
    timed_out INTEGER,             -- a deadline cut a phase (RT30)
    emergency_used INTEGER,        -- the emergency / best-so-far move was played (RT30)
    diagnostics_json TEXT,
    rt_json TEXT,
    undone_at TEXT,
    UNIQUE (game_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS moves_live_ply ON moves (game_id, ply) WHERE undone_at IS NULL;
CREATE INDEX IF NOT EXISTS moves_route ON moves (route);
CREATE INDEX IF NOT EXISTS moves_wall ON moves (wall_seconds);

CREATE TABLE IF NOT EXISTS veto_positions (
    id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games (game_id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,          -- stones on the board before the vetoed move
    recorded_at TEXT NOT NULL,
    reason TEXT NOT NULL,          -- VCT2_PROVEN_LOSS
    tree_move_json TEXT,
    played_json TEXT,
    payload_json TEXT NOT NULL,    -- the raw veto file's content
    raw_path TEXT
);
CREATE INDEX IF NOT EXISTS veto_game ON veto_positions (game_id, ply);
"""

GAME_FIELDS = ('started_at', 'agent_key', 'agent_name', 'seed', 'human_color', 'engine', 'engine_config_hash',
               'engine_commit', 'git_dirty', 'policy_checkpoint_sha256', 'policy_metadata_sha256')
FINISH_FIELDS = ('finished_at', 'status', 'result', 'winner', 'winner_actor', 'number_of_moves',
                 'ai_total_seconds', 'undo_count', 'raw_log_dir')
MOVE_FIELDS = ('seq', 'ply', 'player', 'actor', 'row0', 'col0', 'board_hash', 'engine_seconds', 'queue_seconds',
               'wall_seconds', 'deadline_seconds', 'route', 'timed_out', 'emergency_used')


class SchemaVersionError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec='milliseconds')


def new_game_id() -> str:
    return uuid.uuid4().hex


def _json(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def configure(conn: sqlite3.Connection) -> None:
    conn.execute('PRAGMA journal_mode=WAL')     # readers do not block the writer (local disk only)
    conn.execute('PRAGMA synchronous=NORMAL')   # safe with WAL; a power cut may lose the last commits
    conn.execute('PRAGMA foreign_keys=ON')      # per connection
    conn.execute('PRAGMA busy_timeout=5000')


def initialize(path: str | Path) -> None:
    """Create the schema, or check the version of an existing database."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        configure(conn)
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise SchemaVersionError(f'{path}: schema version {version}, this code writes {SCHEMA_VERSION}')
        conn.executescript(SCHEMA)
        conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        conn.commit()
    finally:
        conn.close()


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """A connection that cannot write (``mode=ro``); rows as ``sqlite3.Row``."""
    conn = sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True)
    conn.execute('PRAGMA busy_timeout=5000')
    conn.row_factory = sqlite3.Row
    return conn


_STOP = object()


class LogStore:
    def __init__(self, path: str | Path, *, log=print):
        self.path = Path(path)
        initialize(self.path)
        self.log = log
        self.failures = 0
        self.written = 0
        self.last_error: str | None = None
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='weblog-writer', daemon=True)
        self._thread.start()

    # -- events (enqueue only) ---------------------------------------------
    def start_game(self, *, game_id: str | None = None, **fields: Any) -> str:
        game_id = game_id or new_game_id()
        row = {'game_id': game_id, **{k: fields.get(k) for k in GAME_FIELDS}}
        row['started_at'] = row['started_at'] or utc_now()
        row['git_dirty'] = None if row['git_dirty'] is None else int(bool(row['git_dirty']))
        row['agent_info_json'] = _json(fields.get('agent_info'))
        self._put('INSERT INTO games ({}) VALUES ({})'.format(','.join(row), ','.join('?' * len(row))),
                  tuple(row.values()))
        return game_id

    def record_move(self, game_id: str, *, diagnostics: dict | None = None, rt: dict | None = None,
                    **fields: Any) -> None:
        row = {'game_id': game_id, **{k: fields.get(k) for k in MOVE_FIELDS}, 'recorded_at': utc_now(),
               'diagnostics_json': _json(diagnostics) if diagnostics else None, 'rt_json': _json(rt)}
        for flag in ('timed_out', 'emergency_used'):
            row[flag] = None if row[flag] is None else int(bool(row[flag]))
        self._put('INSERT INTO moves ({}) VALUES ({})'.format(','.join(row), ','.join('?' * len(row))),
                  tuple(row.values()))

    def mark_undone(self, game_id: str, seqs) -> None:
        stamp = utc_now()
        for seq in seqs:
            self._put('UPDATE moves SET undone_at=? WHERE game_id=? AND seq=? AND undone_at IS NULL',
                      (stamp, game_id, seq))
        self._put('UPDATE games SET undo_count=undo_count+1 WHERE game_id=?', (game_id,))

    def record_veto(self, game_id: str, *, ply: int, payload: dict, reason: str = 'VCT2_PROVEN_LOSS',
                    raw_path: str | None = None) -> None:
        self._put('INSERT INTO veto_positions (game_id, ply, recorded_at, reason, tree_move_json, played_json, '
                  'payload_json, raw_path) VALUES (?,?,?,?,?,?,?,?)',
                  (game_id, ply, utc_now(), reason, _json(payload.get('tree_move')), _json(payload.get('played')),
                   _json(payload), raw_path))

    def finish_game(self, game_id: str, *, final_board=None, agent_info=None, **fields: Any) -> None:
        if fields.get('status', 'finished') not in GAME_STATUSES:
            raise ValueError(f"status must be one of {GAME_STATUSES}")
        row = {k: fields.get(k) for k in FINISH_FIELDS if k in fields}
        row.setdefault('finished_at', utc_now())
        row.setdefault('status', 'finished')
        if final_board is not None:
            row['final_board_json'] = _json(final_board)
        if agent_info is not None:
            row['agent_info_json'] = _json(agent_info)
        self._put('UPDATE games SET {} WHERE game_id=?'.format(','.join(f'{k}=?' for k in row)),
                  (*row.values(), game_id))

    # -- control -------------------------------------------------------------
    def flush(self, timeout: float | None = 10.0) -> bool:
        """Wait until every queued event is applied; False on timeout."""
        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def close(self, timeout: float | None = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout)

    def _put(self, sql: str, params: tuple) -> None:
        if self._closed:
            self._fail(f'closed store, event dropped: {sql[:40]}')
            return
        self._queue.put((sql, params))

    def _fail(self, message: str) -> None:
        self.failures += 1
        self.last_error = message
        self.log(f'[weblog] {message}', flush=True)

    def _run(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            configure(conn)
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                sql, params = item
                try:
                    with conn:  # one transaction per event
                        conn.execute(sql, params)
                    self.written += 1
                except Exception as exc:  # noqa: BLE001 - a bad event must not stop the writer
                    self._fail(f'{type(exc).__name__}: {exc} ({sql[:60]})')
        finally:
            conn.close()
