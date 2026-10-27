"""Stage 1: inspect a schema-v2 database without changing or creating it."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

from weblog.store import SCHEMA_VERSION, open_readonly

# Validate the fields the admin workflow relies on; never migrate a production DB here.
REQUIRED = {
    'games': {'game_id', 'started_at', 'finished_at', 'status', 'result', 'agent_key',
              'engine_config_hash', 'engine_commit', 'human_color', 'undo_count', 'session_hash', 'number_of_moves'},
    'moves': {'game_id', 'seq', 'ply', 'recorded_at', 'undone_at', 'row0', 'col0',
              'engine_seconds', 'queue_seconds', 'wall_seconds', 'rt_json', 'board_hash'},
    'veto_positions': {'id', 'game_id', 'ply', 'recorded_at', 'reason', 'tree_move_json', 'played_json'},
}


def checkout_commit(root: Path) -> str | None:
    try:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'],
                              capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def inspect_database(path: Path, *, now: datetime | None = None, ttl: int = 3600) -> dict:
    if ttl <= 0:
        raise ValueError('session TTL must be positive')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now must have a timezone')
    with closing(open_readonly(path)) as conn:
        conn.execute('BEGIN')
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(f'expected schema {SCHEMA_VERSION}, found {version}; no migration performed')
        for table, fields in REQUIRED.items():
            columns = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}
            if fields - columns:
                raise ValueError(f'{table}: missing columns {sorted(fields - columns)}')
        counts = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in REQUIRED}
        distribution = [dict(row) for row in conn.execute(
            'SELECT status, result, count(*) AS games FROM games GROUP BY status, result ORDER BY status, result')]
        # Include undo timestamps: undo is activity too. No persistent status is rewritten.
        active = conn.execute('''SELECT g.started_at,
            (SELECT max(recorded_at) FROM moves WHERE game_id=g.game_id) AS last_move,
            (SELECT max(undone_at) FROM moves WHERE game_id=g.game_id) AS last_undo
            FROM games g WHERE g.status='in_progress' ''').fetchall()
        stale = unknown_time = 0
        for row in active:
            try:
                stamps = [datetime.fromisoformat(v.replace('Z', '+00:00')) for v in row if v]
                if any(stamp.tzinfo is None for stamp in stamps):
                    raise ValueError('naive timestamp')
                stale += max(stamps) < now - timedelta(seconds=ttl)
            except (ValueError, TypeError):
                unknown_time += 1
        # A LEFT JOIN also reports tbv3 variants that have zero veto rows.
        veto = [dict(row) for row in conn.execute('''SELECT g.agent_key, g.engine_config_hash,
            count(DISTINCT g.game_id) AS games, count(v.id) AS veto_rows
            FROM games g LEFT JOIN veto_positions v ON v.game_id=g.game_id
            GROUP BY g.agent_key, g.engine_config_hash ORDER BY g.agent_key, g.engine_config_hash''')]
        undo = dict(conn.execute('''SELECT count(*) AS games, coalesce(sum(status='finished'),0) AS finished,
            coalesce(sum(status='finished' AND result='AI_WIN'),0) AS ai_wins,
            coalesce(sum(status='finished' AND result='HUMAN_WIN'),0) AS human_wins,
            coalesce(sum(status='finished' AND result='DRAW'),0) AS draws
            FROM games WHERE undo_count>0''').fetchone())
        summary = dict(conn.execute('''SELECT count(*) AS finished,
            coalesce(sum(result='AI_WIN'),0) AS ai_wins,
            coalesce(sum(result='HUMAN_WIN'),0) AS human_wins,
            coalesce(sum(result='DRAW'),0) AS draws,
            coalesce(sum(result IS NULL OR result NOT IN ('AI_WIN','HUMAN_WIN','DRAW')),0) AS unknown_results
            FROM games WHERE status='finished' ''').fetchone())
        latest = conn.execute('''SELECT game_id, started_at, finished_at, status, result, agent_key,
            human_color, number_of_moves, undo_count FROM games
            ORDER BY started_at DESC, game_id DESC LIMIT 1''').fetchone()
        indexes = {table: [row['name'] for row in conn.execute(f'PRAGMA index_list({table})')]
                   for table in REQUIRED}
        pragmas = {key: conn.execute(f'PRAGMA {key}').fetchone()[0]
                   for key in ('journal_mode', 'busy_timeout', 'query_only')}
    summary['ai_win_rate'] = summary['ai_wins'] / summary['finished'] if summary['finished'] else None
    return {
        'observed_at': now.astimezone(timezone.utc).isoformat(), 'schema_version': version,
        'counts': counts, 'status_results': distribution, 'in_progress': len(active),
        'suspected_stopped': stale, 'unclassified_activity_time': unknown_time, 'session_ttl_seconds': ttl,
        'engine_veto_counts': veto, 'undo_games': undo, 'finished_summary': summary,
        'latest_game': dict(latest) if latest else None, 'indexes': indexes, 'reader_pragmas': pragmas,
        'policy': {'include_undo_games': True, 'draws_in_denominator': True,
                   'win_rate_denominator': "status='finished'", 'stale_status_is_display_only': True},
    }
