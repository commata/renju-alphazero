"""Stage 5: filtered statistics from one short read-only snapshot.

Win rate includes undo games and draws by default. Durations count all recorded
AI searches, including undone moves. Quantiles use linear interpolation (n-1)*p.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

from weblog.store import GAME_STATUSES, open_readonly
from .games import AI_MOVE, LAST_ACTIVITY, SOFT_LIMIT, cutoff_at, game_filters


def outcomes(games: list[dict]) -> dict:
    statuses = {status: sum(g['status'] == status for g in games) for status in GAME_STATUSES}
    finished = [g for g in games if g['status'] == 'finished']
    wins = sum(g['result'] == 'AI_WIN' for g in finished)
    losses = sum(g['result'] == 'HUMAN_WIN' for g in finished)
    draws = sum(g['result'] == 'DRAW' for g in finished)
    return {'games': len(games), 'statuses': statuses, 'suspected': sum(bool(g['suspected']) for g in games),
            'finished': len(finished), 'ai_wins': wins, 'human_wins': losses, 'draws': draws,
            'unknown_results': len(finished) - wins - losses - draws,
            'ai_win_rate': wins / len(finished) if finished else None}


def duration_summary(values: list) -> dict:
    valid, missing, invalid = [], 0, 0
    for value in values:
        if value is None:
            missing += 1
        elif isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            valid.append(value)
        else:
            invalid += 1
    valid.sort()
    if valid:
        index = (len(valid) - 1) * .95
        lo, hi = math.floor(index), math.ceil(index)
        p95 = valid[lo] + (valid[hi] - valid[lo]) * (index - lo)
    else:
        p95 = None
    return {'count': len(valid), 'missing': missing, 'invalid': invalid,
            'mean': statistics.fmean(valid) if valid else None,
            'median': statistics.median(valid) if valid else None,
            'p95': p95, 'max': valid[-1] if valid else None}


def game_statistics(path: Path, query: dict[str, str], *, now: datetime | None = None, ttl: int = 3600) -> dict:
    now = now or datetime.now(timezone.utc)
    cutoff = cutoff_at(now, ttl)
    where, params = game_filters(query, cutoff)
    filtered = ' AND '.join(where) or '1'
    with closing(open_readonly(path)) as conn:
        conn.execute('BEGIN')
        games = [dict(row) for row in conn.execute(f'''SELECT g.status, g.result, g.agent_key, g.agent_name,
            g.engine_config_hash, g.human_color, g.undo_count, date(g.started_at, '+9 hours') AS day,
            g.status='in_progress' AND {LAST_ACTIVITY} < julianday(?) AS suspected
            FROM games g WHERE {filtered} ORDER BY g.started_at DESC''', [cutoff, *params])]
        times = [dict(row) for row in conn.execute(f'''SELECT m.engine_seconds, m.queue_seconds, m.wall_seconds,
            {SOFT_LIMIT.replace('rt_json', 'm.rt_json')} AS soft_limit_exceeded
            FROM moves m JOIN games g ON g.game_id=m.game_id
            WHERE ({filtered}) AND m.{AI_MOVE}''', params)]
    # Release the SQLite transaction before quantiles, grouping and JSON serialization.
    groups = {name: defaultdict(list) for name in ('engine', 'color', 'day')}
    for game in games:
        groups['engine'][(game['agent_key'], game['engine_config_hash'])].append(game)
        groups['color'][game['human_color']].append(game)
        if game['day']:
            groups['day'][game['day']].append(game)
    by_engine = [{'agent_key': agent, 'engine_config_hash': digest, 'agent_name': rows[0]['agent_name'],
                  **outcomes(rows)} for (agent, digest), rows in groups['engine'].items()]
    by_color = [{'human_color': color, **outcomes(rows)} for color, rows in sorted(groups['color'].items())]
    daily = [{'day': day, **outcomes(rows)} for day, rows in sorted(groups['day'].items())]
    return {'observed_at': now.astimezone(timezone.utc).isoformat(), 'totals': outcomes(games),
            'undo': outcomes([g for g in games if g['undo_count'] > 0]),
            'by_engine': by_engine, 'by_color': by_color, 'daily': daily,
            'invalid_date_games': sum(g['day'] is None for g in games),
            'timing': {key: duration_summary([m[key] for m in times])
                       for key in ('engine_seconds', 'queue_seconds', 'wall_seconds')},
            'ai_moves': len(times), 'soft_limit_exceeded': sum(bool(m['soft_limit_exceeded']) for m in times),
            'policy': {'include_undo_games': query.get('undo') != 'no', 'draws_in_denominator': True,
                       'win_rate_denominator': "status='finished'", 'timing_includes_undone_moves': True,
                       'percentile': 'linear interpolation at (n-1)*0.95', 'timezone': 'Asia/Seoul'}}
