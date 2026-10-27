"""Stages 3/4: game list, one game's record and its replay check, read-only.

The record is replayed from the stored moves with ``renju.Game`` (the rules the server enforced);
no engine runs. A record that does not replay is reported as it is, never corrected.

Conventions of ``weblog.store``: coordinates are 0-indexed (row 0 is the top line, shown as 15);
an undone move keeps its row with ``undone_at`` set, so the moves on the board are the ones with
``undone_at IS NULL`` ordered by ``ply`` (stones on the board after the move).
"""
from __future__ import annotations

import base64
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

from renju import BLACK, WHITE, Game, IllegalMove
from renju.game import OPENING_MOVE
from renju.rules import SIZE
from weblog.store import GAME_STATUSES, open_readonly

KST = timezone(timedelta(hours=9))
AI_MOVE = "actor NOT IN ('HUMAN', 'OPENING_RULE')"
SOFT_LIMIT = "coalesce(json_extract(rt_json, '$.soft_limit_exceeded'), 0) = 1"
RESULTS = ('HUMAN_WIN', 'AI_WIN', 'DRAW', 'UNFINISHED')
STATUS_FILTERS = GAME_STATUSES + ('suspected',)
GAME_ID = re.compile(r'[0-9A-Za-z_-]{1,64}')
DEFAULT_LIMIT, MAX_LIMIT = 50, 200
# Last activity of a game: its start, its last recorded move or its last undo (julianday: any ISO offset).
LAST_ACTIVITY = """max(julianday(g.started_at),
    coalesce((SELECT julianday(max(recorded_at)) FROM moves WHERE game_id = g.game_id), 0),
    coalesce((SELECT julianday(max(undone_at)) FROM moves WHERE game_id = g.game_id), 0))"""
GAME_COLUMNS = ('game_id', 'started_at', 'finished_at', 'status', 'result', 'winner', 'winner_actor', 'agent_key',
                'agent_name', 'seed', 'human_color', 'number_of_moves', 'ai_total_seconds', 'undo_count', 'engine',
                'engine_config_hash', 'engine_commit', 'git_dirty', 'policy_checkpoint_sha256',
                'policy_metadata_sha256', 'raw_log_dir')


def _cutoff(now: datetime, ttl: int) -> str:
    if now.tzinfo is None:
        raise ValueError('now must have a timezone')
    if ttl <= 0:
        raise ValueError('session TTL must be positive')
    return (now - timedelta(seconds=ttl)).astimezone(timezone.utc).isoformat(timespec='milliseconds')


def _kst_day(value: str, name: str) -> datetime:
    try:
        return datetime.combine(date.fromisoformat(value), time(), KST)
    except ValueError:
        raise ValueError(f'{name}: YYYY-MM-DD') from None


def _encode_cursor(started_at: str, game_id: str) -> str:
    raw = json.dumps([started_at, game_id], separators=(',', ':')).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def _decode_cursor(value: str) -> tuple[str, str]:
    try:
        raw = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
        started_at, game_id = json.loads(raw)
        if isinstance(started_at, str) and isinstance(game_id, str) and GAME_ID.fullmatch(game_id):
            return started_at, game_id
    except (ValueError, TypeError):
        pass
    raise ValueError('cursor: invalid')


def _filters(query: dict[str, str], cutoff: str) -> tuple[list[str], list]:
    """WHERE clauses and parameters of the list filters; unknown values are refused (ValueError)."""
    where, params = [], []
    if query.get('from'):
        where.append('julianday(g.started_at) >= julianday(?)')
        params.append(_kst_day(query['from'], 'from').astimezone(timezone.utc).isoformat())
    if query.get('to'):  # inclusive KST day
        end = _kst_day(query['to'], 'to') + timedelta(days=1)
        where.append('julianday(g.started_at) < julianday(?)')
        params.append(end.astimezone(timezone.utc).isoformat())
    if query.get('agent'):
        if len(query['agent']) > 64:
            raise ValueError('agent: too long')
        where.append('g.agent_key = ?')
        params.append(query['agent'])
    if query.get('color'):
        if query['color'] not in ('BLACK', 'WHITE'):
            raise ValueError('color: BLACK or WHITE')
        where.append('g.human_color = ?')
        params.append(query['color'])
    status = query.get('status')
    if status:
        if status not in STATUS_FILTERS:
            raise ValueError(f'status: one of {STATUS_FILTERS}')
        if status == 'suspected':
            where.append(f"g.status = 'in_progress' AND {LAST_ACTIVITY} < julianday(?)")
            params.append(cutoff)
        else:
            where.append('g.status = ?')
            params.append(status)
    result = query.get('result')
    if result:
        if result == 'none':
            where.append('g.result IS NULL')
        elif result in RESULTS:
            where.append('g.result = ?')
            params.append(result)
        else:
            raise ValueError(f'result: one of {RESULTS} or none')
    undo = query.get('undo')
    if undo:
        if undo not in ('yes', 'no'):
            raise ValueError('undo: yes or no')
        where.append('g.undo_count > 0' if undo == 'yes' else 'g.undo_count = 0')
    return where, params


def list_games(path: Path, query: dict[str, str], *, now: datetime | None = None, ttl: int = 3600) -> dict:
    """One page of games, newest first, with the filters of ``query`` (from / to in KST, agent, color, status,
    result, undo, cursor, limit). Timing columns count every recorded AI move (undone ones too: the engine
    searched them); the move count is the stones on the board."""
    now = now or datetime.now(timezone.utc)
    cutoff = _cutoff(now, ttl)
    try:
        limit = int(query.get('limit') or DEFAULT_LIMIT)
    except ValueError:
        raise ValueError('limit: integer') from None
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f'limit: 1..{MAX_LIMIT}')
    where, params = _filters(query, cutoff)
    filtered = ' AND '.join(where) or '1'
    page_where, page_params = [filtered], list(params)
    if query.get('cursor'):
        started_at, game_id = _decode_cursor(query['cursor'])
        page_where.append('(g.started_at < ? OR (g.started_at = ? AND g.game_id < ?))')
        page_params += [started_at, started_at, game_id]
    with closing(open_readonly(path)) as conn:
        conn.execute('BEGIN')  # one snapshot for the page, its count and its timing
        total = conn.execute(f'SELECT count(*) FROM games g WHERE {filtered}', params).fetchone()[0]
        rows = conn.execute(f'''SELECT g.game_id, g.started_at, g.finished_at, g.status, g.result, g.agent_key,
                g.agent_name, substr(g.engine_config_hash, 1, 12) AS engine_hash, g.human_color, g.undo_count,
                (SELECT count(*) FROM moves WHERE game_id = g.game_id AND undone_at IS NULL) AS moves,
                g.status = 'in_progress' AND {LAST_ACTIVITY} < julianday(?) AS suspected
            FROM games g WHERE {' AND '.join(page_where)}
            ORDER BY g.started_at DESC, g.game_id DESC LIMIT ?''', [cutoff, *page_params, limit + 1]).fetchall()
        games = [dict(row) for row in rows[:limit]]
        if games:
            ids = [game['game_id'] for game in games]
            marks = ','.join('?' * len(ids))
            timing = {row['game_id']: dict(row) for row in conn.execute(f'''SELECT game_id,
                    count(*) AS ai_moves, avg(engine_seconds) AS engine_avg, max(engine_seconds) AS engine_max,
                    avg(wall_seconds) AS wall_avg, max(wall_seconds) AS wall_max,
                    coalesce(sum({SOFT_LIMIT}), 0) AS soft_limit_exceeded
                FROM moves WHERE game_id IN ({marks}) AND {AI_MOVE} GROUP BY game_id''', ids)}
            vetoes = dict(conn.execute(f'''SELECT game_id, count(*) FROM veto_positions
                WHERE game_id IN ({marks}) GROUP BY game_id''', ids).fetchall())
            for game in games:
                stats = timing.get(game['game_id'], {})
                game.update({key: stats.get(key) for key in ('ai_moves', 'engine_avg', 'engine_max', 'wall_avg',
                                                             'wall_max', 'soft_limit_exceeded')})
                game['ai_moves'] = game['ai_moves'] or 0
                game['soft_limit_exceeded'] = game['soft_limit_exceeded'] or 0
                game['vetoes'] = vetoes.get(game['game_id'], 0)
                game['suspected'] = bool(game['suspected'])
    last = games[-1] if games and len(rows) > limit else None
    return {'observed_at': now.astimezone(timezone.utc).isoformat(), 'total': total, 'games': games,
            'next_cursor': _encode_cursor(last['started_at'], last['game_id']) if last else None,
            'session_ttl_seconds': ttl}


def list_options(path: Path) -> dict:
    """The opponents seen in the database (for the list filter), with their engine hashes."""
    with closing(open_readonly(path)) as conn:
        rows = conn.execute('''SELECT agent_key, agent_name, substr(engine_config_hash, 1, 12) AS engine_hash,
                count(*) AS games, max(started_at) AS last_started
            FROM games GROUP BY agent_key, agent_name, engine_config_hash
            ORDER BY last_started DESC''').fetchall()
    return {'agents': [dict(row) for row in rows], 'statuses': list(STATUS_FILTERS), 'results': list(RESULTS)}


def board_hash(board, to_play: int) -> str:
    """The web server's ``board_hash`` (scripts/run_web_play.py): stones and the side to move."""
    raw = json.dumps([[list(line) for line in board], to_play], separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def _issue(code: str, message: str, *, seq: int | None = None, ply: int | None = None) -> dict:
    return {'code': code, 'message': message, 'seq': seq, 'ply': ply}


def verify_record(game: dict, moves: list[dict], final_board) -> dict:
    """Replay the moves on the board and compare them with what was recorded.

    Checks: plies 1..N without gaps or repeats, recording order, the opening at the centre, colours
    alternating from black, every move legal (renju rules, black's forbidden points), the position hash
    before each move, the final board, the move count and the result of a finished game."""
    issues = []
    live = sorted((m for m in moves if m['undone_at'] is None), key=lambda m: (m['ply'], m['seq']))
    plies = [m['ply'] for m in live]
    if plies != list(range(1, len(live) + 1)):
        issues.append(_issue('ply_sequence', f'수 번호가 1..{len(live)}로 이어지지 않습니다: {plies[:12]}…'
                             if len(plies) > 12 else f'수 번호가 1..{len(live)}로 이어지지 않습니다: {plies}'))
    seqs = [m['seq'] for m in live]
    if seqs != sorted(seqs):
        issues.append(_issue('seq_order', '판 위의 수가 기록 순서와 다릅니다'))
    board = Game()
    replayed = 0
    for move in live:
        seq, ply, row, col = move['seq'], move['ply'], move['row0'], move['col0']
        if not (isinstance(row, int) and isinstance(col, int) and 0 <= row < SIZE and 0 <= col < SIZE):
            issues.append(_issue('out_of_range', f'좌표가 바둑판 밖입니다: ({row}, {col})', seq=seq, ply=ply))
            break
        mover = board.to_play
        expected = 'BLACK' if mover == BLACK else 'WHITE'
        if move['player'] != expected:
            issues.append(_issue('player', f'{expected} 차례인데 {move["player"]}로 기록되었습니다', seq=seq, ply=ply))
        if replayed == 0 and ((row, col) != OPENING_MOVE or move['actor'] != 'OPENING_RULE'):
            issues.append(_issue('opening', '첫 수가 중앙 자동 착수(OPENING_RULE)가 아닙니다', seq=seq, ply=ply))
        if move['board_hash'] is not None and move['board_hash'] != board_hash(board.board, mover):
            issues.append(_issue('board_hash', '착수 전 국면이 기록된 hash와 다릅니다', seq=seq, ply=ply))
        try:
            board.play(row, col)
        except IllegalMove as exc:
            issues.append(_issue('illegal', f'규칙상 둘 수 없는 수입니다: {exc}', seq=seq, ply=ply))
            break
        replayed += 1
    complete = replayed == len(live)
    if not complete:
        issues.append(_issue('replay_stopped', f'{replayed}수까지만 재현했습니다 (전체 {len(live)}수)'))
    if complete and final_board is not None and game['status'] != 'in_progress' and final_board != board.board:
        issues.append(_issue('final_board', '재현한 마지막 국면이 저장된 최종 보드와 다릅니다'))
    if game['number_of_moves'] is not None and game['number_of_moves'] != len(live):
        issues.append(_issue('move_count', f'저장된 착수 수 {game["number_of_moves"]} ≠ 판 위의 수 {len(live)}'))
    if complete and game['status'] == 'finished':
        if not board.done:
            issues.append(_issue('result', '정상 종료로 기록되었지만 재현한 국면은 끝나지 않았습니다'))
        else:
            human = BLACK if game['human_color'] == 'BLACK' else WHITE
            expected = 'DRAW' if board.winner is None else 'HUMAN_WIN' if board.winner == human else 'AI_WIN'
            if game['result'] != expected:
                issues.append(_issue('result', f'기록된 결과 {game["result"]} ≠ 재현한 결과 {expected}'))
    return {'ok': not issues, 'replayed': replayed, 'moves': len(live), 'issues': issues}


def _point(value) -> list[int] | None:
    try:
        point = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return None
    if (isinstance(point, list) and len(point) == 2 and all(isinstance(v, int) and 0 <= v < SIZE for v in point)):
        return point
    return None


def game_detail(path: Path, game_id: str, *, since: str | None = None, now: datetime | None = None,
                ttl: int = 3600) -> dict | None:
    """One game, its moves (undone ones included), its VCT2 veto rows and the replay check, read in one
    snapshot. ``version`` changes with every new move, undo, veto or status change: with ``since`` equal to
    it only ``{'unchanged': True}`` comes back. None: no such game."""
    if not GAME_ID.fullmatch(game_id or ''):
        raise ValueError('game id: invalid')
    now = now or datetime.now(timezone.utc)
    cutoff = _cutoff(now, ttl)
    with closing(open_readonly(path)) as conn:
        conn.execute('BEGIN')
        row = conn.execute(f'''SELECT {', '.join('g.' + c for c in GAME_COLUMNS)}, g.final_board_json,
                json_extract(g.agent_info_json, '$.runtime_hash') AS runtime_hash,
                g.status = 'in_progress' AND {LAST_ACTIVITY} < julianday(?) AS suspected
            FROM games g WHERE g.game_id = ?''', (cutoff, game_id)).fetchone()
        if row is None:
            return None
        marker = conn.execute('''SELECT coalesce(max(seq), -1), count(*), count(undone_at)
            FROM moves WHERE game_id = ?''', (game_id,)).fetchone()
        veto_count = conn.execute('SELECT count(*) FROM veto_positions WHERE game_id = ?', (game_id,)).fetchone()[0]
        version = hashlib.sha256(json.dumps([*marker, veto_count, row['status'], row['result'], row['finished_at'],
                                             bool(row['suspected'])]).encode()).hexdigest()[:16]
        if since == version:
            return {'unchanged': True, 'version': version, 'observed_at': now.astimezone(timezone.utc).isoformat()}
        moves = [dict(m) for m in conn.execute(f'''SELECT seq, ply, player, actor, row0, col0, board_hash,
                recorded_at, engine_seconds, queue_seconds, wall_seconds, timed_out, emergency_used, route,
                {SOFT_LIMIT} AS soft_limit_exceeded, undone_at
            FROM moves WHERE game_id = ? ORDER BY seq''', (game_id,))]
        vetoes = [dict(v) for v in conn.execute('''SELECT id, ply, recorded_at, reason, tree_move_json, played_json
            FROM veto_positions WHERE game_id = ? ORDER BY ply, id''', (game_id,))]
    game = {key: row[key] for key in GAME_COLUMNS}
    game.update(runtime_hash=row['runtime_hash'], suspected=bool(row['suspected']))
    try:
        final_board = json.loads(row['final_board_json']) if row['final_board_json'] else None
    except ValueError:
        final_board = None
    for move in moves:
        move['kind'] = 'HUMAN' if move['actor'] == 'HUMAN' else 'OPENING' if move['actor'] == 'OPENING_RULE' else 'AI'
        for flag in ('timed_out', 'emergency_used', 'soft_limit_exceeded'):
            move[flag] = None if move[flag] is None else bool(move[flag])
    for veto in vetoes:
        veto['tree_move'] = _point(veto.pop('tree_move_json'))
        veto['played'] = _point(veto.pop('played_json'))
    check = verify_record(game, moves, final_board)
    for move in moves:
        del move['board_hash']
    return {'version': version, 'observed_at': now.astimezone(timezone.utc).isoformat(), 'game': game,
            'moves': moves, 'vetoes': vetoes, 'verification': check, 'session_ttl_seconds': ttl}
