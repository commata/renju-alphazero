"""Admin stages 3/4: game list, game record, replay check. The records come from the web server's own
code path (PlaySession + LogStore), so the coordinates, hashes and undo marks are the real ones."""
from contextlib import closing
from datetime import datetime, timezone
from http.client import HTTPConnection
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import scripts.run_web_play as web
from renju import BLACK, WHITE
from webadmin.auth import Auth, hash_password
from webadmin.games import board_hash, game_detail, list_games, list_options, verify_record
from webadmin.server import AdminServer
from weblog.store import LogStore

ROOT = Path(__file__).resolve().parents[1]


class CornerAgent:
    """Plays the smallest legal move (top-left first): never blocks, so the human wins on line 8."""

    def __init__(self, seed):
        self.name, self.diagnostics = 'corner', {}

    def select_move(self, game):
        return min(game.legal_moves())


web.register_agent('corner', 'Corner (test)', CornerAgent)


def build_database(folder: Path) -> tuple[Path, dict[str, str]]:
    """Three games through the server code: a finished win, a game with an undo (finished), one in progress."""
    db = folder / 'web.db'
    store = LogStore(db, log=lambda *a, **k: None)
    ids = {}
    session = web.PlaySession(store=store, log_root=folder / 'logs', session_hash='s' * 16)
    session.reset(agent_key='corner', human_color=BLACK, seed=1)
    for col in (8, 9, 10, 11):          # H8 is the opening; I8..L8 make five
        session.play_human(7, col)
    ids['win'] = session.db_game_id
    session.reset(agent_key='corner', human_color=BLACK, seed=2)
    ids['undo'] = session.db_game_id
    session.play_human(6, 6)
    session.undo()                      # the human move and the AI reply are undone
    for col in (8, 9, 10, 11):
        session.play_human(7, col)
    session.reset(agent_key='corner', human_color=WHITE, seed=3)
    ids['live'] = session.db_game_id
    session.play_human(6, 7)
    store.flush()
    store.close()
    return db, ids


class _Records(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.db, cls.ids = build_database(Path(cls.tmp.name))
        cls.now = datetime.now(timezone.utc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class RecordTests(_Records):
    def test_real_records_replay_and_undo_is_kept_off_the_board(self):
        win = game_detail(self.db, self.ids['win'], now=self.now)
        self.assertTrue(win['verification']['ok'], win['verification'])
        self.assertEqual(win['game']['result'], 'HUMAN_WIN')
        live = [m for m in win['moves'] if m['undone_at'] is None]
        self.assertEqual([m['ply'] for m in live], list(range(1, 10)))
        self.assertEqual((live[0]['row0'], live[0]['col0'], live[0]['kind']), (7, 7, 'OPENING'))
        self.assertEqual([m['player'] for m in live[:3]], ['BLACK', 'WHITE', 'BLACK'])
        self.assertEqual(live[1]['kind'], 'AI')
        self.assertIsNotNone(live[1]['engine_seconds'])
        self.assertNotIn('board_hash', live[0])
        undo = game_detail(self.db, self.ids['undo'], now=self.now)
        self.assertTrue(undo['verification']['ok'], undo['verification'])
        self.assertEqual(undo['game']['undo_count'], 1)
        undone = [m for m in undo['moves'] if m['undone_at']]
        self.assertEqual([(m['ply'], m['row0'], m['col0']) for m in undone][0], (3, 6, 6))
        self.assertEqual(len(undone), 2)
        self.assertEqual(undo['verification']['moves'], 9)
        live = game_detail(self.db, self.ids['live'], now=self.now)
        self.assertTrue(live['verification']['ok'], live['verification'])
        self.assertEqual(live['game']['status'], 'in_progress')
        self.assertFalse(live['game']['suspected'])
        self.assertNotIn('session_hash', json.dumps(win))

    def test_version_answers_unchanged_until_the_record_changes(self):
        first = game_detail(self.db, self.ids['live'], now=self.now)
        again = game_detail(self.db, self.ids['live'], since=first['version'], now=self.now)
        self.assertEqual(again, {'unchanged': True, 'version': first['version'],
                                 'observed_at': self.now.isoformat()})
        later = self.now.replace(year=self.now.year + 1)   # stale now: shown as suspected, so a new version
        stale = game_detail(self.db, self.ids['live'], since=first['version'], now=later)
        self.assertTrue(stale['game']['suspected'])
        self.assertNotEqual(stale['version'], first['version'])
        self.assertIsNone(game_detail(self.db, 'f' * 32, now=self.now))
        with self.assertRaises(ValueError):
            game_detail(self.db, '../x', now=self.now)

    def test_version_detects_game_row_updates_without_new_moves(self):
        before = game_detail(self.db, self.ids['live'], now=self.now)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('UPDATE games SET undo_count=undo_count+1 WHERE game_id=?', (self.ids['live'],))
        try:
            after = game_detail(self.db, self.ids['live'], since=before['version'], now=self.now)
            self.assertNotIn('unchanged', after)
            self.assertNotEqual(after['version'], before['version'])
        finally:
            with closing(sqlite3.connect(self.db)) as conn, conn:
                conn.execute('UPDATE games SET undo_count=undo_count-1 WHERE game_id=?', (self.ids['live'],))

    def test_veto_uses_its_saved_historical_position_and_invalid_json_is_reported(self):
        history = [[7, 7], [0, 0]]
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO veto_positions (game_id, ply, recorded_at, reason, payload_json) VALUES (?,2,?,'VCT2_PROVEN_LOSS',?)",
                         (self.ids['undo'], self.now.isoformat(), json.dumps({'history': history})))
            original = conn.execute('SELECT final_board_json FROM games WHERE game_id=?', (self.ids['undo'],)).fetchone()[0]
            conn.execute("UPDATE games SET final_board_json='not JSON', agent_info_json='not JSON' WHERE game_id=?", (self.ids['undo'],))
        try:
            data = game_detail(self.db, self.ids['undo'], now=self.now)
            self.assertEqual(data['vetoes'][-1]['history'], history)
            self.assertIn('final_board_json', {issue['code'] for issue in data['verification']['issues']})
        finally:
            with closing(sqlite3.connect(self.db)) as conn, conn:
                conn.execute('UPDATE games SET final_board_json=?, agent_info_json=NULL WHERE game_id=?', (original, self.ids['undo']))
                conn.execute('DELETE FROM veto_positions WHERE game_id=?', (self.ids['undo'],))

    def test_board_hash_is_the_servers(self):
        board = [[0] * 15 for _ in range(15)]
        board[7][7], board[0][0] = BLACK, WHITE
        self.assertEqual(board_hash(board, BLACK), web.board_hash(board, BLACK))

    def test_broken_records_are_reported_not_corrected(self):
        record = game_detail(self.db, self.ids['win'], now=self.now)
        game, moves = record['game'], [dict(m, board_hash=None) for m in record['moves']]
        self.assertTrue(verify_record(game, moves, None)['ok'])
        shifted = [dict(m) for m in moves]
        shifted[-1]['col0'] += 1         # the winning move one column off: no five, so the result no longer matches
        self.assertEqual({i['code'] for i in verify_record(game, shifted, None)['issues']}, {'result'})
        shifted = [dict(m) for m in moves]
        shifted[2]['col0'] += 1          # one column off onto a later move: that later move is then illegal
        self.assertEqual({i['code'] for i in verify_record(game, shifted, None)['issues']}, {'illegal', 'replay_stopped'})
        with closing(sqlite3.connect(self.db)) as conn:
            final = json.loads(conn.execute('SELECT final_board_json FROM games WHERE game_id=?',
                                            (self.ids['win'],)).fetchone()[0])
        self.assertTrue(verify_record(game, moves, final)['ok'])
        flipped = [dict(m, row0=14 - m['row0']) for m in moves]   # upside down: line 8 stays, A15.. -> A1..
        self.assertEqual({i['code'] for i in verify_record(game, flipped, final)['issues']}, {'final_board'})
        missing = [m for m in moves if m['ply'] != 4]
        codes = {i['code'] for i in verify_record(game, missing, None)['issues']}
        self.assertIn('ply_sequence', codes)
        self.assertIn('player', codes)
        doubled = moves + [dict(moves[3], seq=99)]
        codes = {i['code'] for i in verify_record(game, doubled, None)['issues']}
        self.assertIn('illegal', codes)
        self.assertIn('replay_stopped', codes)
        outside = [dict(m) for m in moves]
        outside[4]['row0'] = 15
        self.assertIn('out_of_range', {i['code'] for i in verify_record(game, outside, None)['issues']})
        opening = [dict(m) for m in moves]
        opening[0]['actor'] = 'HUMAN'
        self.assertIn('opening', {i['code'] for i in verify_record(game, opening, None)['issues']})

    def test_stored_hash_and_final_board_are_compared(self):
        with closing(sqlite3.connect(self.db)) as conn:
            before = conn.execute('SELECT final_board_json FROM games WHERE game_id=?', (self.ids['win'],)).fetchone()[0]
        record = game_detail(self.db, self.ids['win'], now=self.now)
        board = json.loads(before)
        board[0][14] = BLACK
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute('UPDATE games SET final_board_json=? WHERE game_id=?', (json.dumps(board), self.ids['win']))
            conn.execute("UPDATE moves SET board_hash='0000' WHERE game_id=? AND ply=3", (self.ids['win'],))
        try:
            codes = {i['code'] for i in game_detail(self.db, self.ids['win'], now=self.now)['verification']['issues']}
            self.assertEqual(codes, {'final_board', 'board_hash'})
        finally:
            with closing(sqlite3.connect(self.db)) as conn, conn:
                conn.execute('UPDATE games SET final_board_json=? WHERE game_id=?', (before, self.ids['win']))
                conn.execute('UPDATE moves SET board_hash=NULL WHERE game_id=? AND ply=3', (self.ids['win'],))
        self.assertTrue(record['verification']['ok'])


class ListTests(_Records):
    def ids_of(self, query, **kwargs):
        return [g['game_id'] for g in list_games(self.db, query, now=self.now, **kwargs)['games']]

    def test_newest_first_counts_and_timing(self):
        page = list_games(self.db, {}, now=self.now)
        self.assertEqual(page['total'], 3)
        self.assertEqual([g['game_id'] for g in page['games']], [self.ids['live'], self.ids['undo'], self.ids['win']])
        games = {g['game_id']: g for g in page['games']}
        self.assertEqual(games[self.ids['undo']]['moves'], 9)            # undone moves are off the board
        self.assertEqual(games[self.ids['undo']]['ai_moves'], 5)         # but the engine searched them
        self.assertEqual(games[self.ids['win']]['ai_moves'], 4)
        self.assertEqual(games[self.ids['live']]['moves'], 3)   # opening, human, AI reply
        self.assertEqual(games[self.ids['win']]['soft_limit_exceeded'], 0)
        self.assertIsNone(page['next_cursor'])
        self.assertNotIn('session_hash', json.dumps(page))

    def test_filters(self):
        self.assertEqual(self.ids_of({'status': 'in_progress'}), [self.ids['live']])
        self.assertEqual(self.ids_of({'status': 'finished', 'undo': 'yes'}), [self.ids['undo']])
        self.assertEqual(self.ids_of({'undo': 'no', 'result': 'HUMAN_WIN'}), [self.ids['win']])
        self.assertEqual(self.ids_of({'color': 'WHITE'}), [self.ids['live']])
        self.assertEqual(self.ids_of({'result': 'none'}), [self.ids['live']])
        self.assertEqual(self.ids_of({'agent': 'corner'}), [self.ids['live'], self.ids['undo'], self.ids['win']])
        self.assertEqual(self.ids_of({'agent': 'v7'}), [])
        self.assertEqual(self.ids_of({'status': 'suspected'}), [])
        later = self.now.replace(year=self.now.year + 1)
        self.assertEqual([g['game_id'] for g in list_games(self.db, {'status': 'suspected'}, now=later)['games']],
                         [self.ids['live']])
        for bad in ({'status': 'done'}, {'color': 'RED'}, {'undo': '1'}, {'result': 'WIN'}, {'from': '2026/10/01'},
                    {'limit': '0'}, {'limit': 'x'}, {'cursor': 'not-a-cursor'}):
            with self.assertRaises(ValueError, msg=bad):
                list_games(self.db, bad, now=self.now)

    def test_kst_day_boundaries(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            saved = conn.execute('SELECT game_id, started_at FROM games').fetchall()
            # 2026-10-26 23:30 KST = 14:30 UTC; 2026-10-27 00:30 KST = 2026-10-26 15:30 UTC
            conn.execute("UPDATE games SET started_at='2026-10-26T14:30:00.000+00:00' WHERE game_id=?", (self.ids['win'],))
            conn.execute("UPDATE games SET started_at='2026-10-26T15:30:00.000+00:00' WHERE game_id=?", (self.ids['undo'],))
            conn.execute("UPDATE games SET started_at='2026-10-27T15:00:00.000+00:00' WHERE game_id=?", (self.ids['live'],))
        try:
            self.assertEqual(self.ids_of({'from': '2026-10-27', 'to': '2026-10-27'}), [self.ids['undo']])
            self.assertEqual(self.ids_of({'to': '2026-10-26'}), [self.ids['win']])
        finally:
            with closing(sqlite3.connect(self.db)) as conn, conn:
                conn.executemany('UPDATE games SET started_at=? WHERE game_id=?', [(s, g) for g, s in saved])

    def test_cursor_pages(self):
        seen, cursor = [], None
        for _ in range(4):
            query = {'limit': '1', **({'cursor': cursor} if cursor else {})}
            page = list_games(self.db, query, now=self.now)
            seen += [g['game_id'] for g in page['games']]
            cursor = page['next_cursor']
            if not cursor:
                break
        self.assertEqual(seen, [self.ids['live'], self.ids['undo'], self.ids['win']])

    def test_options(self):
        options = list_options(self.db)
        self.assertEqual([(a['agent_key'], a['games']) for a in options['agents']], [('corner', 3)])
        self.assertIn('suspected', options['statuses'])


class HTTPTests(_Records):
    def setUp(self):
        self.origin = 'https://example.org'
        self.server = AdminServer(('127.0.0.1', 0), db=self.db, origin=self.origin,
                                  auth=Auth('admin', hash_password('pw')), assets=ROOT / 'web/admin')
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        patcher = patch('webadmin.server.game_status', return_value={})
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self, path, cookie=None, method='GET', body=None, headers=None):
        conn = HTTPConnection(*self.server.server_address, timeout=5)
        headers = dict(headers or {})
        if cookie:
            headers['Cookie'] = cookie
        try:
            conn.request(method, path, body, headers)
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def cookie(self):
        status, _, headers = self.call('/login', method='POST', body=json.dumps({'username': 'admin', 'password': 'pw'}),
                                       headers={'Origin': self.origin, 'Content-Type': 'application/json'})
        self.assertEqual(status, 200)
        return headers['Set-Cookie'].split(';')[0]

    def test_routes_need_login(self):
        for path in ('/data/games', '/data/games/options', f'/data/games/{self.ids["win"]}', '/games.js',
                     '/record.js', '/admin.css', '/renju/admin/data/games?status=finished'):
            self.assertEqual(self.call(path)[0], 401, path)

    def test_list_detail_and_errors(self):
        cookie = self.cookie()
        status, body, _ = self.call('/renju/admin/data/games?status=finished&limit=1', cookie)
        self.assertEqual(status, 200)
        page = json.loads(body)
        self.assertEqual((page['total'], len(page['games'])), (2, 1))
        status, body, _ = self.call(f'/data/games?status=finished&limit=1&cursor={page["next_cursor"]}', cookie)
        self.assertEqual(json.loads(body)['games'][0]['game_id'], self.ids['win'])
        status, body, _ = self.call(f'/data/games/{self.ids["undo"]}', cookie)
        record = json.loads(body)
        self.assertTrue(record['verification']['ok'])
        status, body, _ = self.call(f'/data/games/{self.ids["undo"]}?since={record["version"]}', cookie)
        self.assertTrue(json.loads(body)['unchanged'])
        self.assertEqual(self.call('/data/games?status=nope', cookie)[0], 400)
        self.assertEqual(self.call('/data/games?status=finished&status=abandoned', cookie)[0], 400)
        self.assertEqual(self.call('/data/games/' + 'a' * 32, cookie)[0], 404)
        self.assertEqual(self.call('/data/games/a.b', cookie)[0], 400)
        self.assertEqual(self.call('/data/games/a/b', cookie)[0], 404)
        for path in ('/games.js', '/record.js', '/admin.css', '/'):
            self.assertEqual(self.call(path, cookie)[0], 200, path)
        self.assertEqual(json.loads(self.call('/data/games/options', cookie)[1])['agents'][0]['agent_key'], 'corner')


if __name__ == '__main__':
    unittest.main()
