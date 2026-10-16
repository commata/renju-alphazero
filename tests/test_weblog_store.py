"""weblog.store: SQLite log of the web games (docs/web-ser5.md §7)."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from renju import BLACK, WHITE
from scripts.run_web_play import PlaySession
from weblog.store import SCHEMA, SCHEMA_VERSION, LogStore, SchemaVersionError, initialize, open_readonly


class _Tmp(unittest.TestCase):
    def setUp(self):
        # Registered first, so it runs last: every store's writer thread is closed before the
        # directory goes (tearDown would run before the stores' cleanups).
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db = self.dir / 'data' / 'renju_web.db'
        self.messages = []

    def store(self):
        store = LogStore(self.db, log=lambda *a, **k: self.messages.append(a[0]))
        self.addCleanup(store.close)
        return store

    def rows(self, sql, *params):
        conn = open_readonly(self.db)
        try:
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()


class SchemaTest(_Tmp):
    def test_pragmas_version_and_reopen(self):
        self.store().close()
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
        self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
        conn.close()
        initialize(self.db)  # an existing database of this version opens again

    def test_other_schema_version_is_refused(self):
        initialize(self.db)
        conn = sqlite3.connect(self.db)
        conn.execute(f'PRAGMA user_version={SCHEMA_VERSION + 1}')
        conn.commit()
        conn.close()
        with self.assertRaises(SchemaVersionError):
            LogStore(self.db)

    def test_version_1_database_is_migrated(self):
        Path(self.db).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db)
        conn.executescript(SCHEMA.replace('    raw_log_dir TEXT,\n    session_hash TEXT', '    raw_log_dir TEXT')
                           .replace('CREATE INDEX IF NOT EXISTS games_session ON games (session_hash, started_at);', ''))
        conn.execute("INSERT INTO games (game_id, started_at, agent_key, agent_name, human_color) "
                     "VALUES ('old', 't', 'v7', 'MCTS-v7', 'BLACK')")
        conn.execute('PRAGMA user_version=1')
        conn.commit()
        conn.close()
        store = LogStore(self.db)
        store.start_game(game_id='new', agent_key='v7', agent_name='MCTS-v7', human_color='WHITE', session_hash='abc')
        store.close()
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)
        self.assertEqual(dict(conn.execute('SELECT game_id, session_hash FROM games').fetchall()),
                         {'old': None, 'new': 'abc'})
        conn.close()

    def test_readonly_connection_cannot_write(self):
        self.store().close()
        conn = open_readonly(self.db)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO games (game_id, started_at, agent_key, agent_name, human_color) "
                         "VALUES ('x', 't', 'a', 'b', 'BLACK')")
        conn.close()


class EventsTest(_Tmp):
    def test_game_moves_undo_veto_finish(self):
        store = self.store()
        gid = store.start_game(agent_key='tb1', agent_name='TB-Engine-v1', seed=7, human_color='BLACK',
                               engine='TB-Engine-v1', git_dirty=False, agent_info={'config': {'a': 1}})
        store.record_move(gid, seq=0, ply=1, player='BLACK', actor='OPENING_RULE', row0=7, col0=7)
        store.record_move(gid, seq=1, ply=2, player='WHITE', actor='TB-Engine-v1', row0=7, col0=8,
                          engine_seconds=1.5, queue_seconds=0.01, wall_seconds=1.52, route='tree',
                          diagnostics={'v8_route': 'tree', 'v8_root_nodes': 1234}, rt={'timeouts': ['root']},
                          timed_out=True)
        store.mark_undone(gid, [1])
        # The same ply is played again after the undo: allowed, the undone row is kept.
        store.record_move(gid, seq=2, ply=2, player='WHITE', actor='TB-Engine-v1', row0=8, col0=8, route='tree')
        store.record_veto(gid, ply=2, payload={'tree_move': [6, 6], 'played': [8, 8], 'history': [[7, 7]]},
                          raw_path='logs/web_play/veto_positions/x.json')
        store.finish_game(gid, status='finished', result='AI_WIN', winner='WHITE', winner_actor='TB-Engine-v1',
                          number_of_moves=2, ai_total_seconds=1.5, undo_count=1, raw_log_dir='logs/web_play/g',
                          final_board=[[0] * 15 for _ in range(15)])
        self.assertTrue(store.flush())
        self.assertEqual(store.failures, 0, self.messages)

        game = self.rows('SELECT * FROM games')[0]
        self.assertEqual((game['status'], game['result'], game['undo_count'], game['git_dirty']), ('finished', 'AI_WIN', 1, 0))
        self.assertEqual(json.loads(game['agent_info_json']), {'config': {'a': 1}})
        self.assertTrue(game['started_at'].endswith('+00:00'))
        live = self.rows('SELECT seq, ply, row0, col0 FROM moves WHERE undone_at IS NULL ORDER BY seq')
        self.assertEqual(live, [{'seq': 0, 'ply': 1, 'row0': 7, 'col0': 7}, {'seq': 2, 'ply': 2, 'row0': 8, 'col0': 8}])
        undone = self.rows('SELECT seq, timed_out FROM moves WHERE undone_at IS NOT NULL')
        self.assertEqual(undone, [{'seq': 1, 'timed_out': 1}])
        # Engine details are read with SQLite's JSON functions.
        self.assertEqual(self.rows("SELECT json_extract(diagnostics_json, '$.v8_root_nodes') AS n FROM moves "
                                   "WHERE seq = 1")[0]['n'], 1234)
        veto = self.rows('SELECT * FROM veto_positions')[0]
        self.assertEqual((veto['ply'], veto['reason'], json.loads(veto['played_json'])), (2, 'VCT2_PROVEN_LOSS', [8, 8]))

    def test_a_bad_event_is_counted_and_the_writer_goes_on(self):
        store = self.store()
        gid = store.start_game(agent_key='v7', agent_name='MCTS-v7', human_color='BLACK')
        store.record_move(gid, seq=0, ply=1, player='BLACK', actor='HUMAN', row0=7, col0=7)
        store.record_move(gid, seq=0, ply=2, player='WHITE', actor='HUMAN', row0=7, col0=8)  # duplicate seq
        store.record_move('no-such-game', seq=0, ply=1, player='BLACK', actor='HUMAN', row0=1, col0=1)  # FK
        store.record_move(gid, seq=1, ply=2, player='WHITE', actor='MCTS-v7', row0=7, col0=8)
        self.assertTrue(store.flush())
        self.assertEqual(store.failures, 2)
        self.assertEqual(len(self.messages), 2)
        self.assertEqual(len(self.rows('SELECT * FROM moves')), 2)

    def test_events_after_close_are_dropped_not_raised(self):
        store = self.store()
        store.close()
        store.start_game(agent_key='v7', agent_name='MCTS-v7', human_color='BLACK')
        self.assertEqual(store.failures, 1)

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            self.store().finish_game('x', status='done')


class WebSessionTest(_Tmp):
    """PlaySession with a store: the database follows the game; the raw files are unchanged."""

    def session(self):
        store = self.store()
        return PlaySession(log_root=self.dir / 'logs', store=store), store

    def test_game_undo_and_finish(self):
        session, store = self.session()
        session.reset(agent_key='v5', human_color=WHITE, seed=7, received_at=None)
        session.play_human(7, 8)
        session.play_human(6, 8)
        session.undo()
        session.play_human(8, 8)
        store.flush()
        gid = session.db_game_id
        moves = self.rows('SELECT seq, ply, actor, undone_at IS NOT NULL AS undone, engine_seconds, route '
                          'FROM moves WHERE game_id=? ORDER BY seq', gid)
        self.assertEqual([(m['seq'], m['ply'], m['undone']) for m in moves],
                         [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 1), (4, 5, 1), (5, 4, 0), (6, 5, 0)])
        self.assertEqual(moves[0]['actor'], 'OPENING_RULE')
        self.assertEqual(moves[2]['actor'], session.agent.name)
        self.assertGreater(moves[2]['engine_seconds'], 0)
        live = self.rows('SELECT row0, col0 FROM moves WHERE game_id=? AND undone_at IS NULL ORDER BY ply', gid)
        self.assertEqual([[m['row0'], m['col0']] for m in live], [list(m) for m in session.game.history])
        game = self.rows('SELECT status, undo_count, human_color FROM games WHERE game_id=?', gid)[0]
        self.assertEqual(game, {'status': 'in_progress', 'undo_count': 1, 'human_color': 'WHITE'})

        # Finishing writes the raw files and closes the row with a link to them.
        session.game.done, session.game.winner = True, WHITE
        log_dir = session._save_completed_game_locked()
        store.flush()
        game = self.rows('SELECT * FROM games WHERE game_id=?', gid)[0]
        self.assertEqual((game['status'], game['result'], game['winner']), ('finished', 'HUMAN_WIN', 'WHITE'))
        self.assertTrue(game['raw_log_dir'].endswith(log_dir.name))
        payload = json.loads((log_dir / 'game.json').read_text(encoding='utf-8'))
        self.assertNotIn('seq', payload['moves'][0])  # raw format unchanged
        self.assertEqual(store.failures, 0, self.messages)

    def test_wall_time_counts_from_the_request(self):
        from time import perf_counter

        session, store = self.session()
        session.reset(agent_key='v5', human_color=BLACK, seed=7)
        received = perf_counter() - 2.0  # the request arrived two seconds ago
        session.play_human(6, 7, received_at=received)
        store.flush()
        ai = self.rows("SELECT engine_seconds, queue_seconds, wall_seconds FROM moves "
                       "WHERE actor NOT IN ('HUMAN', 'OPENING_RULE') ORDER BY seq DESC LIMIT 1")[0]
        self.assertGreaterEqual(ai['queue_seconds'], 2.0)
        self.assertGreaterEqual(ai['wall_seconds'], ai['queue_seconds'] + ai['engine_seconds'] - 1e-6)

    def test_replaced_game_is_abandoned(self):
        session, store = self.session()
        session.reset(agent_key='v5', human_color=BLACK, seed=7)
        first = session.db_game_id
        session.reset(agent_key='v5', human_color=WHITE, seed=8)
        store.flush()
        self.assertNotEqual(session.db_game_id, first)
        status = {r['game_id']: r['status'] for r in self.rows('SELECT game_id, status FROM games')}
        self.assertEqual(status, {first: 'abandoned', session.db_game_id: 'in_progress'})

    def test_initial_session_without_new_game_is_logged(self):
        session, store = self.session()
        session.play_human(7, 7)
        store.flush()
        self.assertIsNotNone(session.db_game_id)
        self.assertEqual(len(self.rows('SELECT * FROM moves')), 2)

    def test_no_store_is_the_old_behaviour(self):
        session = PlaySession(log_root=self.dir / 'logs')
        session.reset(agent_key='v5', human_color=WHITE, seed=7)
        session.play_human(7, 8)
        session.undo()
        self.assertIsNone(session.db_game_id)
        self.assertFalse(self.db.exists())


if __name__ == '__main__':
    unittest.main()
