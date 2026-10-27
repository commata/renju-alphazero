"""Statistics checked against direct SQL, with undo games, empty sets and KST boundaries."""
from contextlib import closing
from datetime import datetime, timezone
import json
import sqlite3
import unittest

from tests.test_web_admin import DatabaseFixture
from webadmin.games import list_games
from webadmin.statistics import duration_summary, game_statistics

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


class StatisticsTests(DatabaseFixture):
    def setUp(self):
        super().setUp()
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE games SET engine_config_hash='aaaa' WHERE game_id IN ('clean','undo')")
            conn.execute("UPDATE games SET engine_config_hash='bbbb' WHERE game_id IN ('stale','active')")
            conn.execute("UPDATE games SET started_at='2026-10-26T15:00:00+00:00' WHERE game_id='clean'")
            for seq, value in enumerate([1., 2., 3., 4., None, -1., 'bad'], 10):
                conn.execute('''INSERT INTO moves (game_id, seq, ply, player, actor, row0, col0,
                    recorded_at, engine_seconds, queue_seconds, wall_seconds, rt_json, undone_at)
                    VALUES ('undo', ?, ?, 'WHITE', 'AI', 0, 0, '2026-10-26T01:00:00+00:00', ?, ?, ?, ?, ?)''',
                             (seq, seq, value, value, value, '{"soft_limit_exceeded":true}' if seq == 10 else None,
                              '2026-10-26T02:00:00+00:00' if seq == 10 else None))

    def test_default_win_rate_and_undo_subgroup_match_sql(self):
        data = game_statistics(self.db, {}, now=NOW)
        with closing(sqlite3.connect(self.db)) as conn:
            finished, wins = conn.execute("SELECT count(*), sum(result='AI_WIN') FROM games WHERE status='finished'").fetchone()
            ai_moves = conn.execute("SELECT count(*) FROM moves WHERE actor NOT IN ('HUMAN','OPENING_RULE')").fetchone()[0]
        self.assertEqual(data['totals']['finished'], finished)
        self.assertEqual(data['totals']['ai_win_rate'], wins / finished)
        self.assertEqual(data['totals']['unknown_results'], 1)
        self.assertEqual(data['undo']['games'], 1)
        self.assertEqual(data['undo']['human_wins'], 1)
        self.assertEqual(data['ai_moves'], ai_moves)
        self.assertEqual(data['soft_limit_exceeded'], 1)  # includes the undone AI search
        self.assertEqual(data['totals']['suspected'], 1)
        self.assertTrue(data['policy']['include_undo_games'])
        self.assertNotIn('PRIVATE', json.dumps(data))

    def test_duration_metrics_ignore_missing_and_invalid_values(self):
        summary = game_statistics(self.db, {}, now=NOW)['timing']['engine_seconds']
        self.assertEqual((summary['count'], summary['missing'], summary['invalid']), (4, 1, 2))
        self.assertEqual((summary['mean'], summary['median'], summary['max']), (2.5, 2.5, 4))
        self.assertAlmostEqual(summary['p95'], 3.85)
        self.assertIsNone(duration_summary([None, float('inf')])['p95'])
        self.assertEqual(duration_summary([5])['p95'], 5)
        self.assertEqual(duration_summary([0])['mean'], 0)

    def test_filters_share_list_semantics_and_undo_default_can_be_excluded(self):
        for query in ({'undo': 'no'}, {'undo': 'yes'}, {'status': 'finished'}, {'color': 'WHITE'},
                      {'agent': 'tbv3', 'engine': 'aaaa'}, {'from': '2026-10-27', 'to': '2026-10-27'},
                      {'status': 'suspected'}, {'result': 'DRAW'}):
            data = game_statistics(self.db, query, now=NOW)
            self.assertEqual(data['totals']['games'], list_games(self.db, query, now=NOW)['total'])
        excluded = game_statistics(self.db, {'undo': 'no'}, now=NOW)
        self.assertEqual(excluded['totals']['finished'], 3)
        self.assertAlmostEqual(excluded['totals']['ai_win_rate'], 1 / 3)
        self.assertEqual(excluded['undo']['games'], 0)
        self.assertEqual(excluded['ai_moves'], 0)

    def test_engine_hash_groups_colors_and_kst_days(self):
        data = game_statistics(self.db, {}, now=NOW)
        groups = {(row['agent_key'], row['engine_config_hash']): row for row in data['by_engine']}
        self.assertEqual(groups['tbv3', 'aaaa']['finished'], 2)
        self.assertEqual(groups['tbv3', 'aaaa']['ai_win_rate'], .5)
        self.assertEqual(groups['tbv3', 'bbbb']['finished'], 0)
        self.assertEqual(data['by_color'][0]['human_color'], 'WHITE')
        days = {row['day']: row for row in data['daily']}
        self.assertEqual(days['2026-10-27']['games'], 1)  # UTC 15:00 is midnight KST
        self.assertEqual(days['2026-10-26']['games'], 7)

    def test_empty_set_and_bad_filter_or_json(self):
        data = game_statistics(self.db, {'agent': 'missing'}, now=NOW)
        self.assertIsNone(data['totals']['ai_win_rate'])
        self.assertEqual(data['by_engine'], [])
        self.assertEqual(data['ai_moves'], 0)
        for query in ({'from': '2026-12-01', 'to': '2026-01-01'}, {'to': '9999-12-31'}, {'engine': '../x'}):
            with self.assertRaises(ValueError):
                game_statistics(self.db, query, now=NOW)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("UPDATE moves SET rt_json='broken-json'")
        self.assertEqual(game_statistics(self.db, {}, now=NOW)['soft_limit_exceeded'], 0)


if __name__ == '__main__':
    unittest.main()
