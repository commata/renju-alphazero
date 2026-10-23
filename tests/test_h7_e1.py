"""analysis.vcf_tt_v1 and scripts/h7_e1_vcf_table.py: the no-win table is exact where the frozen VCF finishes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis import vcf_tt_v1 as tt
from analysis.threats import _unbounded_max_fours
from renju import BLACK, WHITE, Game
from scripts import h6_5d
from scripts import h7_e1_vcf_table as e1
from search.mcts_v7 import _find_vcf_with_stats as frozen

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))


def fours_board(rows, open_three=None):
    """White blocked threes (B W W W at both edges) on ``rows``: independent fours, so every order of them
    reaches the same states. ``open_three`` adds W W W at columns 6-8 of that row: a winning open four."""
    game = Game()
    for r in rows:
        game.board[r][0] = game.board[r][14] = BLACK
        for c in (1, 2, 3, 11, 12, 13):
            game.board[r][c] = WHITE
    if open_three is not None:
        for c in (6, 7, 8):
            game.board[open_three][c] = WHITE
    return game


def both(game, attacker, limit):
    mf = _unbounded_max_fours(game)
    stats = tt.TableStats()
    return (frozen(game, attacker, max_fours=mf, node_limit=limit),
            tt.make_find(stats)(game, attacker, max_fours=mf, node_limit=limit), stats, mf)


def line(result):
    return None if result[0] is None else (result[0].attack_moves, result[0].completion_points, result[0].fours)


class TableTest(unittest.TestCase):
    def test_same_as_frozen_when_frozen_finishes(self):
        cases = [(h6_5d.replay(WEB['moves'][:ply]), a, lim) for ply in (40, 80, 95) for a in (BLACK, WHITE)
                 for lim in (50, 5000)]
        cases += [(fours_board((0, 3)), WHITE, 1000), (fours_board((3, 6), open_three=0), WHITE, 10**6)]
        for game, attacker, limit in cases:
            want, got, _, _ = both(game, attacker, limit)
            if not want[2]:
                self.assertEqual(line(got), line(want))
                self.assertLessEqual(got[1], want[1])
                self.assertFalse(got[2])

    def test_transpositions_cost_no_nodes(self):
        want, got, stats, _ = both(fours_board((0, 3)), WHITE, 1000)
        self.assertEqual((want[1], got[1]), (632, 216))
        self.assertGreater(stats.hits, 0)
        want, got, stats, mf = both(fours_board((3, 6), open_three=0), WHITE, 10**6)
        self.assertEqual(line(got), line(want))  # same witness, found after fewer nodes
        self.assertLess(got[1], want[1])
        self.assertTrue(tt.verify_line(fours_board((3, 6), open_three=0), WHITE, got[0].attack_moves, mf))

    def test_finishes_where_frozen_is_cut(self):
        game = fours_board((0, 3, 6))
        want, got, _, mf = both(game, WHITE, 10_000)
        self.assertEqual((want[0], want[2]), (None, True))
        self.assertEqual((got[0], got[1], got[2]), (None, 2916, False))
        deep = frozen(game, WHITE, max_fours=mf, node_limit=100_000)  # the frozen search agrees with more nodes
        self.assertEqual((deep[0], deep[2]), (None, False))

    def test_table_is_per_call(self):
        stats = tt.TableStats()
        find = tt.make_find(stats)
        game = fours_board((0, 3))
        mf = _unbounded_max_fours(game)
        first = find(game, WHITE, max_fours=mf, node_limit=1000)
        second = find(game, WHITE, max_fours=mf, node_limit=1000)
        self.assertEqual(first[1], second[1])  # nothing carried over from the first call
        self.assertEqual(stats.calls, 2)

    def test_verify_line(self):
        game = fours_board((0, 3, 6), open_three=12)
        _, got, _, mf = both(game, WHITE, 10**6)
        moves = got[0].attack_moves
        self.assertGreater(len(moves), 1)
        self.assertTrue(tt.verify_line(game, WHITE, moves, mf))
        self.assertFalse(tt.verify_line(game, WHITE, moves[:-1], mf))
        self.assertFalse(tt.verify_line(game, WHITE, ((7, 7),) + moves[1:], mf))
        self.assertEqual(game.board, fours_board((0, 3, 6), open_three=12).board)  # restored


class CheckTest(unittest.TestCase):
    def test_modes_agree_on_web_positions(self):
        for ply in (92, 94):
            moves, move = WEB['moves'][:ply], WEB['moves'][ply]
            off = e1.run_check(moves, move, h6_5d.STANDARD, 'off')
            on = e1.run_check(moves, move, h6_5d.STANDARD, 'on')
            shadow = e1.run_check(moves, move, h6_5d.STANDARD, 'shadow')
            self.assertEqual({k: on[k] for k in e1.CHECK_KEYS}, {k: shadow[k] for k in e1.CHECK_KEYS})
            self.assertEqual(off['status'], on['status'])
            self.assertEqual(shadow['shadow']['uncut_mismatch'], 0)
            self.assertEqual(shadow['shadow']['line_fail'], 0)
            self.assertEqual(shadow['shadow']['calls'], on['calls'])
        from analysis import mcts_v8
        self.assertIs(mcts_v8._find_vcf_with_stats, frozen)  # swapped back

    def test_shadow_deep_check(self):
        shadow = e1.Shadow()
        game = fours_board((0, 3, 6))
        shadow(game, WHITE, max_fours=_unbounded_max_fours(game), node_limit=10_000)
        c = shadow.counts
        self.assertEqual((c['cut_to_none'], c['deep_confirmed'], c['uncut_mismatch']), (1, 1, 0))


def check(status, cut_kind=None, seconds=1.0):
    return {'status': status, 'nodes': 1, 'calls': 1, 'budget_cut': cut_kind in ('nodes', 'calls'),
            'cut_kind': cut_kind, 'seconds': seconds, 'table': {'calls': 1, 'stores': 1, 'hits': 0}}


def row(i, group, off, on, arm=h6_5d.CANDIDATE_ARM, d2_large=None, cut='nodes', on_cut=None, **extra):
    shadow = {'calls': 1, 'uncut_same': 1, 'uncut_mismatch': 0, 'cut_to_win': 0, 'cut_to_none': 0,
              'cut_to_cut': 0, 'frozen_nodes': 1, 'table_nodes': 1, 'line_fail': 0, 'deep_confirmed': 0,
              'deep_unconfirmed': 0, 'deep_contradicted': 0}
    on_check = check(on, on_cut if on == 'UNKNOWN' else None, extra.pop('on_seconds', 1.0))
    r = {'id': f'{group}{i}', 'arm': arm, 'game': f'g{i}', 'group': group, 'control_for': None,
         'd2_standard': None if group == 'safe_control' else off, 'd2_large': d2_large,
         'off': check(off, cut if off == 'UNKNOWN' else None), 'on': on_check,
         'shadow': {**on_check, 'shadow': shadow}, 'consistent': True}
    r.update(extra)
    return r


class SummaryTest(unittest.TestCase):
    def build(self, decided, n=20, **kw):
        rows = [row(i, 'unknown4m', 'UNKNOWN', 'SAFE' if i < decided else 'UNKNOWN', d2_large='UNKNOWN',
                    on_cut='calls', **kw) for i in range(n)]
        return rows + [row(i, 'safe_control', 'SAFE', 'SAFE') for i in range(n)]

    def test_verdicts(self):
        self.assertEqual(e1.summarize(self.build(6))['verdict'], 'go')
        self.assertEqual(e1.summarize(self.build(2))['verdict'], 'weak')
        self.assertEqual(e1.summarize(self.build(1))['verdict'], 'no_go')
        self.assertEqual(e1.summarize(self.build(6, on_seconds=3.0))['verdict'], 'cost_fail')
        rows = self.build(6)
        rows[0]['consistent'] = False
        self.assertEqual(e1.summarize(rows)['verdict'], 'tool_error')
        rows = self.build(6)
        rows[0]['shadow']['shadow']['uncut_mismatch'] = 1
        self.assertEqual(e1.summarize(rows)['verdict'], 'tool_error')
        rows = self.build(6)
        rows.append(row(99, 'resolved_safe', 'UNKNOWN', 'UNSAFE', d2_large='SAFE'))  # contradicts the 4M recheck
        self.assertEqual(e1.summarize(rows)['verdict'], 'unsound')
        rows = self.build(6)
        rows[0]['shadow']['shadow']['deep_contradicted'] = 1
        self.assertEqual(e1.summarize(rows)['verdict'], 'unsound')

    def test_effect_by_arm_and_migration(self):
        rows = [row(0, 'resolved_safe', 'UNKNOWN', 'SAFE', d2_large='SAFE'),
                row(1, 'resolved_safe', 'UNKNOWN', 'UNKNOWN', d2_large='SAFE', on_cut='calls'),
                row(2, 'unknown4m', 'UNKNOWN', 'UNKNOWN', arm=h6_5d.BASELINE_ARM, d2_large='UNKNOWN', cut='calls',
                    on_cut='calls'),
                row(3, 'unsafe4m', 'UNSAFE', 'UNSAFE', d2_large='UNSAFE')]
        d = e1.summarize(rows)['checks']['D_effect']
        self.assertEqual(d['pooled'], {'n': 1, 'of': 3, 'rate': 0.3333})
        self.assertEqual(d[h6_5d.CANDIDATE_ARM]['rate'], 0.5)
        self.assertEqual(d[h6_5d.BASELINE_ARM]['rate'], 0.0)
        self.assertEqual(d['cut_migration'], {'calls->calls': 1, 'nodes->calls': 1, 'nodes->resolved': 1})


class MainTest(unittest.TestCase):
    def test_end_to_end(self):
        def tree_move(ply, status):
            move = list(WEB['moves'][ply])
            return {'route': 'tree', 'ply': ply, 'v7_move': move, 'played': move, 'root': {'checked': [[move, status]]}}

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths, d2 = [], {'rows': []}
            for arm in (h6_5d.CANDIDATE_ARM, h6_5d.BASELINE_ARM):
                key = f'{arm}[c]@v8:full/8404/0/black'
                r = {'arm': arm, 'seed': 8404, 'opponent': 'v8:full', 'games': [
                    {'key': key, 'result': 'loss', 'length': 100, 'moves': WEB['moves'][:100],
                     'v8_moves': [tree_move(88, 'SAFE'), tree_move(94, 'UNSAFE')]}]}
                path = tmp / f'{arm}.json'
                path.write_text(json.dumps(r), encoding='utf-8')
                paths.append(path)
                d2['rows'].append({'id': f'{arm}|{key}|94', 'standard': {'status': 'UNSAFE'},
                                   'large': {'status': 'UNSAFE'}})
            (tmp / 'd2.json').write_text(json.dumps(d2), encoding='utf-8')
            out = tmp / 'e1.json'
            code = e1.main(['--runs', *map(str, paths), '--d2', str(tmp / 'd2.json'), '--output', str(out)])
            self.assertEqual(code, 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(len(result['rows']), 4)  # two unsafe4m + two matched controls
            s = result['summary']
            self.assertEqual(s['checks']['C_off_reproduction'], {'mismatches': [], 'shadow_inconsistent': []})
            self.assertEqual(s['checks']['B_soundness']['contradictions'], {})
            self.assertEqual(s['checks']['D_effect']['pooled']['of'], 0)


if __name__ == '__main__':
    unittest.main()
