"""scripts/h7r_vt_measure.py (R2-S400-VT-MEASURE-1): the classes (S50 search contest: V-A prior top, V-B visit gap,
fewer than two visited children never V-B, the tree's choice against the final move); the value scale and sign on a
real PUCT search for black and for white; the position branch (its own key, a fake Piskvork engine for black and white
to move, replay checks, failures before and after the first move); the Q1 / Q2 statistics (conservative bounds,
unresolved bounds, the minimum conditions); the entry table; the fixed contract; the run on a temporary repository
(smoke selection, both branch kinds, full statistics, resume, HEAD / input checks before publishing); the desktop
blocks under real bash. The engine search and Rapfi are stand-ins where named; the rules and records are the tool's."""
import argparse
import copy
import json
from pathlib import Path
from random import Random
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7c_adjudicate as ad
import h7r_vt_measure as vt
from hybrid import h7r_d4 as d4
from renju import BLACK, Game

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'scripts' / 'fake_piskvork_engine.py')
DOUBLE_THREE = [[7, 7], [0, 0], [7, 8], [0, 14], [8, 9], [14, 0], [9, 9], [14, 14]]  # black to move, (7, 9) forbidden
QUIET = lambda *a, **k: None  # noqa: E731


def git(root, *args):
    import subprocess
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def engine(*extra):
    return ad.engine_spec(ROOT / 'docs/h7c/b2_program.json', '/nonexistent-rapfi', [sys.executable, FAKE, *extra],
                          allow_unregistered=True)


def search_set(move, visits, route='tree', tree_move=None, prior_top=None):
    return {'route': route, 'engine_move': move, 'visits': visits, 'tree_move': tree_move or move,
            'prior_top': prior_top, 'seconds': 0.1}


def row(s50, s400, value=0.0):
    return {'ply': 5, 'color': 'black', 'sets': {'S50': s50, 'S400': s400}, 'c5_value': value}


class ClassTest(unittest.TestCase):
    V = [[1, 1, 60, 0.1], [1, 2, 40, 0.0], [1, 3, 0, 0.0]]  # top two 60 / 40 of 100: a gap of exactly 20 %

    def test_contest_by_prior_top_and_visit_gap(self):
        same = search_set([1, 1], self.V, prior_top=[1, 1])
        k = vt.classify(row(same, search_set([1, 1], self.V)))
        self.assertEqual((k['tree'], k['contest_a'], k['contest_b'], k['contest'], k['move_diff']),
                         (True, False, True, True, False))  # 20 % is inside the contest (<=)
        wide = [[1, 1, 61, 0.1], [1, 2, 39, 0.0]]
        k = vt.classify(row(search_set([1, 1], wide, prior_top=[1, 1]), search_set([1, 2], wide)))
        self.assertEqual((k['contest_a'], k['contest_b'], k['contest'], k['move_diff']), (False, False, False, True))
        k = vt.classify(row(search_set([1, 1], wide, prior_top=[1, 2]), search_set([1, 1], wide)))
        self.assertEqual((k['contest_a'], k['contest_b'], k['contest']), (True, False, True))
        k = vt.classify(row(search_set([1, 1], wide, prior_top=None), search_set([1, 1], wide)))
        self.assertFalse(k['contest_a'])

    def test_fewer_than_two_visited_children_is_never_a_visit_contest(self):
        one = [[1, 1, 3, 0.2], [1, 2, 0, 0.0]]  # total 3, a gap of 3: one visited child only
        k = vt.classify(row(search_set([1, 1], one, prior_top=[1, 1]), search_set([1, 1], one)))
        self.assertEqual((k['tree'], k['contest_b'], k['contest']), (True, False, False))

    def test_only_tree_positions_are_classed_and_tree_choices_are_counted_apart(self):
        k = vt.classify(row(search_set([1, 1], [], route='stage1', prior_top=[1, 2]), search_set([1, 1], self.V)))
        self.assertEqual((k['tree'], k['contest']), (False, False))
        k = vt.classify(row(search_set([1, 1], self.V, tree_move=[1, 2], prior_top=[1, 1]),
                            search_set([1, 2], self.V)))
        self.assertEqual(k['tree_final_diff'], {'S50': True, 'S400': False})  # S50's root checks changed its move
        rows = [{**vt.classify(row(search_set([1, 1], self.V, tree_move=[1, 2], prior_top=[1, 1]),
                                   search_set([1, 1], self.V))), 'game_index': 0}]
        counts = vt.class_counts(rows)
        self.assertEqual((counts['V']['positions'], counts['V']['tree_final_diff']['S50']), (1, 1))


class ValueTest(unittest.TestCase):
    """The S400 root value is (Q + 1) / 2 for the side to move, Q the visit-weighted mean Q of the visited root
    children (a real ``search_tree_puct_value``: its children's mean values are from the root mover's view)."""

    def search(self, moves):
        from analysis.puct_value_v1 import search_tree_puct_value
        game = Game()
        for m in moves:
            game.play(*m)
        roots = sorted({(r, c) for r in range(15) for c in range(15)} - {tuple(m) for m in moves})
        near = [m for m in roots if any(abs(m[0] - a) <= 1 and abs(m[1] - b) <= 1 for a, b in moves)]
        _, children = search_tree_puct_value(game, near, 200, c_puct=1.5, prior='uniform', leaf_value=lambda s: 0.0,
                                             candidate_limit=12, neighborhood_radius=2, priority_top_k=4)
        return [[c.move[0], c.move[1], c.visits, c.mean_value] for c in children], game.to_play

    def test_a_win_for_the_side_to_move_is_above_one_half_for_black_and_for_white(self):
        black_four = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [0, 6]]  # black: 7,6 or 7,11
        white_four = [[7, 7], [0, 0], [14, 14], [0, 1], [14, 12], [0, 2], [12, 14], [0, 3], [13, 10]]  # white: 0,4
        for moves, colour in ((black_four, BLACK), (white_four, -BLACK)):
            visits, to_play = self.search(moves)
            self.assertEqual(to_play, colour)
            self.assertGreater(vt.root_value(visits), 0.9)

    def test_scale_and_visited_children_only(self):
        self.assertEqual(vt.root_value([[0, 0, 3, 1.0], [0, 1, 1, -1.0], [0, 2, 0, 0.7]]), 0.75)
        self.assertEqual((vt.network_value(-1.0), vt.network_value(0.0), vt.network_value(1.0)), (0.0, 0.5, 1.0))
        with self.assertRaisesRegex(ValueError, 'without root visits'):
            vt.root_value([[0, 0, 0, 0.5]])


class PositionBranchTest(unittest.TestCase):
    def task(self, moves, spec=None):
        return {'key': vt.position_key(moves), 'order': 0, 'identity_sha256': 'i' * 64, 'moves': moves,
                'spec': spec or engine('--seed', '1')}

    def test_keys_are_apart_from_move_branches(self):
        moves = [[7, 7], [7, 8]]
        self.assertNotEqual(vt.position_key(moves), ad.branch_key(moves, [8, 8]))
        self.assertNotEqual(vt.position_key(moves), vt.position_key(moves[:1]))
        self.assertEqual(vt.position_key(moves), vt.position_key([list(m) for m in moves]))

    def test_black_and_white_to_move_are_scored_for_the_side_to_move_and_replayed(self):
        policy = d4.POLICY
        for moves, mover in (([[7, 7], [7, 8]], 'black'), ([[7, 7], [7, 8], [8, 8]], 'white')):
            t = self.task(moves)
            r = vt.play_position(t)
            self.assertEqual((r['status'], r['mover'], r['kind'], r['ply']), ('VALID', mover, 'position', len(moves)))
            out = vt.validate_position(r, t, policy)
            g = Game()
            for m in moves + r['continuation']:
                g.play(*m)
            side = BLACK if mover == 'black' else -BLACK
            expected = 0.5 if g.winner is None else 1.0 if g.winner == side else 0.0
            self.assertEqual((out['kind'], out['score'], r['score']), ('resolved', expected, expected))
            for bad in ({**r, 'score': 1.0 - expected if expected != 0.5 else 1.0},
                        {**r, 'continuation': r['continuation'][:-1]}, {**r, 'mover': 'white' if mover == 'black'
                                                                      else 'black'}, {**r, 'kind': 'move'}):
                with self.assertRaises(ValueError):
                    vt.validate_position(bad, t, policy)

    def test_failures_before_and_after_the_first_move(self):
        policy = d4.POLICY
        t = self.task([[7, 7], [7, 8]], engine('--mode', 'crash'))
        r = vt.play_position(t)
        self.assertEqual((r['status'], r['reason'], r['continuation']), ('INVALID', 'engine_error', []))
        self.assertEqual(vt.validate_position(r, t, policy)['kind'], 'unresolved')
        with self.assertRaisesRegex(ValueError, 'data error'):
            vt.validate_position({**r, 'reason': 'truncated'}, t, policy)
        t = self.task(DOUBLE_THREE, engine('--forbid-lie'))
        r = vt.play_position(t)
        self.assertEqual((r['reason'], r['ply_at'], r['continuation']), ('forbid_mismatch', len(DOUBLE_THREE), []))
        self.assertEqual(vt.validate_position(r, t, policy)['kind'], 'unresolved')
        with self.assertRaisesRegex(ValueError, 'forbid_mismatch differs'):
            vt.validate_position({**r, 'ours_only': [[0, 7]]}, t, policy)
        t = self.task([[7, 7], [7, 8]], engine('--seed', '2', '--forbid-lie'))
        r = vt.play_position(t)  # white to move first: the lie shows on black's first turn, after a move
        if r['status'] == 'INVALID':
            self.assertTrue(r['continuation'])
            self.assertEqual(vt.validate_position(r, t, policy)['kind'], 'unresolved')

    def test_a_move_branch_keeps_the_r2_d4_checks(self):
        moves = [[7, 7], [7, 8]]
        t = {'key': ad.branch_key(moves, [8, 8]), 'order': 0, 'identity_sha256': 'i' * 64, 'moves': moves,
             'move': [8, 8], 'spec': engine('--seed', '3')}
        r = ad.play_branch(t)
        self.assertIn(vt.validate_move(r, t, d4.POLICY)['kind'], ('resolved', 'structural'))
        with self.assertRaises(ValueError):
            vt.validate_move({**r, 'move': [9, 9]}, t, d4.POLICY)


def counts(n_games, samples=400):
    return d4.multiplicities(n_games, samples, 7)


def outcome(score):
    return {'kind': 'resolved', 'score': score, 'reason': None}


UNRESOLVED = {'kind': 'unresolved', 'score': None, 'reason': 'timeout'}


class StatsTest(unittest.TestCase):
    def move_rows(self, n_games=40, per=3, a=1.0, b=0.0):
        rows, outcomes = [], {}
        for g in range(n_games):
            for i in range(per):
                ka, kb = f'a{g}-{i}', f'b{g}-{i}'
                rows.append({'game_index': g, 'move_diff': True, 'key_a': ka, 'key_b': kb})
                outcomes[ka], outcomes[kb] = outcome(a if i else 0.5), outcome(b if i else 0.5)
        return rows, outcomes

    def test_q1_reads_the_conservative_bound_and_the_minimum_conditions(self):
        rows, outcomes = self.move_rows()
        out = vt.move_stats(rows, outcomes, list(range(40)), counts(40))
        self.assertEqual((out['observations'], out['games'], out['label'], out['failures']), (120, 40, 'BETTER', []))
        worse = vt.move_stats(*self.move_rows(a=0.0, b=1.0), list(range(40)), counts(40))
        self.assertEqual(worse['label'], 'WORSE')
        few = vt.move_stats(*self.move_rows(n_games=20, per=6), list(range(40)), counts(40))
        self.assertEqual((few['label'], few['failures']), ('INCONCLUSIVE', ['TOO_FEW_GAMES']))
        rows, outcomes = self.move_rows()
        for k in [f'a{g}-1' for g in range(7)]:  # 7 of 120 unresolved (> 5 %), against S400
            outcomes[k] = UNRESOLVED
        out = vt.move_stats(rows, outcomes, list(range(40)), counts(40))
        self.assertIn('UNRESOLVED_OVER_5_PERCENT', out['failures'])
        self.assertEqual(out['label'], 'INCONCLUSIVE')
        lo, hi = out['mean_delta_bounds']
        self.assertLess(lo, hi)

    def value_rows(self, n_games=40, per=3, v_c5=0.5, v_s400=0.9, score=1.0):
        rows, outcomes = [], {}
        for g in range(n_games):
            for i in range(per):
                key = f'p{g}-{i}'
                rows.append({'game_index': g, 'position_key': key, 'v_c5': v_c5, 'v_s400': v_s400, 'v_s50': 0.6,
                             'label': 1.0})
                outcomes[key] = outcome(score)
        return rows, outcomes

    def test_q2_squared_error_against_the_fixed_result(self):
        rows, outcomes = self.value_rows()
        out = vt.value_stats(rows, outcomes, list(range(40)), counts(40))
        self.assertEqual((out['positions'], out['games'], out['label'], out['failures']), (120, 40, 'BETTER', []))
        self.assertAlmostEqual(out['mean_d_bounds'][0], 0.25 - 0.01)
        self.assertAlmostEqual(out['resolved_mse']['S400'], 0.01)
        self.assertAlmostEqual(out['resolved_mean_d_s50'], 0.25 - 0.16)
        self.assertAlmostEqual(out['hdev_label_mse']['c5'], 0.25)  # label +1: score 1 for the side to move
        worse = vt.value_stats(*self.value_rows(v_c5=0.9, v_s400=0.5), list(range(40)), counts(40))
        self.assertEqual(worse['label'], 'WORSE')

    def test_an_unresolved_result_bounds_d_by_its_two_ends(self):
        rows, outcomes = self.value_rows(n_games=1, per=1, v_c5=0.3, v_s400=0.8)
        outcomes['p0-0'] = UNRESOLVED
        out = vt.value_stats(rows, outcomes, [0], counts(1, 50))
        d0, d1 = 0.3 ** 2 - 0.8 ** 2, 0.7 ** 2 - 0.2 ** 2
        self.assertAlmostEqual(out['mean_d_bounds'][0], min(d0, d1))
        self.assertAlmostEqual(out['mean_d_bounds'][1], max(d0, d1))
        self.assertEqual(set(out['failures']), {'TOO_FEW_OBSERVATIONS', 'TOO_FEW_GAMES', 'UNRESOLVED_OVER_5_PERCENT'})
        rows, outcomes = self.value_rows(n_games=1, per=1, v_c5=0.3, v_s400=0.8)
        outcomes['p0-0'] = {'kind': 'structural', 'score': 0.5, 'reason': 'abnormal_end'}
        out = vt.value_stats(rows, outcomes, [0], counts(1, 50))
        self.assertAlmostEqual(out['mean_d_bounds'][0], 0.2 ** 2 - 0.3 ** 2)
        self.assertEqual(out['structural'], 1)

    def test_the_entry_table(self):
        self.assertEqual(vt.entry('BETTER', 'BETTER')['decision'], vt.ENTRY_OPEN)
        for q1, q2 in (('BETTER', 'INCONCLUSIVE'), ('INCONCLUSIVE', 'BETTER'), ('WORSE', 'BETTER'),
                       ('INCONCLUSIVE', 'INCONCLUSIVE')):
            out = vt.entry(q1, q2)
            self.assertEqual(out['decision'], 'NO_TRAINING_CONTRACT')
            self.assertEqual(out['worse'], ['Q1'] if q1 == 'WORSE' else [])


class FixedContractTest(unittest.TestCase):
    def test_the_committed_contract(self):
        c, _, eid = vt.load_contract(None, ROOT)
        self.assertEqual(c['approved'], {'prepare': False, 'smoke': False, 'full': False})
        self.assertEqual({k: v for k, v in c.items() if k != 'sources'},
                         {k: v for k, v in vt.contract_body(ROOT).items() if k != 'sources'})
        self.assertEqual((c['stats']['seed'], c['classification']['contest_visit_gap'], c['stats']['min_observations'],
                          c['stats']['min_games'], c['stats']['max_unresolved_share']), (20261151, '1/5', 100, 30, '1/20'))
        self.assertEqual([s['path'] for s in c['sources']], list(vt.SOURCES))
        with self.assertRaisesRegex(ValueError, 'the smoke stage is not approved'):
            vt.load_contract(None, ROOT, 'smoke')
        self.assertEqual(eid, vt.experiment_sha256(c))


# --- the run on a temporary repository ------------------------------------------------------------------------------

QUIET_POINTS = [[3, 3], [11, 11], [3, 11], [11, 3], [5, 1], [9, 13], [1, 5], [13, 9]]
OPENING = [[7, 7], [7, 8], [8, 7]]


def alt(move):
    return [move[0] + 1, move[1]] if move[0] < 12 else [move[0] - 1, move[1]]


# per game: ply -> (S50 visit split, prior top differs, S400 moves elsewhere)
SCENARIOS = {0: {6: ('wide', False, True)}, 1: {},
             2: {7: ('wide', True, True), 8: ('close', False, False)},
             3: {5: ('close', False, True), 9: ('wide', True, False)}}


def fake_game(index):
    plies, moves = [], [list(m) for m in OPENING]
    for i, m in enumerate(QUIET_POINTS):
        ply = len(OPENING) + i
        split, _, _ = SCENARIOS[index].get(ply, ('wide', False, False))
        n1, n2 = (60, 40) if split == 'close' else (90, 10)
        plies.append({'ply': ply, 'color': 'black' if ply % 2 == 0 else 'white', 'route': 'tree', 'engine_move': m,
                      'played': m, 'visits': [[m[0], m[1], n1, 0.1], [alt(m)[0], alt(m)[1], n2, 0.0]]})
        moves.append(m)
    return {'game_index': index, 'game_seed': 100 + index, 'opening': [list(m) for m in OPENING], 'plies': plies,
            'moves': moves}


def fake_replay(tasks):
    out = []
    for t in tasks:
        g = t['game']
        plies = []
        for p in g['plies']:
            _, prior_differs, s400_differs = SCENARIOS[g['game_index']].get(p['ply'], ('wide', False, False))
            m = p['engine_move']
            s50 = search_set(m, p['visits'], prior_top=alt(m) if prior_differs else m)
            other = alt(m) if s400_differs else m
            s400 = search_set(other, [[other[0], other[1], 700, 0.3], [m[0], m[1], 100, 0.1]] if s400_differs else
                              [[m[0], m[1], 700, 0.3], [alt(m)[0], alt(m)[1], 100, 0.1]], prior_top=m)
            plies.append({'ply': p['ply'], 'color': p['color'], 'actor_route': p['route'], 'actor_move': m,
                          'sets': {'S50': s50, 'S400': s400}, 'c5_value': 0.2})
        out.append({'game_index': g['game_index'], 'plies': plies, 'identity_sha256': t['identity_sha256']})
    return out


def _winner(key, mover):
    return mover if int(key[:8], 16) % 3 else -mover


def fake_moves(tasks):
    from tests.h7r_helpers import winning_moves
    out = []
    for t in tasks:
        prefix = t['moves'] + [t['move']]
        g = Game()
        for m in t['moves']:
            g.play(*m)
        mover = g.to_play
        full = winning_moves(prefix, _winner(t['key'], mover))
        out.append({'key': t['key'], 'order': t['order'], 'identity_sha256': t['identity_sha256'],
                    'ply': len(t['moves']), 'mover': 'black' if mover == BLACK else 'white', 'move': t['move'],
                    'continuation': full[len(prefix):], 'forbid_checked': 0, 'status': 'VALID', 'reason': None,
                    'score': 1.0 if _winner(t['key'], mover) == mover else 0.0,
                    'winner': 'black' if _winner(t['key'], mover) == BLACK else 'white', 'length': len(full),
                    'seconds': 0.0})
    return out


def fake_positions(tasks):
    from tests.h7r_helpers import winning_moves
    out = []
    for t in tasks:
        g = Game()
        for m in t['moves']:
            g.play(*m)
        mover = g.to_play
        w = _winner(t['key'], mover)
        full = winning_moves(t['moves'], w)
        out.append({'key': t['key'], 'order': t['order'], 'identity_sha256': t['identity_sha256'], 'kind': 'position',
                    'ply': len(t['moves']), 'mover': 'black' if mover == BLACK else 'white',
                    'continuation': full[len(t['moves']):], 'forbid_checked': 0, 'status': 'VALID', 'reason': None,
                    'score': 1.0 if w == mover else 0.0, 'winner': 'black' if w == BLACK else 'white',
                    'length': len(full), 'seconds': 0.0})
    return out


class RunBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', root / '.gitignore')
        for rel in (vt.BASELINE, vt.SAMPLE):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        self.c = vt.contract_body(root, sources=())
        self.write_contract()
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.games = {i: fake_game(i) for i in SCENARIOS}
        self.manifest = {'format': 'h7r-d4-eligible-v1', 'games': [
            {'game_id': f'g{i}', 'game_index': i, 'plies': [p['ply'] for p in g['plies'] if p['ply'] >= 5]}
            for i, g in self.games.items()], 'rows': 24, 'warmup_plies': 8}
        self.labels = {(f'g{i}', p['ply']): 1.0 for i, g in self.games.items() for p in g['plies']}
        self.calls = {'pre': 0, 'replay': []}
        self.pre_changes = {}
        for target, fake in (('preflight', self.fake_preflight), ('replay_chunk', self.counted_replay),
                             ('move_chunk', fake_moves), ('position_chunk', fake_positions)):
            p = mock.patch.object(vt, target, fake)
            p.start()
            self.addCleanup(p.stop)

    def write_contract(self):
        path = self.root / vt.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.c, indent=1) + '\n')

    def approve(self, *stages):
        for s in stages:
            self.c['approved'][s] = True
        self.write_contract()
        git(self.root, 'commit', '-qam', 'approve ' + ' '.join(stages))

    def commit(self, rel):
        git(self.root, 'add', rel)
        git(self.root, 'commit', '-qm', rel)

    def fake_preflight(self, c, args, root):
        self.calls['pre'] += 1
        out = {'manifest': copy.deepcopy(self.manifest), 'games': self.games, 'hdev': {'contract_sha256': 'h' * 64},
               'labels': self.labels, 'model': {'value_checkpoint_sha256': 'm' * 64},
               'runtime': {'engine_hash': 'e' * 64, 'runtime_hash': 'r' * 64},
               'rapfi': {'identity': {'registered_engine': True}}}
        for key, change in self.pre_changes.items():
            if self.calls['pre'] >= key[0]:
                change(out)
        return out

    def counted_replay(self, tasks):
        self.calls['replay'] += [t['game_index'] for t in tasks]
        if getattr(self, 'replay_fails', None) in [t['game_index'] for t in tasks]:
            self.replay_fails = None
            raise RuntimeError('stand-in crash')
        return fake_replay(tasks)

    def args(self, mode=None, **k):
        a = {'contract': None, 'cache': 'cache', 'device': 'cuda', 'rapfi_dir': 'rapfi', 'log_name': 'x.log',
             'workers': 1, 'threads_for_tests': True}
        if mode:
            a.update(mode=mode, out_dir=self.root / self.c['output']['raw'].format(mode=mode))
        return argparse.Namespace(**{**a, **k})

    def summary(self, mode):
        return vt.record_path(self.c, mode, self.root)

    def through_prepare(self):
        self.approve('prepare')
        rec = vt.prepare(self.args(), self.root)
        self.assertTrue(rec['passed'])
        self.commit(self.c['output']['prepare'])

    def through_smoke(self):
        self.through_prepare()
        self.approve('smoke')
        out = vt.run(self.args('smoke'), self.root)
        self.commit(str(self.summary('smoke').relative_to(self.root)))
        return out


class RunTest(RunBase):
    def test_prepare_smoke_full(self):
        self.through_prepare()
        with self.assertRaisesRegex(ValueError, 'exists: never overwritten'):
            vt.prepare(self.args(), self.root)
        self.approve('smoke')
        smoke = vt.run(self.args('smoke'), self.root)
        # games 0 and 1 have no S400 / S50 disagreement in V: game 2 is added (the replay alone decides)
        self.assertEqual((smoke['reading'], smoke['games'], smoke['branches']),
                         ('SMOKE', [0, 1, 2], {'move': 2, 'position': 2}))
        self.assertEqual(self.calls['replay'], [0, 1, 2])
        self.assertTrue(smoke['actor_check']['reproduced'])
        self.assertEqual(smoke['classes']['V']['positions'], 2)
        self.assertEqual(smoke['projection']['V']['projected_positions'], round(2 / 18 * 24))
        self.commit(str(self.summary('smoke').relative_to(self.root)))
        with self.assertRaisesRegex(ValueError, 'the full stage is not approved'):
            vt.run(self.args('full'), self.root)
        self.approve('full')
        full = vt.run(self.args('full'), self.root)
        cl = full['classes']
        self.assertEqual((cl['E']['positions'], cl['T']['positions'], cl['V']['positions'], cl['V']['move_diff'],
                          cl['T']['move_diff'], cl['V-A']['positions'], cl['V-B']['positions']), (24, 24, 4, 2, 3, 2, 2))
        # the fake games share their move order: a position met in several games is one branch (as in R2-D4)
        self.assertEqual(full['branches'], {'move': 6, 'position': 6})
        self.assertEqual((full['Q1']['V']['observations'], full['Q1']['T']['observations'],
                          full['Q2']['V']['positions'], full['Q2']['T']['positions']), (2, 3, 4, 24))
        self.assertEqual(full['readings'], {'Q1': 'INCONCLUSIVE', 'Q2': 'INCONCLUSIVE'})  # below 100 / 30
        self.assertIn('TOO_FEW_OBSERVATIONS', full['Q2']['V']['failures'])
        self.assertEqual(full['entry']['decision'], 'NO_TRAINING_CONTRACT')
        self.assertEqual(json.loads(self.summary('full').read_text()), full)
        with self.assertRaisesRegex(ValueError, 'exists: never overwritten'):
            vt.run(self.args('full'), self.root)

    def test_stage_order(self):
        self.approve('prepare', 'smoke')
        with self.assertRaisesRegex(ValueError, 'prepare record is not committed'):
            vt.run(self.args('smoke'), self.root)
        self.c['approved']['full'] = True
        self.write_contract()
        git(self.root, 'commit', '-qam', 'full')
        with self.assertRaisesRegex(ValueError, 'smoke summary is not committed'):
            vt.run(self.args('full'), self.root)
        self.c['approved'] = {'prepare': False, 'smoke': True, 'full': False}
        self.write_contract()
        git(self.root, 'commit', '-qam', 'out of order')
        with self.assertRaisesRegex(ValueError, 'approvals: prepare, smoke, full in that order'):
            vt.run(self.args('smoke'), self.root)

    def test_resume_after_an_interruption(self):
        self.through_prepare()
        self.approve('smoke')
        self.replay_fails = 1
        with self.assertRaisesRegex(RuntimeError, 'stand-in crash'):
            vt.run(self.args('smoke'), self.root)
        self.assertFalse(self.summary('smoke').exists())
        calls = {'positions': 0}

        def flaky_positions(tasks):
            calls['positions'] += 1
            if calls['positions'] == 2:
                raise RuntimeError('stand-in Rapfi crash')
            return fake_positions(tasks)
        with mock.patch.object(vt, 'position_chunk', flaky_positions):
            with self.assertRaisesRegex(RuntimeError, 'Rapfi crash'):
                vt.run(self.args('smoke'), self.root)
            journal = self.root / self.c['output']['raw'].format(mode='smoke') / 'position_branches.jsonl'
            done = len(journal.read_text().splitlines()) if journal.exists() else 0
            self.assertLessEqual(done, 1)  # a finished branch is journaled or, at worst, played again
            out = vt.run(self.args('smoke'), self.root)
        self.assertEqual(self.calls['replay'], [0, 1, 1, 2])  # game 0 once; game 1 again after its crash
        self.assertEqual(calls['positions'], 2 + (2 - done))  # only the branches not journaled are played again
        self.assertEqual(out['branches'], {'move': 2, 'position': 2})

    def test_a_resume_from_another_head_is_refused(self):
        self.through_prepare()
        self.approve('smoke')
        self.replay_fails = 1
        with self.assertRaises(RuntimeError):
            vt.run(self.args('smoke'), self.root)
        git(self.root, 'commit', '-q', '--allow-empty', '-m', 'moved')
        with self.assertRaisesRegex(ValueError, 'resume identity differs'):
            vt.run(self.args('smoke'), self.root)

    def check_nothing_published(self, message, **patch):
        self.through_prepare()
        self.approve('smoke')
        with self.assertRaisesRegex(ValueError, message):
            if patch:
                with mock.patch.object(vt, 'position_chunk', patch['positions']):
                    vt.run(self.args('smoke'), self.root)
            else:
                vt.run(self.args('smoke'), self.root)
        self.assertFalse(self.summary('smoke').exists())

    def test_a_head_change_during_the_run_publishes_nothing(self):
        def moving(tasks):
            git(self.root, 'commit', '-q', '--allow-empty', '-m', 'moved during the run')
            return fake_positions(tasks)
        self.check_nothing_published('HEAD changed during the measurement', positions=moving)

    def test_an_input_change_during_the_run_publishes_nothing(self):
        for name, change in (('manifest', lambda o: o['manifest']['games'][0]['plies'].pop()),
                             ('model', lambda o: o['model'].update(value_checkpoint_sha256='0' * 64)),
                             ('rapfi', lambda o: o['rapfi']['identity'].update(registered_engine=False)),
                             ('hdev', lambda o: o['hdev'].update(contract_sha256='0' * 64))):
            with self.subTest(name):
                self.setUp()
                self.pre_changes = {(3, name): change}  # prepare, the run's start, then the check before publishing
                self.check_nothing_published('an input changed during the measurement')

    def test_a_prior_record_change_publishes_nothing(self):
        self.through_prepare()
        self.approve('smoke')

        def rewrite(tasks):
            path = self.root / self.c['output']['prepare']
            rec = json.loads(path.read_text())
            path.write_text(json.dumps({**rec, 'note': 'edited'}))
            return fake_positions(tasks)
        with mock.patch.object(vt, 'position_chunk', rewrite):
            with self.assertRaises(ValueError):
                vt.run(self.args('smoke'), self.root)
        self.assertFalse(self.summary('smoke').exists())


# --- the desktop blocks (section 13) under real bash ----------------------------------------------------------------

STUB = """import json, os, signal, subprocess, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-S400-VT-MEASURE-1'
cmd = sys.argv[1]
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
def kill():
    sys.stdout.flush()
    os.killpg(os.getpgid(0), signal.SIGKILL)
if os.environ.get('VT_FORBID'):
    print('STOP: must not run again'); sys.exit(9)
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if cmd == 'prepare':
    if os.environ.get('VT_FAIL') == 'prepare':
        print('STOP: ValueError: the Rapfi identity differs'); sys.exit(1)
    out, mode = Path(D, 'prepare.json'), 'prepare'
else:
    mode, raw = arg('--mode'), Path(arg('--out-dir'))
    raw.mkdir(parents=True, exist_ok=True)
    ident = raw / 'identity.json'
    if ident.exists():
        if json.loads(ident.read_text())['git']['commit'] != head:
            print('STOP: ValueError: resume identity differs'); sys.exit(1)
    else:
        ident.write_text(json.dumps({'git': {'commit': head, 'dirty': False}}))
    with (raw / 'replay.jsonl').open('a') as f:
        f.write('{}' + chr(10))
    if os.environ.get('VT_FAIL') == mode:
        print('STOP: RuntimeError: CUDA error'); sys.exit(1)
    out = Path(D, mode, 'summary.json')
out.parent.mkdir(parents=True, exist_ok=True)
lines = len((raw / 'replay.jsonl').read_text().splitlines()) if mode != 'prepare' else 0
out.write_text(json.dumps({'log_name': arg('--log-name'), 'mode': mode, 'head': head, 'journal_lines': lines}))
if os.environ.get('VT_KILL') == mode + ':after':
    kill()
print(mode, 'done')
"""


class BlockTest(unittest.TestCase):
    def setUp(self):
        import re
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.origin, self.work = base / 'origin.git', base / 'work'
        git(base, 'init', '-q', '--bare', str(self.origin))
        git(base, 'init', '-q', '-b', 'feat/h7-c1', str(self.work))
        w = self.work
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', w / '.gitignore')
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        (w / 'scripts/h7r_vt_measure.py').write_text(STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        (w / 'docs').mkdir()
        (w / 'docs/README').write_text('stand-in repository' + chr(10))
        hook = w / '.git/hooks/pre-commit'  # VT_KILL_AT_COMMIT: the block dies with its records staged
        hook.write_text('#!/bin/sh' + chr(10) + '[ -z "$VT_KILL_AT_COMMIT" ] || kill -9 0' + chr(10))
        hook.chmod(0o755)
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-S400-VT-MEASURE-1-draft.md').read_text()
        blocks = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 3)
        self.blocks = dict(zip(('vt-P', 'vt-S', 'vt-F'), (b.replace('~/renju-h6', w.as_posix()) for b in blocks)))
        self.D = 'docs/h7r/r2/R2-S400-VT-MEASURE-1'
        self.R = 'runs/h7r/R2-S400-VT-MEASURE-1'

    def bash(self, name, **env):
        import os
        import subprocess
        e = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e,
                           start_new_session=True)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def head(self):
        return git(self.work, 'rev-parse', 'HEAD')

    def logs(self, prefix):
        return [f for f in self.files(f'{self.D}/stage-logs') if Path(f).name.startswith(prefix) and f.endswith('.log')]

    def assert_pushed(self):
        self.assertEqual(self.head(), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        self.assertEqual(git(self.work, 'status', '--porcelain', '--untracked-files=no'), '')

    def through(self, *names):
        for n in names:
            code, out = self.bash(n)
            self.assertEqual(code, 0, out)

    def test_prepare_smoke_full(self):
        self.through('vt-P', 'vt-S', 'vt-F')
        for rel, prefix in (('prepare.json', 'prepare-'), ('smoke/summary.json', 'smoke-'), ('full/summary.json', 'full-')):
            self.assertTrue(self.files(f'{self.D}/{rel}'))
            [log] = self.logs(prefix)
            rec = json.loads((self.work / self.D / rel).read_text())
            self.assertEqual(rec['log_name'], Path(log).name)
        self.assertIn('full exit 0', git(self.work, 'log', '-1', '--format=%s'))
        self.assert_pushed()
        for n in ('vt-P', 'vt-S', 'vt-F'):
            code, out = self.bash(n)
            self.assertEqual(code, 1)
            self.assertIn('is already recorded', out)

    def test_a_failed_prepare_is_recorded(self):
        code, out = self.bash('vt-P', VT_FAIL='prepare')
        self.assertEqual(code, 1, out)
        self.assertIn('prepare exit 1', git(self.work, 'log', '-1', '--format=%s'))
        self.assertEqual(len(self.logs('prepare-')), 1)
        self.assertFalse(self.files(f'{self.D}/prepare.json'))
        self.assert_pushed()

    def test_a_failed_run_resumes_at_its_head_and_records_every_log(self):
        self.through('vt-P')
        pinned = self.head()
        code, out = self.bash('vt-S', VT_FAIL='smoke')
        self.assertEqual(code, 1, out)
        self.assertIn('nothing committed, HEAD kept', out)
        self.assertEqual(self.head(), pinned)
        git(self.work, 'commit', '-q', '--allow-empty', '-m', 'moved')
        code, out = self.bash('vt-S')
        self.assertEqual(code, 1)
        self.assertIn('HEAD moved since this smoke run started', out)
        git(self.work, 'reset', '-q', '--hard', pinned)
        code, out = self.bash('vt-S')
        self.assertEqual(code, 0, out)
        self.assertIn('resuming the smoke run', out)
        rec = json.loads((self.work / self.D / 'smoke/summary.json').read_text())
        self.assertEqual((rec['head'], rec['journal_lines']), (pinned, 2))  # resumed from its journal
        self.assertEqual(len(self.logs('smoke-')), 2)  # the failed attempt's log too
        self.assert_pushed()

    def test_a_summary_left_uncommitted_is_recorded_without_a_new_run(self):
        self.through('vt-P')
        for env in ({'VT_KILL': 'smoke:after'}, {'VT_KILL_AT_COMMIT': '1'}):
            with self.subTest(env):
                if self.files(f'{self.D}/smoke/summary.json'):
                    self.setUp()
                    self.through('vt-P')
                before = self.head()
                code, out = self.bash('vt-S', **env)
                self.assertNotEqual(code, 0, out)
                self.assertEqual(self.head(), before)
                code, out = self.bash('vt-S', VT_FORBID='1')
                self.assertEqual(code, 0, out)
                self.assertIn('recording it (no new run)', out)
                self.assertTrue(self.files(f'{self.D}/smoke/summary.json'))
                name = json.loads((self.work / self.D / 'smoke/summary.json').read_text())['log_name']
                lines = (self.work / self.D / 'stage-logs' / name).read_text().splitlines()
                self.assertEqual(lines[-1], '__EXIT__ 0')
                if 'VT_KILL' in env:  # killed before the exit line: recovered from the record
                    self.assertEqual(lines[-2], f'Recovered exit code from {self.D}/smoke/summary.json')
                self.assert_pushed()

    def test_a_summary_whose_log_disagrees_is_refused(self):
        self.through('vt-P')
        code, out = self.bash('vt-S', VT_KILL='smoke:after')
        self.assertNotEqual(code, 0, out)
        name = json.loads((self.work / self.D / 'smoke/summary.json').read_text())['log_name']
        log = self.work / self.R / 'logs' / name
        log.write_text(log.read_text() + '__EXIT__ 1' + chr(10))
        before = self.head()
        code, out = self.bash('vt-S', VT_FORBID='1')
        self.assertEqual(code, 1)
        self.assertIn('disagree on the exit code', out)
        self.assertEqual(self.head(), before)


class NativeReplayTest(unittest.TestCase):
    """The real replay worker: a stand-in self-play game of the real fast5 actor on CPU, replayed by the real S50
    (must reproduce the actor) and S400 agents; the added fields are read from the engines' own diagnostics."""

    def test_real_s50_and_s400_replay_with_stand_in_models(self):
        from rt30 import fast5
        if not fast5.available():
            self.skipTest('build native fast5 first')
        from tests.test_tb_v3_net import write_models
        import h7c_selfplay as sp
        from analysis.tb_v3_net import model_identity
        from hybrid.h6_fast import install
        import h7c_arena as ar
        try:
            with TemporaryDirectory() as tmp:
                policy, value = write_models(Path(tmp) / 'models', seed=5)
                sp._init_worker(str(policy), str(value), 'cpu')
                ident = model_identity(policy, value)
                t = {'identity': {**sp.run_identity(8, 8808, ident, 'cpu'),
                                  'generator_git': {'commit': 'fixture', 'dirty': False}},
                     'game_index': 0, 'game_seed': sp.game_seed(8808, 8, 0), 'opening_seed': sp.opening_seed(8808, 8, 0)}
                g = sp.play_game(t, max_plies=10)
                specs = {name: {'policy': str(policy), 'value': str(value), 'budget': k} for name, k in vt.SEARCHES.items()}
                out = vt.replay_chunk([{'game': g, 'specs': specs, 'device': 'cpu', 'identity_sha256': 'i'}])[0]
                vt.check_replay(out, {0: g})  # S50 reproduces every actor move, route and visit
                leaf = ar._loaded(str(policy), str(value), 'cpu')[1]
                position = Game()
                for m in g['opening']:
                    position.play(*m)
                tree = 0
                for p, original in zip(out['plies'], g['plies']):
                    self.assertAlmostEqual(p['c5_value'], float(leaf(position)), places=6)
                    for name, s in p['sets'].items():
                        if s['route'] == 'tree':
                            tree += 1
                            self.assertIsNotNone(s['prior_top'])
                            self.assertIsNotNone(s['tree_move'])
                            self.assertGreater(sum(v[2] for v in s['visits']), 0)
                            self.assertTrue(0.0 <= vt.root_value(s['visits']) <= 1.0)
                        else:
                            self.assertEqual((s['prior_top'], s['tree_move']), (None, None))
                    s50, s400 = (sum(v[2] for v in p['sets'][n]['visits']) for n in ('S50', 'S400'))
                    if p['sets']['S50']['route'] == p['sets']['S400']['route'] == 'tree':
                        self.assertGreater(s400, s50)  # the S400 tree budget
                    position.play(*original['played'])
                self.assertGreater(tree, 0)
                manifest = {'games': [{'game_id': 'g', 'game_index': 0,
                                       'plies': [p['ply'] for p in g['plies'] if p['ply'] >= 5]}]}
                rows = vt.table([out], {0: g}, manifest, {})
                self.assertEqual(len(rows), len(manifest['games'][0]['plies']))
        finally:
            fast5.uninstall()
            install('off')
            ar._MODELS.clear()
