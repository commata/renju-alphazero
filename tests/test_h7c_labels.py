"""H7-c self-play labels: H6 value rows, policy rows, probes, splits and the input checks."""
import copy
import unittest

from hybrid.h6_labels import canonical_hash
from renju import BLACK
from scripts import h7c_labels as lab
from scripts import h7c_selfplay as sp

# Black makes five on row 7 while white plays far away (the opening is black's centre and two moves).
MOVES = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [0, 6], [7, 11]]


def game(index=0, **changes):
    identity = sp.run_identity(1, 8801, {'policy_checkpoint_sha256': 'a' * 64, 'policy_metadata_sha256': 'b' * 64,
                                         'value_checkpoint_sha256': 'c' * 64, 'value_metadata_sha256': 'd' * 64},
                               'cuda')
    plies = []
    for ply in range(3, len(MOVES)):
        move = MOVES[ply]
        if ply == 8:  # black's five: a proven win
            target = {'format': 'h7c-target-v1', 'kind': 'proof_one_hot', 'kl': False, 'target': [[*move, 1.0]]}
            route, visits = 'stage1', []
        elif ply in (5, 6):  # forced moves: no target
            target, route, visits = {'format': 'h7c-target-v1', 'kind': 'none', 'kl': False, 'target': None}, \
                'stage2', []
        else:
            visits = [[*move, 40, 0.1], [3, 3, 10, 0.0]]
            target = {'format': 'h7c-target-v1', 'kind': 'visits', 'kl': True,
                      'target': [[3, 3, 0.2], [*move, 0.8]]}
            route = 'tree'
        plies.append({'ply': ply, 'color': 'black' if ply % 2 == 0 else 'white', 'route': route,
                      'engine_move': move, 'played': move, 'event': 'engine', 'seconds': 0.1, 'visits': visits,
                      'proven_loss': [], 'target': target})
    g = {**identity, 'game_index': index, 'game_seed': 1, 'opening_seed': 2, 'opening': MOVES[:3],
         'moves': [list(m) for m in MOVES], 'length': len(MOVES), 'result': 'black', 'winner': BLACK,
         'real_draw': False, 'truncated': False, 'events': {}, 'plies': plies, 'game_seconds': 1.0,
         'git': {'commit': 'x', 'dirty': False}}
    g.update(changes)
    return g


def build(games, probes=None, **kw):
    settings = {'probes': probes or {}, 'from_ply': 3, 'outcome_weight': 0.1, 'proof_weight': 1.0, 'prove_losses': False,
                    'vct1': False, 'probe_game_min_ply': 4, 'workers': 1, 'fast': 'off', 'progress_seconds': 1e9}
    return lab.build(games, **{**settings, **kw})


class LabelTest(unittest.TestCase):
    def test_rows(self):
        rows, policy, summary = build([game(0), game(1)])
        self.assertEqual(summary['games_kept'], 2)
        by = {(r.game_id, r.ply): r for r in rows}
        win = by[('h7c:c1:8801:0', 8)]  # black to move with an open four: a proof row, side-to-move +1
        self.assertEqual((win.value_target, win.label_kind.startswith('proof'), win.weight), (1, True, 1.0))
        white = by[('h7c:c1:8801:0', 3)]  # white to move in a game black won: outcome -1 at 0.1
        self.assertEqual((white.value_target, white.label_kind, white.weight), (-1, 'outcome', 0.1))
        self.assertEqual(white.engine_config_hash, game()['engine_hash'])
        self.assertEqual(white.model_hash, game()['model_hash'])
        kinds = [r['kind'] for r in policy if r['game_id'] == 'h7c:c1:8801:0']
        self.assertEqual(kinds, ['visits', 'visits', 'visits', 'proof_one_hot'])  # forced plies give no row
        one_hot = next(r for r in policy if r['kind'] == 'proof_one_hot')
        self.assertEqual((one_hot['kl'], one_hot['target']), (False, [[7, 11, 1.0]]))
        self.assertTrue(all(r['kl'] for r in policy if r['kind'] == 'visits'))
        self.assertEqual({r.split for r in rows}, {r['split'] for r in policy})
        # identical games: positions of a later split repeating train ones are masked in both row kinds
        if summary['policy']['unmasked_by_split'].get('val'):
            self.fail('identical games in train and val must be masked')

    def test_probe_games_and_rows(self):
        probe5 = {canonical_hash(MOVES[:5]): {'ply': 5, 'file': 'x', 'label': 'p'}}
        _, _, summary = build([game(0)], probes=probe5)
        self.assertEqual((summary['games_kept'], summary['dropped']['probe_game']), (0, 1))
        opening = {canonical_hash(MOVES[:3]): {'ply': 3, 'file': 'x', 'label': 'opening'}}
        rows, policy, summary = build([game(0)], probes=opening)
        self.assertEqual(summary['games_kept'], 1)  # a 3-stone opening probe masks its rows only
        self.assertIn('probe_position', next(r for r in rows if r.ply == 3).flags)
        self.assertTrue(next(r for r in policy if r['ply'] == 3)['mask'])
        with self.assertRaises(ValueError):  # a later probe may never be row-only
            build([game(0)], probes=probe5, probe_game_min_ply=6)

    def test_input_checks(self):
        bad = [
            [game(0, truncated=True, result='truncated')],
            [game(0, run_seed=8800)],                              # the throughput measurement's games
            [game(0), game(0)],                                    # one game twice
            [game(0), game(1, model_hash='f' * 64)],               # another model in the same cycle
            [game(0), game(1, cycle=2)],
        ]
        moved = game(0)
        moved['plies'][0] = {**moved['plies'][0], 'played': [9, 9]}
        bad.append([moved])                                        # the per-move records disagree with the moves
        for games in bad:
            with self.assertRaises(ValueError):
                build(games)

    def test_split_is_per_game_and_stable(self):
        rows, _, _ = build([game(i) for i in range(40)])
        splits = {}
        for r in rows:
            splits.setdefault(r.game_id, set()).add(r.split)
        self.assertTrue(all(len(s) == 1 for s in splits.values()))
        again, _, _ = build([copy.deepcopy(game(i)) for i in range(40)])
        self.assertEqual([r.split for r in rows], [r.split for r in again])


if __name__ == '__main__':
    unittest.main()
