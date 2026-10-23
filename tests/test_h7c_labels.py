"""H7-c self-play labels: H6 value rows, policy rows, probes, splits and the fail-closed input checks."""
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hybrid import h7c_targets as targets
from hybrid.h6_labels import canonical_hash
from renju import BLACK
from scripts import h7c_labels as lab
from scripts import h7c_selfplay as sp

# Black makes five on row 7 while white plays far away (the opening is black's centre and two moves).
MOVES = [[7, 7], [0, 0], [7, 8], [0, 2], [7, 9], [0, 4], [7, 10], [0, 6], [7, 11]]
GIT = {'commit': 'abc123', 'dirty': False}
# ply -> (route, event): plies 3-6 are the sampled opening plies, then the engine's moves; ply 8 is the five.
PLAN = {3: ('tree', 'sampled'), 4: ('tree', 'sampled'), 5: ('stage2', 'no_distribution'),
        6: ('stage2', 'no_distribution'), 7: ('tree', 'engine'), 8: ('stage1', 'engine')}


def game(index=0, generator_git=GIT, **changes):
    identity = sp.run_identity(1, 8801, {'policy_checkpoint_sha256': 'a' * 64, 'policy_metadata_sha256': 'b' * 64,
                                         'value_checkpoint_sha256': 'c' * 64, 'value_metadata_sha256': 'd' * 64},
                               'cuda')
    plies = []
    for ply in range(3, len(MOVES)):
        move, (route, event) = MOVES[ply], PLAN[ply]
        visits = [[*move, 40, 0.1], [3, 3, 10, 0.0]] if route == 'tree' else []
        plies.append({'ply': ply, 'color': 'black' if ply % 2 == 0 else 'white', 'route': route,
                      'engine_move': move, 'played': move, 'event': event, 'seconds': 0.1, 'visits': visits,
                      'proven_loss': [], 'target': targets.policy_target(route, move, visits, [])})
    g = {**identity, 'generator_git': generator_git, 'game_index': index, 'game_seed': 1, 'opening_seed': 2,
         'opening': MOVES[:3], 'moves': [list(m) for m in MOVES], 'length': len(MOVES), 'result': 'black',
         'winner': BLACK, 'real_draw': False, 'truncated': False, 'events': {}, 'plies': plies, 'game_seconds': 1.0}
    g.update(changes)
    return g


def build(games, probes=None, **kw):
    settings = {'probes': probes or {}, 'from_ply': 5, 'outcome_weight': 0.1, 'proof_weight': 1.0,
                'prove_losses': False, 'vct1': False, 'probe_game_min_ply': 4, 'workers': 1, 'fast': 'off',
                'progress_seconds': 1e9}
    return lab.build(games, **{**settings, **kw})


class LabelTest(unittest.TestCase):
    def test_rows(self):
        rows, policy, summary = build([game(0), game(1)], from_ply=3)
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

    def test_provenance_on_every_row(self):
        rows, policy, _ = build([game(0)])
        want = {'runtime_hash': game()['runtime_hash'], 'policy_checkpoint_sha256': 'a' * 64,
                'policy_metadata_sha256': 'b' * 64, 'value_checkpoint_sha256': 'c' * 64,
                'value_metadata_sha256': 'd' * 64, 'selfplay_git_commit': 'abc123'}
        for detail in [lab.value_row_dict(r)['h7c_provenance'] for r in rows] + [p['h7c_provenance'] for p in policy]:
            self.assertEqual({k: detail[k] for k in want}, want)
            self.assertIn('label_git_commit', detail)

    def test_value_rows_stay_h6_cache_compatible(self):
        """The extra provenance field must not break hybrid.h6_value_cache's reader."""
        from hybrid.h6_value_cache import compact_game, iter_games

        rows, policy, _ = build([game(0)])
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / 'value.jsonl.gz'
            lab.write_atomic({out: [lab.value_row_dict(r) for r in rows], Path(tmp) / 'p.jsonl.gz': policy})
            games = list(iter_games(out))
            self.assertEqual(len(games), 1)
            game_id, split, moves, labelled, masked = compact_game(games[0][1])
            self.assertEqual((game_id, len(labelled)), ('h7c:c1:8801:0', 4))  # plies 5-8 (default from_ply 5)

    def test_probe_games_and_rows(self):
        probe5 = {canonical_hash(MOVES[:5]): {'ply': 5, 'file': 'x', 'label': 'p'}}
        _, _, summary = build([game(0)], probes=probe5)
        self.assertEqual((summary['games_kept'], summary['dropped']['probe_game']), (0, 1))
        with self.assertRaises(ValueError):  # a later probe may never be row-only
            build([game(0)], probes=probe5, probe_game_min_ply=6)

    def test_opening_probe_masks_policy_rows_at_the_default_from_ply(self):
        """Policy rows start at ply 3, value rows at 5: a 3-stone opening probe must still mask its policy row."""
        opening = {canonical_hash(MOVES[:3]): {'ply': 3, 'file': 'x', 'label': 'opening'}}
        rows, policy, summary = build([game(0)], probes=opening)
        self.assertEqual(summary['games_kept'], 1)  # an opening probe masks its rows only
        self.assertEqual(min(r.ply for r in rows), 5)
        row3 = next(r for r in policy if r['ply'] == 3)
        self.assertEqual((row3['mask'], row3['flags']), (True, ['probe_position']))
        self.assertFalse(any(r['mask'] for r in policy if r['ply'] != 3))
        self.assertEqual(summary['policy']['masked']['probe_position'], 1)

    def test_input_checks(self):
        bad = {
            'truncated': [game(0, truncated=True, result='truncated')],
            'measurement seed': [game(0, run_seed=8800)],
            'one game twice': [game(0), game(0)],
            'two models': [game(0), game(1, model_hash='f' * 64)],
            'two cycles': [game(0), game(1, cycle=2)],
            'two generator commits': [game(0), game(1, generator_git={'commit': 'def456', 'dirty': False})],
            'dirty generator': [game(0, generator_git={'commit': 'abc123', 'dirty': True})],
            'unknown generator': [game(0, generator_git={'commit': None, 'dirty': None})],
            'wrong winner': [game(0, winner=-BLACK, result='white')],
        }
        moved = game(0)
        moved['plies'][0] = {**moved['plies'][0], 'played': [9, 9]}
        bad['moves vs records'] = [moved]
        occupied = game(0)
        occupied['moves'][3] = [7, 7]
        occupied['plies'][0] = {**occupied['plies'][0], 'played': [7, 7], 'engine_move': [7, 7]}
        bad['illegal move'] = [occupied]
        for name, games in bad.items():
            with self.assertRaises(ValueError, msg=name):
                build(games)

    def test_stored_targets_are_recomputed(self):
        """A target or sampling event the rule would not give is refused, never trained on."""
        def corrupt(ply, change):
            g = copy.deepcopy(game(0))
            p = next(p for p in g['plies'] if p['ply'] == ply)
            change(p)
            return [g]

        cases = {
            'probability changed': corrupt(3, lambda p: p['target']['target'][0].__setitem__(2, 0.5)),
            'kind changed': corrupt(7, lambda p: p['target'].update(kind='proof_one_hot')),
            'kl on a proof': corrupt(8, lambda p: p['target'].update(kl=True)),
            'target on a forced move': corrupt(5, lambda p: p['target'].update(
                kind='proof_one_hot', target=[[0, 4, 1.0]])),
            'proven loss kept': corrupt(3, lambda p: p.update(proven_loss=[[3, 3]])),
            'engine event in the opening plies': corrupt(4, lambda p: p.update(event='engine')),
            'sampled after the opening plies': corrupt(7, lambda p: p.update(event='sampled')),
        }
        for name, games in cases.items():
            with self.assertRaises(ValueError, msg=name):
                build(games)

    def test_split_is_per_game_and_stable(self):
        rows, _, _ = build([game(i) for i in range(40)])
        splits = {}
        for r in rows:
            splits.setdefault(r.game_id, set()).add(r.split)
        self.assertTrue(all(len(s) == 1 for s in splits.values()))
        again, _, _ = build([copy.deepcopy(game(i)) for i in range(40)])
        self.assertEqual([r.split for r in rows], [r.split for r in again])

    def test_outputs_are_atomic(self):
        with TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / 'value.jsonl.gz', Path(tmp) / 'policy.jsonl.gz'
            with self.assertRaises(TypeError):  # the second output cannot be written: neither appears
                lab.write_atomic({good: [{'a': 1}], bad: [{'b': object()}]})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [])
            lab.write_atomic({good: [{'a': 1}]}, (Path(tmp) / 'report.json', '{}\n'))
            with gzip.open(good, 'rt', encoding='utf-8') as handle:
                self.assertEqual([json.loads(x) for x in handle], [{'a': 1}])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['report.json', 'value.jsonl.gz'])


if __name__ == '__main__':
    unittest.main()
