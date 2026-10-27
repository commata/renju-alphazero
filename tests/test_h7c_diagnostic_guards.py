"""Fail-closed regression checks for E2 sensitivity and SB evidence."""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from scripts import h7c_arena as ar, h7c_budget as sb, h7c_sensitivity as sn
from tests.test_h7c_budget import write_arm


class SensitivityGuards(unittest.TestCase):
    def setUp(self):
        self.game = {'game_index': 7, 'opening': [[7, 7], [6, 6], [8, 7]], 'plies': [
            {'ply': 3, 'color': 'white', 'route': 'tree', 'engine_move': [7, 8], 'played': [7, 8]},
            {'ply': 4, 'color': 'black', 'route': 'stage2', 'engine_move': [8, 8], 'played': [8, 8]}]}
        self.models = {'actor': {}, 'candidate': {}}
        self.record = {'game_index': 7, 'plies': [
            {'ply': p['ply'], 'color': p['color'], 'actor_route': p['route'], 'actor_move': p['engine_move'],
             'sets': {name: {'route': p['route'], 'engine_move': p['engine_move'],
                            'visits': [[*p['engine_move'], 50, .2]] if p['route'] == 'tree' else []}
                      for name in self.models}}
            for p in self.game['plies']]}
        for p, r in zip(self.game['plies'], self.record['plies']):
            p['visits'] = copy.deepcopy(r['sets']['actor']['visits'])

    def check(self, record):
        sn.check_replay(record, {7: self.game}, self.models, 'actor')

    def test_full_actor_replay_is_accepted(self):
        self.check(self.record)

    def test_empty_or_partial_ply_coverage_is_rejected(self):
        self.assertFalse(sn.actor_check([{'game_index': 7, 'plies': []}], {7: self.game}, 'actor')['reproduced'])
        for plies in ([], self.record['plies'][:1], list(reversed(self.record['plies']))):
            with self.assertRaises(ValueError):
                self.check({**self.record, 'plies': plies})

    def test_wrong_actor_move_or_route_is_rejected(self):
        for key, value in (('engine_move', [5, 5]), ('route', 'stage2')):
            record = copy.deepcopy(self.record)
            record['plies'][0]['sets']['actor'][key] = value
            with self.assertRaisesRegex(ValueError, 'reproduction failed'):
                self.check(record)

    def test_wrong_game_or_missing_model_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check({**self.record, 'game_index': 8})
        record = copy.deepcopy(self.record)
        del record['plies'][0]['sets']['candidate']
        with self.assertRaises(ValueError):
            self.check(record)

    def test_same_move_with_different_actor_visits_is_rejected(self):
        record = copy.deepcopy(self.record)
        record['plies'][0]['sets']['actor']['visits'][0][3] += .01
        with self.assertRaisesRegex(ValueError, 'reproduction failed'):
            self.check(record)

    def test_non_finite_or_duplicate_visits_are_rejected(self):
        for visits in ([[7, 8, 50, float('nan')]], [[7, 8, 50, .2], [7, 8, 10, .1]]):
            record = copy.deepcopy(self.record)
            record['plies'][0]['sets']['candidate']['visits'] = visits
            with self.assertRaises(ValueError):
                self.check(record)

    def test_illegal_counterfactual_move_is_rejected(self):
        record = copy.deepcopy(self.record)
        record['plies'][0]['sets']['candidate']['engine_move'] = [7, 7]
        with self.assertRaisesRegex(ValueError, 'illegal selected move'):
            self.check(record)

    def test_cli_does_not_publish_actor_failure(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'summary.json'
            argv = ['--games', 'g', '--evidence', 'e', '--model', 'a=p,v', '--compare', 'a:b',
                    '--jsonl', 'r', '--output', str(output)]
            with mock.patch.object(sn, 'run', side_effect=ValueError('actor reproduction failed')):
                with self.assertRaises(SystemExit) as caught:
                    sn.main(argv)
            self.assertNotEqual(caught.exception.code, 0)
            self.assertFalse(output.exists())


class BudgetGuards(unittest.TestCase):
    def test_duplicate_seed_color_opening_and_invalid_games_are_rejected(self):
        with TemporaryDirectory() as tmp:
            paths = [write_arm(tmp, k, [('win', 'loss')] * 50) for k in sb.FACTORS]
            original = json.loads(Path(paths[0]).read_text())
            changes = (
                lambda r: r['games'].append(copy.deepcopy(r['games'][0])),
                lambda r: r['games'].__setitem__(1, copy.deepcopy(r['games'][0])),
                lambda r: r['games'][0].update(seed=123),
                lambda r: r['games'][0].update(tested_color='white'),
                lambda r: r['games'][0].update(moves=[[0, 0]]),
                lambda r: r['games'][0].update(illegal={'side': 'tested'}, result='loss'),
            )
            for change in changes:
                record = copy.deepcopy(original)
                change(record)
                # Recompute the derived outputs: rejection must come from source-game checks.
                record['summary'] = sb.summarize_arm(record['games'])
                record['fingerprint'] = ar.fingerprint(record['games'])
                Path(paths[0]).write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    sb.summary(argparse.Namespace(arms=paths, positions=None))

    def test_positions_of_other_models_or_seed_are_rejected(self):
        with TemporaryDirectory() as tmp:
            paths = [write_arm(tmp, k, [('win', 'loss')] * 50) for k in sb.FACTORS]
            ident = json.loads(Path(paths[0]).read_text())['identity']
            positions = Path(tmp) / 'positions.json'
            positions.write_text(json.dumps({'identity': {**ident, 'seed': 8742}, 'summary': {}}))
            with self.assertRaisesRegex(ValueError, 'positions and SB arms differ'):
                sb.summary(argparse.Namespace(arms=paths, positions=positions))

    def test_champion_must_follow_the_verdict(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v.json'
            path.write_text(json.dumps({'format': 'h7c-arena-v1-verdict',
                                       'promotion': {'decision': 'KEEP_CHAMPION'},
                                       'parent': {'value': 'parent'}, 'next_champion': {'value': 'candidate'}}))
            with self.assertRaisesRegex(ValueError, 'next champion'):
                sb.champion_models(path)


if __name__ == '__main__':
    unittest.main()
