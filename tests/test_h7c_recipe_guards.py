"""Fail-closed VALUE_ONLY provenance checks; no torch, checkpoints or native runtime required."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

from hybrid import h7c_recipe as recipe
from scripts import h7c_arena as arena
from scripts import h7c_train_value as trainer


class CacheGamesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = self.root / 'rows.jsonl.gz'
        self.games = {'train': ['h7c:c3:8803:0', 'h7c:c3:8803:1'], 'val': [], 'test': ['h7c:c3:8803:2']}
        self.write_rows()
        for split, games in self.games.items():
            self.write_list(split, games)

    def write_list(self, split, games):
        (self.root / f'{split}_games.json').write_text(json.dumps(games), encoding='utf-8')

    def write_rows(self, games=None):
        with gzip.open(self.rows, 'wt', encoding='utf-8') as handle:
            for split, ids in (games or self.games).items():
                for game in ids:
                    for ply in (3, 4):
                        # One fully masked game must still be checked and remain in the game list.
                        handle.write(json.dumps({'game_id': game, 'split': split, 'ply': ply,
                                                 'value_mask': game.endswith(':1')}) + '\n')

    def check(self):
        trainer.check_cycle_cache(self.root, 3, 8803, rows=self.rows)

    def test_valid_rows_and_empty_split(self):
        self.check()

    def test_missing_game_list(self):
        (self.root / 'val_games.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing cache game list'):
            self.check()

    def test_empty_list_cannot_hide_rows(self):
        self.write_list('train', [])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.check()

    def test_other_seed_even_if_lists_claim_the_decision_seed(self):
        self.write_rows({'train': ['h7c:c3:8802:0']})
        with self.assertRaisesRegex(ValueError, 'not from'):
            self.check()

    def test_other_seed_in_fully_masked_game(self):
        self.write_rows({'train': ['h7c:c3:8803:0', 'h7c:c3:8802:1']})
        with self.assertRaisesRegex(ValueError, 'not from'):
            self.check()

    def test_reordered_duplicate_or_missing_game(self):
        for ids in (list(reversed(self.games['train'])), self.games['train'] * 2, self.games['train'][:1]):
            with self.subTest(ids=ids):
                self.write_list('train', ids)
                with self.assertRaises(ValueError):
                    self.check()

    def test_malformed_list(self):
        for value in ({'h7c:c3:8803:0': True}, [None]):
            with self.subTest(value=value):
                self.write_list('train', value)
                with self.assertRaisesRegex(ValueError, 'expected a list'):
                    self.check()

    def test_empty_rows(self):
        with gzip.open(self.rows, 'wt'):
            pass
        with self.assertRaisesRegex(ValueError, 'no self-play games'):
            self.check()

    def test_split_mismatch(self):
        self.write_rows({'val': self.games['train'], 'train': [], 'test': self.games['test']})
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.check()


class NoCandidateProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = {k: 'a' * 64 for k in recipe.MODEL_KEYS}
        self.history = []
        for cycle in (1, 2):
            path = self.root / f'c{cycle}.json'
            path.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': cycle,
                'parent': self.parent, 'next_champion': self.parent,
                'promotion': {'improvement_lower': -0.1}, 'tactical': {'verdict': 'PASS'}}), encoding='utf-8')
            self.history.append(path)
        self.decision = {'format': recipe.FORMAT, 'cycle': 3, 'recipe': recipe.VALUE_ONLY,
            'candidate_mode': recipe.VALUE_ONLY, 'parent': self.parent,
            'champion_record_sha256': recipe.file_sha256(self.history[-1]),
            'policy': {'action': 'freeze_parent', 'training': False}, 'retarget': False,
            'value': {'config_sha256': 'b' * 64, 'replay_anchor_labels_sha256': 'c' * 64},
            'seeds': {'selfplay_run_seed': 8803, 'arena': 8733, 'x2': 8713}}
        self.dp = self.root / 'decision.json'
        self.dp.write_text(json.dumps(self.decision), encoding='utf-8')
        self.sha = recipe.file_sha256(self.dp)
        self.record = {'format': 'h7c-value-train-v1', 'cycle': 3, 'verdict': 'NO_CANDIDATE',
            'git': {'commit': 'training-code', 'dirty': False}, 'inputs': {
                'parent_checkpoint_sha256': self.parent['value_checkpoint_sha256'],
                'parent_metadata_sha256': self.parent['value_metadata_sha256'],
                'config_sha256': 'b' * 64, 'selfplay_run_seed': 8803, 'cycle_labels_sha256': 'd' * 64,
                'replay_labels_sha256': 'c' * 64, 'recipe_decision_sha256': self.sha,
                'source': {'models': self.parent, 'selfplay_git_commit': 'selfplay-code',
                           'runtime_hash': 'e' * 64, 'rows': 100}}}
        self.rp = self.root / 'record.json'
        self.args = argparse.Namespace(cycle=3, recipe_decision=self.dp, policy_record=None, value_record=self.rp,
            parent_policy='policy.pt', parent_value='value.pt', champion=self.history[-1],
            history=self.history, allow_stand_in=False)
        # Only model file identity is substituted; the recipe, parent/history checks and writer are real.
        analysis = types.ModuleType('analysis')
        analysis.tb_v3_net = types.SimpleNamespace(model_identity=lambda *args: self.parent)
        analysis.tb_v3 = types.SimpleNamespace(MODELS=self.parent)
        patcher = mock.patch.dict(sys.modules, {'analysis': analysis})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_record(self, record=None, dirty=False):
        self.rp.write_text(json.dumps(self.record if record is None else record), encoding='utf-8')
        with mock.patch.object(arena, '_git', return_value={'commit': 'arena-code', 'dirty': dirty}):
            return arena.no_candidate(self.args)

    def test_valid_no_candidate_keeps_parent_and_breaks_convergence(self):
        out = self.run_record()
        self.assertEqual(out['next_champion'], self.parent)
        self.assertEqual(out['promotion']['decision'], 'KEEP_CHAMPION')
        self.assertIsNone(out['promotion']['improvement_lower'])
        self.assertFalse(out['convergence']['converged'])
        self.assertEqual(out['recipe_decision_sha256'], self.sha)
        self.assertEqual(out['git'], {'commit': 'arena-code', 'dirty': False})

    def test_dirty_or_unknown_execution_tree(self):
        for dirty in (True, None):
            with self.subTest(dirty=dirty), self.assertRaisesRegex(ValueError, 'tree is dirty'):
                self.run_record(dirty=dirty)

    def test_incomplete_or_foreign_training_record(self):
        changes = {
            'format': 'other', 'cycle': 2, 'verdict': 'CANDIDATE', 'git.dirty': True, 'git.commit': '',
            'inputs.parent_metadata_sha256': 'f' * 64, 'inputs.parent_checkpoint_sha256': 'f' * 64,
            'inputs.source.models': {}, 'inputs.source.selfplay_git_commit': '',
            'inputs.source.runtime_hash': None, 'inputs.source.rows': 0,
            'inputs.cycle_labels_sha256': None, 'inputs.replay_labels_sha256': 'f' * 64,
            'inputs.config_sha256': 'f' * 64, 'inputs.selfplay_run_seed': 8802,
            'inputs.recipe_decision_sha256': 'f' * 64}
        for field, value in changes.items():
            record = copy.deepcopy(self.record)
            node = record
            *parents, leaf = field.split('.')
            for key in parents:
                node = node[key]
            node[leaf] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_record(record)

    def test_minimal_old_record_is_not_evidence_of_this_recipe(self):
        with self.assertRaises(ValueError):
            self.run_record({'cycle': 3, 'verdict': 'NO_CANDIDATE', 'inputs': {
                'parent_checkpoint_sha256': self.parent['value_checkpoint_sha256'],
                'recipe_decision_sha256': self.sha}})


if __name__ == '__main__':
    unittest.main()
