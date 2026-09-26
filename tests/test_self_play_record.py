import dataclasses
import json
import random
import unittest

from model.config import action_to_coordinate
from renju import BLACK, WHITE, Game
from search.alphazero import SearchConfig
from search.evaluator import UniformEvaluator
from training.self_play import (DIVERSITY_START_PLY, GameRecord, ReplayError, Sample,
                                canonical_json_bytes, canonical_sha256, compute_z, config_hash,
                                game_hash, opening_diversity, play_self_play_game, record_hash,
                                replay_record, summarize_timing)

CONFIG = SearchConfig(num_simulations=4, temperature_moves=10)
SEED = 20260926


class SelfPlayRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(777)
        cls.global_before = random.getstate()
        cls.evaluator = UniformEvaluator()
        cls.game = play_self_play_game(cls.evaluator, CONFIG, SEED)
        cls.global_after = random.getstate()
        cls.record = cls.game.record

    def test_module_random_state_unchanged(self):
        self.assertEqual(self.global_before, self.global_after)

    def test_record_metadata(self):
        record = self.record
        self.assertEqual(record.format_version, 'stage5-record-v1')
        self.assertEqual(record.seed, SEED)
        self.assertEqual(record.search_config, CONFIG.to_dict())
        self.assertEqual(record.config_hash, config_hash(CONFIG))
        self.assertEqual(record.encoder_version, 'renju-relative-6p-v1')
        self.assertEqual(record.action_index_version, 'row-major-15x15-v1')
        self.assertIsNone(record.checkpoint_hash)
        self.assertIn('python', record.runtime_env)
        self.assertTrue(record.git_commit is None or len(record.git_commit) == 40)

    def test_samples_use_ply_not_prefix_copies(self):
        fields = {f.name for f in dataclasses.fields(Sample)}
        self.assertEqual(fields, {'ply', 'to_play', 'visit_counts', 'action', 'z'})
        self.assertEqual([s.ply for s in self.record.samples], list(range(len(self.record.moves))))
        self.assertEqual(tuple(s.action for s in self.record.samples), self.record.moves)

    def test_initial_center_one_hot_sample_kept(self):
        first = self.record.samples[0]
        self.assertEqual((first.ply, first.to_play, first.action), (0, BLACK, 112))
        self.assertEqual(sum(first.visit_counts), 1)
        self.assertEqual(first.visit_counts[112], 1)
        self.assertIn(first.z, (-1.0, 0.0, 1.0))
        self.assertTrue(self.game.move_stats[0].fast_path)
        self.assertEqual(self.game.move_stats[0].evaluator_calls, 0)

    def test_actions_legal_and_count_contract(self):
        game = Game()
        for sample in self.record.samples:
            legal = {r * 15 + c for r, c in game.legal_moves()}
            self.assertIn(sample.action, legal)
            self.assertEqual(sample.to_play, game.to_play)
            self.assertTrue(all(n == 0 for a, n in enumerate(sample.visit_counts) if a not in legal))
            self.assertEqual(sum(sample.visit_counts), 1 if len(legal) == 1 else 4)
            game.play(*action_to_coordinate(sample.action))

    def test_z_uses_winner_and_to_play(self):
        winner = self.record.winner
        for sample in self.record.samples:
            expected = 0.0 if winner is None else (1.0 if winner == sample.to_play else -1.0)
            self.assertEqual(sample.z, expected)
        self.assertEqual(compute_z(None, BLACK), 0.0)
        self.assertEqual(compute_z(WHITE, WHITE), 1.0)
        self.assertEqual(compute_z(WHITE, BLACK), -1.0)

    def test_replay_matches_final_state(self):
        replayed = replay_record(self.record, self.game.final_game)
        self.assertEqual(replayed.board, self.game.final_game.board)
        self.assertEqual(replayed.history, self.game.final_game.history)
        self.assertEqual(replayed.winner, self.record.winner)
        self.assertTrue(replayed.done)

    def test_json_round_trip(self):
        text = canonical_json_bytes(self.record.to_dict()).decode('utf-8')
        restored = GameRecord.from_dict(json.loads(text))
        self.assertEqual(restored, self.record)
        replay_record(restored, self.game.final_game)

    def test_same_seed_same_hashes(self):
        again = play_self_play_game(UniformEvaluator(), CONFIG, SEED).record
        self.assertEqual(game_hash(again), game_hash(self.record))
        self.assertEqual(record_hash(again), record_hash(self.record))
        self.assertEqual(again.moves, self.record.moves)

    def test_hash_excludes_runtime_env(self):
        changed = dataclasses.replace(self.record, runtime_env={'threads': 99})
        self.assertEqual(game_hash(changed), game_hash(self.record))
        self.assertEqual(record_hash(changed), record_hash(self.record))

    def test_tampered_records_rejected(self):
        samples = list(self.record.samples)
        second = samples[1]
        legal_counts = list(second.visit_counts)
        stone = 112  # occupied after the first move
        bad_counts = legal_counts.copy()
        bad_counts[stone] = 1
        index = next(a for a, n in enumerate(bad_counts) if n > 0 and a != stone)
        bad_counts[index] -= 1

        # The played action must come from the searched support, not merely be legal.
        unvisited_action_counts = list(second.visit_counts)
        removed = unvisited_action_counts[second.action]
        self.assertGreater(removed, 0)
        replay_state = Game()
        replay_state.play(*action_to_coordinate(self.record.moves[0]))
        replacement = next(
            a for a in (r * 15 + c for r, c in replay_state.legal_moves())
            if a != second.action
        )
        unvisited_action_counts[second.action] = 0
        unvisited_action_counts[replacement] += removed

        cases = {
            'illegal count': dataclasses.replace(second, visit_counts=tuple(bad_counts)),
            'played action unvisited': dataclasses.replace(
                second, visit_counts=tuple(unvisited_action_counts)),
            'wrong z': dataclasses.replace(second, z=-second.z if second.z else 1.0),
            'wrong to_play': dataclasses.replace(second, to_play=-second.to_play),
            'wrong sum': dataclasses.replace(
                second, visit_counts=tuple(n * 2 for n in second.visit_counts)),
        }
        for name, sample in cases.items():
            tampered = dataclasses.replace(
                self.record, samples=tuple(samples[:1] + [sample] + samples[2:]))
            with self.assertRaises(ReplayError, msg=name):
                replay_record(tampered)
        with self.assertRaises(ReplayError):
            replay_record(dataclasses.replace(self.record, config_hash='0' * 64))
        with self.assertRaises(ReplayError):
            replay_record(dataclasses.replace(self.record, winner=-(self.record.winner or 1)))
        truncated = dataclasses.replace(self.record, moves=self.record.moves[:-1],
                                        samples=self.record.samples[:-1])
        with self.assertRaises(ReplayError):
            replay_record(truncated)

    def test_timing_and_diversity_summary(self):
        summary = summarize_timing(self.game.move_stats)
        self.assertEqual(summary['fast_path']['count'], 1)
        self.assertEqual(summary['searched']['count'], len(self.record.moves) - 1)
        self.assertLessEqual(summary['searched']['evaluator_calls'], 5)  # terminals skip it
        diversity = opening_diversity([self.record], 10)
        self.assertEqual(DIVERSITY_START_PLY, 1)
        self.assertEqual((diversity['start_ply'], diversity['distinct_prefixes']), (1, 1))


class CanonicalHashTest(unittest.TestCase):
    def test_canonical_serialization(self):
        a = {'b': [1, 2], 'a': {'y': None, 'x': 1.5}, 'k': '흑'}
        b = {'k': '흑', 'a': {'x': 1.5, 'y': None}, 'b': [1, 2]}
        self.assertEqual(canonical_json_bytes(a), canonical_json_bytes(b))
        self.assertEqual(canonical_json_bytes(a),
                         '{"a":{"x":1.5,"y":null},"b":[1,2],"k":"흑"}'.encode('utf-8'))
        self.assertEqual(canonical_sha256(a), canonical_sha256(b))
        with self.assertRaises(ValueError):
            canonical_json_bytes({'x': float('nan')})

    def test_game_hash_payload(self):
        record = GameRecord('stage5-record-v1', 1, None, {}, '', None, None, None, '', '', {},
                            BLACK, (112, 113), ())
        self.assertEqual(game_hash(record), canonical_sha256(
            {'format': 'stage5-game-v1', 'winner': 1, 'moves': [112, 113]}))


if __name__ == '__main__':
    unittest.main()
