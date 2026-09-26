import random
import unittest
from dataclasses import replace
from functools import lru_cache

from renju import BLACK, WHITE, Game
from search.alphazero import SearchConfig
from search.evaluator import UniformEvaluator
from training.self_play import play_self_play_game

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.config import coordinate_to_action
    from model.encoding import encode_game
    from model.masking import legal_moves_to_mask
    from model.symmetry import transform_coordinate
    from training.dataset import (SampleValidationError, augment_batch, build_batch,
                                  samples_from_record, validate_samples)
    from training.replay_buffer import Batch, ReplayBuffer

# Black to play with forbidden points (same fixture as the Stage 5 neural tests).
FORBIDDEN_MOVES = ((7, 7), (14, 14), (3, 4), (14, 12), (3, 5), (14, 10), (1, 6), (14, 8),
                   (2, 6), (12, 14))
CONFIG = SearchConfig(num_simulations=2)


@lru_cache(maxsize=None)
def record_with_winner(winner):
    for seed in range(200):
        record = play_self_play_game(UniformEvaluator(), CONFIG, seed).record
        if record.winner == winner:
            return record
    raise AssertionError(f'no uniform self-play game with winner {winner}')


@lru_cache(maxsize=None)
def black_win_samples():
    return tuple(samples_from_record(record_with_winner(BLACK), generation=0, game_id=0))


def play(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


@unittest.skipIf(torch is None, 'requires torch')
class SamplesFromRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = record_with_winner(BLACK)
        cls.samples = samples_from_record(cls.record, generation=3, game_id=1)

    def test_one_valid_sample_per_move(self):
        self.assertEqual(len(self.samples), len(self.record.moves))
        validate_samples(self.samples)
        self.assertEqual([s.ply for s in self.samples], list(range(len(self.samples))))
        self.assertTrue(all(s.generation_id == 3 and s.game_id == 1 for s in self.samples))

    def test_state_matches_replayed_game(self):
        game = Game()
        for move, sample in zip(self.record.moves, self.samples):
            self.assertTrue(torch.equal(sample.state, encode_game(game)))
            self.assertTrue(torch.equal(sample.legal_mask,
                                        legal_moves_to_mask(game.legal_moves())))
            game.play(*divmod(move, 15))

    def test_policy_is_visit_distribution(self):
        for sample, stage5 in zip(self.samples, self.record.samples):
            total = sum(stage5.visit_counts)
            expected = torch.tensor([n / total for n in stage5.visit_counts])
            self.assertTrue(torch.equal(sample.policy, expected))
            self.assertEqual(float(sample.policy[~sample.legal_mask].abs().sum()), 0.0)
            self.assertAlmostEqual(float(sample.policy.sum()), 1.0, places=5)

    def test_z_is_side_to_move_perspective_black_win(self):
        # Plane 3 = current_is_black. Black won: black-to-move z=+1, white-to-move z=-1.
        for sample in self.samples:
            black_to_move = bool(sample.state[3, 0, 0])
            self.assertEqual(sample.value, 1.0 if black_to_move else -1.0)

    def test_z_is_side_to_move_perspective_white_win(self):
        samples = samples_from_record(record_with_winner(WHITE), generation=0, game_id=0)
        for sample in samples:
            self.assertEqual(sample.value, -1.0 if bool(sample.state[3, 0, 0]) else 1.0)


@unittest.skipIf(torch is None, 'requires torch')
class ValidateSamplesTest(unittest.TestCase):
    def setUp(self):
        self.sample = black_win_samples()[3]

    def assertRejected(self, sample):
        with self.assertRaises(SampleValidationError):
            validate_samples([sample])

    def test_illegal_policy_mass(self):
        policy = self.sample.policy.clone()
        illegal = int((~self.sample.legal_mask).nonzero()[0])
        legal = int(self.sample.legal_mask.nonzero()[0])
        policy[legal] -= 0.1
        policy[illegal] += 0.1
        self.assertRejected(replace(self.sample, policy=policy))

    def test_policy_sum(self):
        self.assertRejected(replace(self.sample, policy=self.sample.policy * 0.5))

    def test_negative_policy(self):
        policy = self.sample.policy.clone()
        legal = self.sample.legal_mask.nonzero().flatten()[:2].tolist()
        policy[legal[0]] -= 0.5
        policy[legal[1]] += 0.5
        policy[legal[0]] = -abs(policy[legal[0]]) - 0.1
        self.assertRejected(replace(self.sample, policy=policy))

    def test_value_domain(self):
        self.assertRejected(replace(self.sample, value=0.5))
        self.assertRejected(replace(self.sample, value=float('nan')))

    def test_non_finite_state(self):
        state = self.sample.state.clone()
        state[0, 0, 0] = float('inf')
        self.assertRejected(replace(self.sample, state=state))

    def test_terminal_empty_mask(self):
        self.assertRejected(replace(self.sample,
                                    legal_mask=torch.zeros(225, dtype=torch.bool)))

    def test_mask_shape_and_dtype(self):
        self.assertRejected(replace(self.sample, legal_mask=self.sample.legal_mask.float()))

    def test_mask_disagrees_with_legal_plane(self):
        mask = self.sample.legal_mask.clone()
        mask[int(mask.nonzero()[0])] = False
        self.assertRejected(replace(self.sample, legal_mask=mask))


@unittest.skipIf(torch is None, 'requires torch')
class AugmentationTest(unittest.TestCase):
    def test_forbidden_position_has_forbidden_points(self):
        game = play(FORBIDDEN_MOVES)
        self.assertEqual(game.to_play, BLACK)
        empty = sum(cell == 0 for row in game.board for cell in row)
        self.assertLess(len(game.legal_moves()), empty)

    def test_transformed_mask_matches_engine_on_transformed_board(self):
        for moves in (FORBIDDEN_MOVES, FORBIDDEN_MOVES + ((0, 0),), ((7, 7), (6, 8))):
            game = play(moves)
            mask = legal_moves_to_mask(game.legal_moves())
            state = encode_game(game, mask)
            policy = mask.float() / mask.sum()
            batch = Batch(state.unsqueeze(0), policy.unsqueeze(0), torch.zeros(1, 1),
                          mask.unsqueeze(0), torch.zeros(1, 3, dtype=torch.int64))
            for symmetry in range(8):
                rng = _FixedRandom(symmetry)
                augmented, used = augment_batch(batch, rng)
                self.assertEqual(used, [symmetry])
                # Replay the transformed history; the engine recomputes legality itself.
                transformed = play(tuple(transform_coordinate(r, c, symmetry)
                                         for r, c in moves))
                engine_mask = legal_moves_to_mask(transformed.legal_moves())
                self.assertTrue(torch.equal(augmented.legal_masks[0], engine_mask),
                                (moves[-1], symmetry))
                self.assertTrue(torch.equal(augmented.states[0],
                                            encode_game(transformed, engine_mask)))
                # Policy follows the same permutation as the mask.
                for r, c in game.legal_moves():
                    a = coordinate_to_action(r, c)
                    b = coordinate_to_action(*transform_coordinate(r, c, symmetry))
                    self.assertEqual(float(augmented.policies[0, b]), float(policy[a]))

    def test_build_batch_rng_separation(self):
        buffer = ReplayBuffer(512)
        buffer.extend(black_win_samples())
        sample_rng, augment_rng = random.Random(1), random.Random(2)
        plain = build_batch(buffer, 16, sample_rng=sample_rng, augment_rng=augment_rng,
                            augment=False)
        self.assertEqual(augment_rng.getstate(), random.Random(2).getstate())
        augmented = build_batch(buffer, 16, sample_rng=random.Random(1),
                                augment_rng=random.Random(2), augment=True)
        # Same indices (sample_rng only), values untouched by symmetry.
        self.assertTrue(torch.equal(plain.provenance, augmented.provenance))
        self.assertTrue(torch.equal(plain.values, augmented.values))
        validate_rows(self, augmented)

    def test_global_random_untouched(self):
        buffer = ReplayBuffer(512)
        buffer.extend(black_win_samples())
        before = random.getstate()
        build_batch(buffer, 8, sample_rng=random.Random(0), augment_rng=random.Random(0),
                    augment=True)
        self.assertEqual(before, random.getstate())


def validate_rows(test, batch):
    for policy, mask, state in zip(batch.policies, batch.legal_masks, batch.states):
        test.assertEqual(float(policy[~mask].abs().sum()), 0.0)
        test.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        test.assertTrue(torch.equal(state[5].reshape(-1), mask.float()))


class _FixedRandom(random.Random):
    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def randrange(self, *args, **kwargs):
        return self.value


if __name__ == '__main__':
    unittest.main()
