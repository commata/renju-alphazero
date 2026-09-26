"""Stage 5 GameRecord -> TrainingSample adapter, invariants, and augmented batches.

States are rebuilt by replaying ``record.moves[:ply]`` through ``Game`` (Stage 5 stores
no tensors). The legal mask comes from ``Game.legal_moves()`` and is shared by encoder
plane 5 and the loss mask. ``z`` is Stage 5's ``Sample.z`` unchanged: it is already
from the sample's side-to-move perspective.
"""
from __future__ import annotations

from collections.abc import Sequence
from random import Random

import torch

from model.config import ACTION_COUNT, action_to_coordinate
from model.encoding import encode_game
from model.masking import legal_moves_to_mask
from model.symmetry import SYMMETRIES, transform_mask, transform_policy, transform_spatial
from renju import Game
from search.alphazero import visit_policy
from search.evaluator import POLICY_SUM_TOLERANCE

from .replay_buffer import STATE_SHAPE, Batch, ReplayBuffer, TrainingSample
from .self_play import GameRecord, replay_record

LEGAL_PLANE = 5


class SampleValidationError(ValueError):
    pass


def samples_from_record(record: GameRecord, *, generation: int, game_id: int
                        ) -> list[TrainingSample]:
    replay_record(record)  # Stage 5 contract: legality, visit sums, z, final result
    game = Game()
    samples = []
    for move, sample in zip(record.moves, record.samples):
        mask = legal_moves_to_mask(game.legal_moves())
        policy = torch.tensor(visit_policy(sample.visit_counts), dtype=torch.float32)
        samples.append(TrainingSample(encode_game(game, mask), policy, float(sample.z), mask,
                                      generation, game_id, sample.ply))
        game.play(*action_to_coordinate(move))
    return samples


def _fail(sample: TrainingSample, message: str) -> None:
    raise SampleValidationError(
        f'sample (gen {sample.generation_id}, game {sample.game_id}, ply {sample.ply}): '
        f'{message}')


def validate_samples(samples: Sequence[TrainingSample]) -> None:
    """Raise on the first invariant violation; never drop or repair samples."""
    for s in samples:
        if tuple(s.state.shape) != STATE_SHAPE or s.state.dtype != torch.float32:
            _fail(s, f'state must be float32 {STATE_SHAPE}')
        if not torch.isfinite(s.state).all():
            _fail(s, 'state must be finite')
        if tuple(s.policy.shape) != (ACTION_COUNT,) or s.policy.dtype != torch.float32:
            _fail(s, 'policy must be float32 (225,)')
        if tuple(s.legal_mask.shape) != (ACTION_COUNT,) or s.legal_mask.dtype != torch.bool:
            _fail(s, 'legal_mask must be bool (225,)')
        if not s.legal_mask.any():
            _fail(s, 'terminal/no-legal state (empty legal mask)')
        if not torch.equal(s.state[LEGAL_PLANE].reshape(-1), s.legal_mask.float()):
            _fail(s, 'encoder legal plane differs from legal_mask')
        if not torch.isfinite(s.policy).all() or (s.policy < 0).any():
            _fail(s, 'policy must be finite and non-negative')
        if (s.policy[~s.legal_mask] != 0).any():
            _fail(s, 'illegal action policy mass must be exactly 0')
        if abs(float(s.policy.sum()) - 1.0) > POLICY_SUM_TOLERANCE:
            _fail(s, 'policy must sum to 1')
        if type(s.value) is not float or s.value not in (-1.0, 0.0, 1.0):
            _fail(s, 'value z must be -1, 0 or +1')


def augment_batch(batch: Batch, rng: Random) -> tuple[Batch, list[int]]:
    """Apply one random D4 symmetry per row (identical to state, policy and mask).

    Uses the Stage 4 symmetry functions; values are invariant. One ``rng.randrange(8)``
    per row, in row order.
    """
    symmetries = [rng.randrange(len(SYMMETRIES)) for _ in range(batch.states.shape[0])]
    states = torch.stack([transform_spatial(x, k) for x, k in zip(batch.states, symmetries)])
    policies = torch.stack([transform_policy(p, k)
                            for p, k in zip(batch.policies, symmetries)])
    masks = torch.stack([transform_mask(m, k) for m, k in zip(batch.legal_masks, symmetries)])
    return Batch(states, policies, batch.values.clone(), masks, batch.provenance), symmetries


def build_batch(buffer: ReplayBuffer, batch_size: int, *, sample_rng: Random,
                augment_rng: Random, augment: bool) -> Batch:
    """Indices come only from ``sample_rng``; symmetries only from ``augment_rng``."""
    batch = buffer.get(buffer.sample_indices(batch_size, rng=sample_rng))
    if augment:
        batch, _ = augment_batch(batch, augment_rng)
    return batch
