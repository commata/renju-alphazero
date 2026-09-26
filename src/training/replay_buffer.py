"""Sample-count FIFO replay buffer backed by fixed-size tensor ring storage.

Only original (non-augmented) samples are stored. Sampling is uniform **with
replacement** and always uses a caller-supplied ``random.Random``; the module-level
``random`` state is never touched. Indices are *logical* (0 = oldest sample), so the
same RNG state selects the same samples before and after a serialization round-trip.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from random import Random

import torch

from model.config import ACTION_COUNT, BOARD_SIZE, INPUT_PLANE_NAMES

STATE_SHAPE = (len(INPUT_PLANE_NAMES), BOARD_SIZE, BOARD_SIZE)
BUFFER_FORMAT = 'stage6-replay-buffer-v1'


@dataclass(frozen=True)
class TrainingSample:
    """One training example; generation_id/game_id/ply are provenance, not features."""

    state: torch.Tensor        # [6,15,15] float32, Stage 4 encode_game output
    policy: torch.Tensor       # [225] float32, pi = N / sum(N)
    value: float               # z from the state's side-to-move perspective
    legal_mask: torch.Tensor   # [225] bool
    generation_id: int
    game_id: int
    ply: int


@dataclass(frozen=True)
class Batch:
    states: torch.Tensor       # [B,6,15,15] float32
    policies: torch.Tensor     # [B,225] float32
    values: torch.Tensor       # [B,1] float32 (Stage 4 value_loss shape)
    legal_masks: torch.Tensor  # [B,225] bool
    provenance: torch.Tensor   # [B,3] int64 (generation, game, ply)


class ReplayBuffer:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        # Encoder planes are exactly 0/1, so uint8 storage is lossless (4x smaller).
        self._states = torch.zeros((capacity, *STATE_SHAPE), dtype=torch.uint8)
        self._policies = torch.zeros((capacity, ACTION_COUNT), dtype=torch.float32)
        self._values = torch.zeros(capacity, dtype=torch.float32)
        self._masks = torch.zeros((capacity, ACTION_COUNT), dtype=torch.bool)
        self._provenance = torch.zeros((capacity, 3), dtype=torch.int64)
        self._start = 0      # physical slot of the oldest sample
        self._size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self._size

    def _physical(self, logical: torch.Tensor) -> torch.Tensor:
        return (logical + self._start) % self.capacity

    def append(self, sample: TrainingSample) -> None:
        self.extend([sample])

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        for sample in samples:
            state = sample.state
            if tuple(state.shape) != STATE_SHAPE or state.dtype != torch.float32:
                raise ValueError(f'state must be float32 {STATE_SHAPE}')
            stored = state.to(torch.uint8)
            if not torch.equal(stored.to(torch.float32), state):
                raise ValueError('state planes must be exactly 0 or 1')
            if self._size < self.capacity:
                slot = (self._start + self._size) % self.capacity
                self._size += 1
            else:  # FIFO: overwrite the oldest sample
                slot = self._start
                self._start = (self._start + 1) % self.capacity
            self._states[slot] = stored
            self._policies[slot] = sample.policy
            self._values[slot] = float(sample.value)
            self._masks[slot] = sample.legal_mask
            self._provenance[slot] = torch.tensor(
                [sample.generation_id, sample.game_id, sample.ply], dtype=torch.int64)
            self.total_added += 1

    def sample_indices(self, batch_size: int, *, rng: Random) -> list[int]:
        """Uniform logical indices with replacement: one ``rng.randrange`` per item."""
        if not isinstance(rng, Random):
            raise TypeError('rng must be a caller-owned random.Random')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if self._size == 0:
            raise ValueError('cannot sample from an empty replay buffer')
        return [rng.randrange(self._size) for _ in range(batch_size)]

    def get(self, indices: Sequence[int]) -> Batch:
        logical = torch.as_tensor(list(indices), dtype=torch.int64)
        if logical.numel() == 0 or logical.min() < 0 or logical.max() >= self._size:
            raise IndexError('replay buffer index out of range')
        slots = self._physical(logical)
        return Batch(self._states[slots].to(torch.float32), self._policies[slots].clone(),
                     self._values[slots].unsqueeze(1), self._masks[slots].clone(),
                     self._provenance[slots].clone())

    def _ordered(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self._physical(torch.arange(self._size))].clone()

    def state_dict(self) -> dict:
        """Filled samples in logical (oldest-first) order; weights_only-loadable."""
        return {
            'format_version': BUFFER_FORMAT,
            'capacity': self.capacity,
            'total_added': self.total_added,
            'states': self._ordered(self._states),
            'policies': self._ordered(self._policies),
            'values': self._ordered(self._values),
            'legal_masks': self._ordered(self._masks),
            'provenance': self._ordered(self._provenance),
        }

    def load_state_dict(self, data: dict) -> None:
        if data.get('format_version') != BUFFER_FORMAT:
            raise ValueError('unsupported replay buffer format')
        if data['capacity'] != self.capacity:
            raise ValueError(f"replay capacity mismatch: checkpoint {data['capacity']}, "
                             f'config {self.capacity}')
        size = data['states'].shape[0]
        if size > self.capacity:
            raise ValueError('replay buffer state exceeds capacity')
        expected = {'states': ((size, *STATE_SHAPE), torch.uint8),
                    'policies': ((size, ACTION_COUNT), torch.float32),
                    'values': ((size,), torch.float32),
                    'legal_masks': ((size, ACTION_COUNT), torch.bool),
                    'provenance': ((size, 3), torch.int64)}
        for key, (shape, dtype) in expected.items():
            if tuple(data[key].shape) != shape or data[key].dtype != dtype:
                raise ValueError(f'replay buffer field {key} has wrong shape/dtype')
        self.__init__(self.capacity)
        self._states[:size] = data['states']
        self._policies[:size] = data['policies']
        self._values[:size] = data['values']
        self._masks[:size] = data['legal_masks']
        self._provenance[:size] = data['provenance']
        self._size = size
        self.total_added = int(data['total_added'])

    def equals(self, other: ReplayBuffer) -> bool:
        """Content and order equality (physical ring offset is irrelevant)."""
        a, b = self.state_dict(), other.state_dict()
        return all(torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]
                   for k in a)
