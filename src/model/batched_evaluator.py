"""Stage 8-D batched evaluator: one CPU encode, one host->device copy, one forward.

``PolicyValueEvaluator`` encodes every snapshot separately on the model's device, which on
a GPU costs about ten tiny kernels and a small host->device copy per position. Here the
whole batch is encoded on the CPU into one contiguous tensor (the same planes as
``encode_game``), copied once (from a reused pinned buffer when the device is CUDA), run
through the network once and copied back once.

On the CPU the outputs equal ``PolicyValueEvaluator`` for the same batch bit for bit (same
input tensor, same forward, same masked softmax). On a GPU they differ from the CPU only in
float rounding (grade E1, docs/track-a-gpu-plan.md §1).

``submit``/``collect`` let ``search.scheduler`` keep advancing other searches while a batch
is on the device. The masked softmax here does not run the data-dependent checks of
``masking.mask_policy_logits`` (they would block on the device); a non-finite output still
fails ``validate_evaluation`` in the caller, and every snapshot has a legal move.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from renju import BLACK
from search.evaluator import EvaluationResult, EvaluationSnapshot

from .config import ACTION_COUNT, BOARD_SIZE, INPUT_PLANE_NAMES

PLANES = len(INPUT_PLANE_NAMES)


def encode_snapshots(snapshots: Sequence[EvaluationSnapshot]) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU ``[B,6,15,15]`` float32 planes and ``[B,225]`` bool masks, as ``encode_game``."""
    count = len(snapshots)
    board = torch.tensor([s.board for s in snapshots], dtype=torch.int8)
    to_play = torch.tensor([s.to_play for s in snapshots], dtype=torch.int8).view(count, 1, 1)
    masks = torch.zeros((count, ACTION_COUNT), dtype=torch.bool)
    legal = [b * ACTION_COUNT + r * BOARD_SIZE + c
             for b, s in enumerate(snapshots) for r, c in s.legal_moves]
    masks.view(-1)[legal] = True
    planes = torch.zeros((count, PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    planes[:, 0] = board == to_play
    planes[:, 1] = board == -to_play
    for b, s in enumerate(snapshots):
        if s.last_move is not None:
            planes[b, 2, s.last_move[0], s.last_move[1]] = 1
    planes[:, 3] = (to_play == BLACK).to(torch.float32)
    planes[:, 4] = 1
    planes[:, 5] = masks.view(count, BOARD_SIZE, BOARD_SIZE)
    return planes, masks


class _Pending:
    __slots__ = ('count', 'output', 'event')

    def __init__(self, count, output, event):
        self.count = count
        self.output = output
        self.event = event

    def done(self) -> bool:
        return self.event is None or self.event.query()


class BatchedEvaluator:
    """``Evaluator`` with batch encoding and optional asynchronous device execution.

    At most one submitted batch may be outstanding (the scheduler guarantees this per
    evaluator); its buffers are reused by the next submit.
    """

    def __init__(self, model, *, device: str | torch.device = 'cpu', pin_memory: bool | None = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.cuda = self.device.type == 'cuda'
        self.pin_memory = self.cuda if pin_memory is None else pin_memory
        self.calls = 0
        self.batches = 0
        self._capacity = 0
        self._inputs = self._masks = self._outputs = None
        self._outstanding = False

    def _reserve(self, count: int) -> None:
        if count <= self._capacity:
            return
        capacity = max(count, 2 * self._capacity)
        pin = self.pin_memory
        self._inputs = torch.empty((capacity, PLANES, BOARD_SIZE, BOARD_SIZE),
                                   dtype=torch.float32, pin_memory=pin)
        self._masks = torch.empty((capacity, ACTION_COUNT), dtype=torch.bool, pin_memory=pin)
        self._outputs = torch.empty((capacity, ACTION_COUNT + 1), dtype=torch.float32,
                                    pin_memory=pin)
        self._capacity = capacity

    def submit(self, snapshots: Sequence[EvaluationSnapshot]) -> _Pending:
        if not snapshots:
            raise ValueError('submit needs at least one snapshot')
        if self.model.training:
            raise RuntimeError('BatchedEvaluator requires model.eval()')
        if self._outstanding:
            raise RuntimeError('collect the previous batch before submitting another')
        count = len(snapshots)
        planes, masks = encode_snapshots(snapshots)
        if not self.cuda:
            with torch.inference_mode():
                logits, values = self.model(planes.to(self.device))
                priors = torch.softmax(logits.masked_fill(~masks.to(self.device), -torch.inf), dim=-1)
                output = torch.cat([priors, values.reshape(-1, 1)], dim=1).cpu()
            self._outstanding = True
            return _Pending(count, output, None)
        self._reserve(count)
        self._inputs[:count].copy_(planes)
        self._masks[:count].copy_(masks)
        with torch.inference_mode():
            x = self._inputs[:count].to(self.device, non_blocking=True)
            m = self._masks[:count].to(self.device, non_blocking=True)
            logits, values = self.model(x)
            priors = torch.softmax(logits.masked_fill(~m, -torch.inf), dim=-1)
            self._outputs[:count].copy_(torch.cat([priors, values.reshape(-1, 1)], dim=1),
                                        non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        self._outstanding = True
        return _Pending(count, self._outputs[:count], event)

    def collect(self, pending: _Pending) -> list[EvaluationResult]:
        if pending.event is not None:
            pending.event.synchronize()
        rows = pending.output.tolist()
        self._outstanding = False
        self.calls += pending.count
        self.batches += 1
        return [EvaluationResult(tuple(row[:ACTION_COUNT]), row[ACTION_COUNT]) for row in rows]

    def evaluate_batch(self, snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
        if not snapshots:
            return []
        return self.collect(self.submit(snapshots))

    def evaluate(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        return self.evaluate_batch([snapshot])[0]
