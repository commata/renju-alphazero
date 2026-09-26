"""Stage 5 neural evaluator: the only Stage 5 module that touches torch/model code."""
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import torch

from search.evaluator import EvaluationResult, EvaluationSnapshot
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .encoding import encode_game
from .masking import legal_moves_to_mask, masked_softmax
from .network import PolicyValueNet


class _SnapshotView:
    """Read-only object exposing exactly what ``encode_game`` reads from a Game.

    It has no ``legal_moves()``: legality comes from the search-supplied snapshot, and
    the caller always passes the mask so encode_game never recomputes it.
    """

    __slots__ = ('board', 'history', 'to_play')

    def __init__(self, snapshot: EvaluationSnapshot):
        self.board = snapshot.board
        self.history = (snapshot.last_move,) if snapshot.last_move is not None else ()
        self.to_play = snapshot.to_play


class PolicyValueEvaluator:
    """Evaluate snapshots with a Stage 4 PolicyValueNet (eval mode, inference mode)."""

    def __init__(self, model: PolicyValueNet, *, device: str | torch.device = 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.calls = 0

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, config: ModelConfig = ModelConfig(),
                        device: str | torch.device = 'cpu') -> 'PolicyValueEvaluator':
        return cls(load_checkpoint(path, config, device=device), device=device)

    def encode(self, snapshot: EvaluationSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        """One mask per snapshot, shared by the encoder plane 5 and policy masking."""
        mask = legal_moves_to_mask(snapshot.legal_moves, device=self.device)
        return encode_game(_SnapshotView(snapshot), mask), mask

    def evaluate_batch(self, snapshots: Sequence[EvaluationSnapshot]) -> list[EvaluationResult]:
        if not snapshots:
            return []
        if self.model.training:
            raise RuntimeError('PolicyValueEvaluator requires model.eval()')
        encoded = [self.encode(snapshot) for snapshot in snapshots]
        planes = torch.stack([x for x, _ in encoded])
        masks = torch.stack([m for _, m in encoded])
        with torch.inference_mode():
            logits, values = self.model(planes)
            priors = masked_softmax(logits, masks)
        self.calls += len(snapshots)
        prior_rows = priors.cpu().tolist()
        value_rows = values.reshape(-1).cpu().tolist()
        return [EvaluationResult(tuple(row), value) for row, value in zip(prior_rows, value_rows)]

    def evaluate(self, snapshot: EvaluationSnapshot) -> EvaluationResult:
        return self.evaluate_batch([snapshot])[0]


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def create_random_checkpoint(path: str | Path, model_seed: int,
                             config: ModelConfig = ModelConfig()) -> str:
    """Save a seeded random-init checkpoint once; return the saved file's SHA256.

    The torch seed is fixed inside ``fork_rng`` so the caller's global RNG is preserved.
    Re-saving with the same seed may differ in bytes (git provenance is embedded), so the
    returned hash of the file actually written is the experiment input identity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = PolicyValueNet(config).eval()
    save_checkpoint(path, model)
    return file_sha256(path)
