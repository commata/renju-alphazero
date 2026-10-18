"""H6-4: the H6-2a' value head as the PUCT leaf (docs/h6-value.md, H6-4).

``ValueLeaf(checkpoint)`` is a callable ``game -> v`` in [-1, 1] from the side to move's view, which
``analysis.mcts_v8`` takes as ``leaf_value`` (``tree_mode="puct_value"``). It is the evaluator the H6-3
final gate passed: the frozen v3 checkpoint with **D4-averaged** inference (mean over the 8 symmetries).

Loading fails loudly (missing files, not an H6-2a value checkpoint, a checkpoint other than the expected
one): a value arm must never run with another evaluator. ``h4_policy.RootPolicy`` refuses value-trained
checkpoints, so this is a separate loader; the policy prior stays the H3 ``RootPolicy``.

Values are cached per exact position (stones, side to move, last move: the network's input), which
changes no result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from model.checkpoint import load_checkpoint
from model.config import ModelConfig
from model.encoding import encode_game
from renju import Game

from .h6_gate import d4_average
from .h6_value_train import value_forward

# H6-2a' (v3) best.pt: passed the H6-2a' dev deep-probe check and the H6-3 final gate (D4 average).
V3_CHECKPOINT_SHA256 = 'a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca'
INFERENCE = 'd4_average'
CACHE_LIMIT = 200_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class ValueLeaf:
    def __init__(self, checkpoint: str | Path, *, device: str = 'cpu', threads: int | None = 1,
                 expected_sha256: str | None = V3_CHECKPOINT_SHA256):
        checkpoint = Path(checkpoint)
        meta_path = checkpoint.with_name('best.json')
        if not checkpoint.exists() or not meta_path.exists():
            raise FileNotFoundError(f'value checkpoint or its best.json is missing: {checkpoint}, {meta_path}')
        self.sha256 = _sha256(checkpoint)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError(f'{checkpoint} is not the gated value checkpoint '
                             f'(sha256 {self.sha256[:8]}..., expected {expected_sha256[:8]}...)')
        self.meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if self.meta.get('stage') != 'H6-2a' or self.meta.get('value_trained') is not True:
            raise ValueError(f'{meta_path} is not an H6-2a value checkpoint')
        if threads:
            torch.set_num_threads(threads)  # benchmark workers run in parallel processes
        self.device = device
        self.model = load_checkpoint(checkpoint, ModelConfig(**self.meta['model_config']), device=device).eval()
        self._fn = d4_average(lambda planes: value_forward(self.model, planes), device)
        self.checkpoint = str(checkpoint)
        self._cache: dict = {}
        self.calls = 0
        self.cache_hits = 0
        self.seconds = 0.0

    @staticmethod
    def key(game: Game):
        return (tuple(map(tuple, game.board)), game.to_play, game.history[-1] if game.history else None)

    def __call__(self, game: Game) -> float:
        started = perf_counter()
        key = self.key(game)
        value = self._cache.get(key)
        if value is None:
            with torch.inference_mode():
                value = float(self._fn(encode_game(game).unsqueeze(0).to(self.device))[0])
            if len(self._cache) >= CACHE_LIMIT:
                self._cache.clear()
            self._cache[key] = value
        else:
            self.cache_hits += 1
        self.calls += 1
        self.seconds += perf_counter() - started
        return value

    def describe(self) -> dict:
        return {'checkpoint': self.checkpoint, 'sha256': self.sha256, 'inference': INFERENCE,
                'step': self.meta.get('step'), 'labels_sha256': self.meta.get('labels_sha256'),
                'device': self.device}
