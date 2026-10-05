"""H4: the H3 policy as a root scorer for V8 (docs/mcts-v8-teacher.md §12.13).

``RootPolicy(checkpoint)`` is a callable ``game -> {move: probability}`` over the legal
moves (masked softmax), which ``analysis.mcts_v8`` takes as ``root_policy``. V8 never
imports torch; only this adapter does. The value head is never run (H3 trained the
policy only: ``value_trained: false``).

Loading fails loudly (missing checkpoint, missing metadata, a value-trained flag that
is not false, wrong architecture): a policy arm must never fall back to the baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch

from model.checkpoint import load_checkpoint
from model.config import BOARD_SIZE, ModelConfig
from model.encoding import encode_game
from model.masking import legal_moves_to_mask, masked_softmax
from renju import Game


def policy_probabilities(model, game: Game, device='cpu') -> dict[tuple[int, int], float]:
    """Masked policy over the legal moves; trunk + policy head only."""
    legal = game.legal_moves()
    if not legal:
        return {}
    mask = legal_moves_to_mask(legal)
    planes = encode_game(game, mask).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model.policy_head(model.trunk(planes))
        probs = masked_softmax(logits, mask.unsqueeze(0).to(device))[0].cpu()
    return {(r, c): float(probs[r * BOARD_SIZE + c]) for r, c in legal}


class RootPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str = 'cpu', threads: int | None = 1):
        checkpoint = Path(checkpoint)
        meta_path = checkpoint.with_suffix('.json')
        if not checkpoint.exists() or not meta_path.exists():
            raise FileNotFoundError(f'H3 checkpoint or its metadata is missing: {checkpoint}, {meta_path}')
        self.meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if self.meta.get('policy_trained') is not True or self.meta.get('value_trained') is not False:
            raise ValueError('expected an H3 policy-only checkpoint (policy_trained true, value_trained false)')
        if threads:
            torch.set_num_threads(threads)  # benchmark workers run in parallel processes
        cfg = self.meta['config']
        self.device = device
        self.model = load_checkpoint(checkpoint, ModelConfig(channels=cfg['channels'], blocks=cfg['blocks']),
                                     device=device)
        self.model.eval()
        self.checkpoint = str(checkpoint)
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, game: Game) -> dict[tuple[int, int], float]:
        started = perf_counter()
        result = policy_probabilities(self.model, game, self.device)
        self.calls += 1
        self.seconds += perf_counter() - started
        return result

    def describe(self) -> dict:
        return {'checkpoint': self.checkpoint, 'step': self.meta.get('step'),
                'h2_output_sha256': self.meta.get('h2_output_sha256'), 'device': self.device}
