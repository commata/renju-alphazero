"""Mutable training state, component RNGs and RNG (de)serialization.

Stateful component RNGs (saved in every checkpoint):
- ``self_play_rng``: derives one Stage 5 game seed per self-play game (that game's
  ``Random(seed)`` drives temperature sampling and Dirichlet noise)
- ``sample_rng``: replay buffer indices
- ``augment_rng``: D4 symmetry choice

Evaluation owns no state: each game derives a fresh RNG from
``(seed, generation, opponent, game_index)`` via ``derive_seed``.

RNG states are converted to plain ints/lists/tensors so checkpoints load with
``torch.load(weights_only=True)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from random import Random

import torch

from model.checkpoint import load_checkpoint
from model.network import PolicyValueNet

from .config import model_config
from .replay_buffer import ReplayBuffer
from .trainer import build_optimizer

try:  # NumPy is not a project dependency; save its global state only when present.
    import numpy
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    numpy = None

COMPONENT_RNGS = ('self_play_rng', 'sample_rng', 'augment_rng')


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from labelled parts (independent of PYTHONHASHSEED)."""
    text = '|'.join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big') >> 1


@dataclass
class TrainingState:
    config: dict
    model: PolicyValueNet
    optimizer: torch.optim.Optimizer
    scheduler: None                 # MVP: constant lr, no scheduler
    buffer: ReplayBuffer
    generation: int                 # next generation to run
    global_step: int
    self_play_rng: Random
    sample_rng: Random
    augment_rng: Random
    source_checkpoint_hash: str | None = None   # file whose contents equal this state
    run_dir: Path | None = None                 # set by the training loop


def python_rng_state(rng: Random) -> dict:
    version, internal, gauss_next = rng.getstate()
    return {'version': version, 'internal': list(internal), 'gauss_next': gauss_next}


def restore_python_rng(state: dict) -> Random:
    rng = Random()
    set_python_rng_state(rng, state)
    return rng


def set_python_rng_state(rng: Random, state: dict) -> None:
    rng.setstate((state['version'], tuple(state['internal']), state['gauss_next']))


def global_rng_state() -> dict:
    state = {'python': python_rng_state(random._inst), 'torch': torch.get_rng_state(),
             'numpy': None}
    if numpy is not None:
        name, keys, pos, has_gauss, cached = numpy.random.get_state()
        state['numpy'] = {'name': name, 'keys': [int(k) for k in keys], 'pos': int(pos),
                          'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)}
    return state


def restore_global_rng_state(state: dict) -> None:
    set_python_rng_state(random._inst, state['python'])
    torch.set_rng_state(state['torch'])
    if numpy is not None and state.get('numpy') is not None:
        n = state['numpy']
        numpy.random.set_state((n['name'], numpy.array(n['keys'], dtype=numpy.uint32),
                                n['pos'], n['has_gauss'], n['cached_gaussian']))


def seed_global_rngs(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if numpy is not None:
        numpy.random.seed(seed % 2**32)


def init_training_state(config: dict) -> TrainingState:
    """generation 0 state: from ``training.init_checkpoint`` or a seeded random init."""
    seed = config['seed']
    seed_global_rngs(seed)
    device = torch.device(config['device'])
    init_path = config['training']['init_checkpoint']
    if init_path is not None:
        model = load_checkpoint(init_path, model_config(config), device=device)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed(seed, 'model_init'))
            model = PolicyValueNet(model_config(config)).to(device)
    model.train()
    return TrainingState(
        config=config, model=model, optimizer=build_optimizer(model, config), scheduler=None,
        buffer=ReplayBuffer(config['training']['replay_capacity']), generation=0,
        global_step=0,
        self_play_rng=Random(derive_seed(seed, 'self_play')),
        sample_rng=Random(derive_seed(seed, 'sample')),
        augment_rng=Random(derive_seed(seed, 'augment')),
    )
