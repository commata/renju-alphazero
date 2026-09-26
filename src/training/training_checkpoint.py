"""Atomic Stage 6 training checkpoints (resume snapshots, not Stage 4 weight files).

One file is enough to resume: model, optimizer, replay buffer, generation/global_step,
global and component RNG states, and the full resolved config. ``generation`` is the
**next generation to run**; ``checkpoint_gen{N:03d}.pt`` always stores generation N.

Every value is a tensor or a plain int/float/str/bool/None/list/dict, so loading uses
``torch.load(weights_only=True)``.
"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import platform
import re
import shutil

import torch

from model.config import (ACTION_INDEX_VERSION, CHECKPOINT_FORMAT_VERSION, ENCODER_VERSION,
                          INPUT_PLANE_NAMES, ModelConfig)
from model.evaluator import file_sha256
from model.network import PolicyValueNet

from .config import (ConfigError, config_differences, critical_config,
                     critical_config_hash, model_config, validate_config)
from .provenance import git_provenance
from .replay_buffer import ReplayBuffer
from .trainer import build_optimizer
from .training_state import (COMPONENT_RNGS, TrainingState, global_rng_state,
                             python_rng_state, restore_global_rng_state, restore_python_rng)

TRAINING_CHECKPOINT_FORMAT = 'stage6-training-checkpoint-v1'
INIT_NAME = 'checkpoint_init.pt'
LATEST_NAME = 'latest.pt'
_GENERATION_FILE = re.compile(r'^checkpoint_gen(\d{3,})\.pt$')


class CheckpointCompatibilityError(ValueError):
    pass


def generation_checkpoint_name(generation: int) -> str:
    return f'checkpoint_gen{generation:03d}.pt'


def model_contract() -> dict:
    return {'encoder_version': ENCODER_VERSION, 'action_index_version': ACTION_INDEX_VERSION,
            'input_plane_names': list(INPUT_PLANE_NAMES),
            'model_checkpoint_format_version': CHECKPOINT_FORMAT_VERSION}


def build_checkpoint(state: TrainingState) -> dict:
    commit, dirty = git_provenance()
    return {
        'format_version': TRAINING_CHECKPOINT_FORMAT,
        'model_state_dict': state.model.state_dict(),
        'optimizer_state_dict': state.optimizer.state_dict(),
        'scheduler_state_dict': None,
        'generation': state.generation,
        'global_step': state.global_step,
        'replay_buffer': state.buffer.state_dict(),
        'global_rng': global_rng_state(),
        'component_rng': {name: python_rng_state(getattr(state, name))
                          for name in COMPONENT_RNGS},
        'config': state.config,
        'critical_config_hash': critical_config_hash(state.config),
        'git_commit': commit,
        'git_dirty': dirty,
        'device': str(state.config['device']),
        'model_config': asdict(state.model.config),
        'model_contract': model_contract(),
        'python_version': platform.python_version(),
        'torch_version': str(torch.__version__),
    }


def _fsync_dir(directory: Path) -> None:
    if os.name == 'nt':  # directories cannot be opened for fsync on Windows
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, write) -> None:
    """write(file) into a same-directory temp file, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        with open(temp, 'wb') as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # The handle is closed here: Windows cannot replace a file that is still open.
        os.replace(temp, path)
        _fsync_dir(path.parent)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def save_atomic(path: str | Path, payload: dict) -> str:
    """Atomically save ``payload``; return the SHA256 of the file written."""
    _atomic_write(Path(path), lambda handle: torch.save(payload, handle))
    return file_sha256(path)


def copy_atomic(source: str | Path, destination: str | Path) -> None:
    with open(source, 'rb') as src:
        _atomic_write(Path(destination), lambda handle: shutil.copyfileobj(src, handle))


def prune_checkpoints(directory: str | Path, keep: int) -> list[Path]:
    """Keep the newest ``keep`` generation files; never touch init/latest."""
    files = sorted((int(m.group(1)), p) for p in Path(directory).iterdir()
                   if (m := _GENERATION_FILE.match(p.name)))
    removed = [p for _, p in files[:-keep]] if keep > 0 else []
    for path in removed:
        path.unlink()
    return removed


def load_checkpoint_payload(path: str | Path, *, device: str | torch.device = 'cpu') -> dict:
    """Trusted and untrusted files alike load with weights_only=True (no pickle code)."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError('training checkpoint must be a dictionary')
    if payload.get('format_version') != TRAINING_CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError(
            f"unsupported training checkpoint format: {payload.get('format_version')!r} "
            f'(expected {TRAINING_CHECKPOINT_FORMAT})')
    if payload.get('model_contract') != model_contract():
        raise CheckpointCompatibilityError('encoder/action/model contract mismatch')
    return payload


def resolve_resume_config(checkpoint_config: dict, requested: dict | None) -> dict:
    """Checkpoint wins for training-critical values; differing --config values are errors.

    Execution-control values (generations, keep_checkpoints, output, device, threads)
    are taken from ``requested`` when given.
    """
    config = validate_config(dict(checkpoint_config))
    if requested is None:
        return config
    diffs = config_differences(critical_config(checkpoint_config), critical_config(requested))
    if diffs:
        raise ConfigError('training-critical config differs from checkpoint: '
                          + ', '.join(diffs))
    merged = dict(requested)
    return validate_config(merged)


def restore_training_state(payload: dict, config: dict,
                           source_hash: str | None = None) -> TrainingState:
    if payload['critical_config_hash'] != critical_config_hash(payload['config']):
        raise CheckpointCompatibilityError('checkpoint config hash does not match its config')
    if critical_config_hash(config) != payload['critical_config_hash']:
        raise ConfigError('training-critical config differs from checkpoint: '
                          + ', '.join(config_differences(critical_config(payload['config']),
                                                         critical_config(config))))
    expected_model = asdict(model_config(config))
    if payload['model_config'] != expected_model:
        raise CheckpointCompatibilityError(
            f"model architecture mismatch: checkpoint {payload['model_config']}, "
            f'config {expected_model}')
    device = torch.device(config['device'])
    model = PolicyValueNet(ModelConfig(**payload['model_config'])).to(device)
    try:
        model.load_state_dict(payload['model_state_dict'], strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError('incompatible model_state_dict') from exc
    model.train()
    optimizer = build_optimizer(model, config)
    try:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    except (ValueError, KeyError) as exc:
        raise CheckpointCompatibilityError('incompatible optimizer_state_dict') from exc
    if payload['scheduler_state_dict'] is not None:
        raise CheckpointCompatibilityError('MVP has no lr scheduler')
    buffer = ReplayBuffer(config['training']['replay_capacity'])
    buffer.load_state_dict(payload['replay_buffer'])
    restore_global_rng_state(payload['global_rng'])
    rngs = {name: restore_python_rng(payload['component_rng'][name]) for name in COMPONENT_RNGS}
    return TrainingState(config=config, model=model, optimizer=optimizer, scheduler=None,
                         buffer=buffer, generation=int(payload['generation']),
                         global_step=int(payload['global_step']),
                         source_checkpoint_hash=source_hash, **rngs)


def load_training_state(path: str | Path, requested_config: dict | None = None
                        ) -> TrainingState:
    payload = load_checkpoint_payload(path)
    config = resolve_resume_config(payload['config'], requested_config)
    return restore_training_state(payload, config, file_sha256(path))
