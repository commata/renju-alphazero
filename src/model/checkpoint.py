"""Strict, versioned weights-only checkpoints (not training-resume snapshots)."""
from dataclasses import asdict
from pathlib import Path
import subprocess

import torch

from .config import (ACTION_INDEX_VERSION, CHECKPOINT_FORMAT_VERSION, ENCODER_VERSION,
                     INPUT_PLANE_NAMES, ModelConfig)
from .network import PolicyValueNet


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save_checkpoint(path: str | Path, model: PolicyValueNet) -> None:
    torch.save({
        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
        'model_state': model.state_dict(),
        'model_config': asdict(model.config),
        'encoder_version': ENCODER_VERSION,
        'action_index_version': ACTION_INDEX_VERSION,
        'input_plane_names': list(INPUT_PLANE_NAMES),
        'torch_version': str(torch.__version__),
        'git_commit': _git_commit(),
    }, path)


def load_checkpoint(path: str | Path, expected_config: ModelConfig = ModelConfig(),
                    *, device: str | torch.device = 'cpu') -> PolicyValueNet:
    """Require the caller's architecture contract; return a strict-loaded eval model."""
    data = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(data, dict):
        raise ValueError('checkpoint must be a dictionary')
    expected = {
        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
        'model_config': asdict(expected_config),
        'encoder_version': ENCODER_VERSION,
        'action_index_version': ACTION_INDEX_VERSION,
        'input_plane_names': list(INPUT_PLANE_NAMES),
    }
    for key, value in expected.items():
        if type(data.get(key)) is not type(value) or data[key] != value:
            raise ValueError(f'incompatible checkpoint metadata: {key}')
    try:
        parsed = ModelConfig(**data['model_config'])
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid model_config') from exc
    if not isinstance(data.get('torch_version'), str) or 'git_commit' not in data:
        raise ValueError('missing torch/git metadata')
    if data['git_commit'] is not None and not isinstance(data['git_commit'], str):
        raise ValueError('invalid git_commit')
    model = PolicyValueNet(parsed).to(device)
    try:
        model.load_state_dict(data['model_state'], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError('incompatible model_state') from exc
    return model.eval()
