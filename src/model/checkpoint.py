"""Strict, versioned weights-only checkpoints (not training-resume snapshots)."""
from dataclasses import asdict
from pathlib import Path
import subprocess

import torch

from .config import (ACTION_INDEX_VERSION, CHECKPOINT_FORMAT_VERSION, ENCODER_VERSION,
                     INPUT_PLANE_NAMES, ModelConfig)
from .network import PolicyValueNet


def _git_text(args: list[str], cwd: Path) -> str | None:
    """Read Git text as UTF-8, including non-ASCII Windows worktree paths."""
    try:
        output = subprocess.check_output(
            args, cwd=cwd, stderr=subprocess.DEVNULL, text=True,
            encoding='utf-8', errors='strict', timeout=5,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    if not isinstance(output, str):
        return None
    return output.strip()


def _git_provenance() -> tuple[str | None, bool | None]:
    """Return source-worktree HEAD/dirty state, never an unrelated parent repo."""
    source = Path(__file__).resolve()
    root_text = _git_text(['git', 'rev-parse', '--show-toplevel'], source.parent)
    if not root_text:
        return None, None
    try:
        root = Path(root_text).resolve()
    except (OSError, ValueError):
        return None, None
    if source != (root / 'src' / 'model' / 'checkpoint.py').resolve():
        return None, None

    commit = _git_text(['git', 'rev-parse', 'HEAD'], root)
    status = _git_text(['git', 'status', '--porcelain', '--untracked-files=no'], root)
    if not commit or status is None:
        return None, None
    return commit, bool(status)


def save_checkpoint(path: str | Path, model: PolicyValueNet) -> None:
    git_commit, git_dirty = _git_provenance()
    torch.save({
        'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
        'model_state': model.state_dict(),
        'model_config': asdict(model.config),
        'encoder_version': ENCODER_VERSION,
        'action_index_version': ACTION_INDEX_VERSION,
        'input_plane_names': list(INPUT_PLANE_NAMES),
        'torch_version': str(torch.__version__),
        'git_commit': git_commit,
        'git_dirty': git_dirty,
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
    if (not isinstance(data.get('torch_version'), str)
            or 'git_commit' not in data or 'git_dirty' not in data):
        raise ValueError('missing torch/git metadata')
    if data['git_commit'] is not None and not isinstance(data['git_commit'], str):
        raise ValueError('invalid git_commit')
    if data['git_dirty'] is not None and type(data['git_dirty']) is not bool:
        raise ValueError('invalid git_dirty')
    model = PolicyValueNet(parsed).to(device)
    try:
        model.load_state_dict(data['model_state'], strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError('incompatible model_state') from exc
    return model.eval()
