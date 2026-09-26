"""Torch-free provenance for Stage 5 records.

``model.checkpoint`` has an equivalent git helper, but importing it loads torch, so
core self-play keeps its own copy. Checkpoint-embedded git fields are separate data.
"""
from pathlib import Path
import platform
import subprocess
import sys


def _git_text(args: list[str], cwd: Path) -> str | None:
    try:
        output = subprocess.check_output(
            args, cwd=cwd, stderr=subprocess.DEVNULL, text=True,
            encoding='utf-8', errors='strict', timeout=5,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return output.strip() if isinstance(output, str) else None


def git_provenance() -> tuple[str | None, bool | None]:
    """Return (HEAD, dirty) of this source worktree, or (None, None) when unavailable."""
    source = Path(__file__).resolve()
    root_text = _git_text(['git', 'rev-parse', '--show-toplevel'], source.parent)
    if not root_text:
        return None, None
    try:
        root = Path(root_text).resolve()
    except (OSError, ValueError):
        return None, None
    if source != (root / 'src' / 'training' / 'provenance.py').resolve():
        return None, None  # never report an unrelated parent repository
    commit = _git_text(['git', 'rev-parse', 'HEAD'], root)
    status = _git_text(['git', 'status', '--porcelain', '--untracked-files=no'], root)
    if not commit or status is None:
        return None, None
    return commit, bool(status)


def base_runtime_env() -> dict:
    """Torch-free runtime facts; neural callers add torch/device/thread details."""
    return {
        'python': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'byteorder': sys.byteorder,
    }
