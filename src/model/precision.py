"""float32 precision of CUDA matmul / convolution (TF32 on NVIDIA Ampere and newer).

PyTorch runs cuDNN convolutions in TF32 by default on these GPUs (10-bit mantissa,
relative error about 1e-3). That is far above the CPU-parity tolerance of the Stage 8
GPU gate, so the training loop and the gate use full IEEE float32 unless TF32 is
explicitly requested. ROCm and the CPU ignore these switches.
"""
from __future__ import annotations

import torch


def _new_api() -> bool:
    conv = getattr(torch.backends.cudnn, 'conv', None)
    return conv is not None and hasattr(conv, 'fp32_precision')


def set_tf32(enabled: bool) -> None:
    """Allow (True) or forbid (False) TF32 for CUDA matmul and cuDNN convolutions."""
    if _new_api():
        mode = 'tf32' if enabled else 'ieee'
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
    else:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def tf32_state() -> dict:
    if _new_api():
        return {'matmul': torch.backends.cuda.matmul.fp32_precision,
                'conv': torch.backends.cudnn.conv.fp32_precision}
    return {'matmul': 'tf32' if torch.backends.cuda.matmul.allow_tf32 else 'ieee',
            'conv': 'tf32' if torch.backends.cudnn.allow_tf32 else 'ieee'}
