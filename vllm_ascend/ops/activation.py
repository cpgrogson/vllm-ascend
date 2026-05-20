# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from vllm-project/vllm for Ascend NPU support.

"""NPU-optimized activation functions for vllm-ascend.

Provides fused activation kernels leveraging torch_npu ops where available,
with fallback to standard PyTorch implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import torch_npu
    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


def _check_torch_npu() -> bool:
    """Check whether torch_npu is available for NPU-accelerated ops."""
    return _TORCH_NPU_AVAILABLE


def npu_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation with gating (SwiGLU-style).

    Splits the last dimension of `x` in half, applies SiLU to the first half,
    and multiplies element-wise with the second half.

    Args:
        x: Input tensor of shape (..., 2 * d). The last dimension is split
           into two equal parts.

    Returns:
        Output tensor of shape (..., d) after SiLU gating.
    """
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]

    if _check_torch_npu():
        # Use torch_npu fused op when available for better performance
        try:
            return torch_npu.npu_silu(gate) * up
        except (AttributeError, RuntimeError):
            pass

    return F.silu(gate) * up


def npu_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused GELU activation with gating (GeGLU-style).

    Splits the last dimension of `x` in half, applies GELU to the first half,
    and multiplies element-wise with the second half.

    Args:
        x: Input tensor of shape (..., 2 * d).

    Returns:
        Output tensor of shape (..., d) after GELU gating.
    """
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return F.gelu(gate) * up


def npu_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused GELU (tanh approximation) with gating.

    Args:
        x: Input tensor of shape (..., 2 * d).

    Returns:
        Output tensor of shape (..., d) after GELU (tanh approx) gating.
    """
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return F.gelu(gate, approximate="tanh") * up


class NPUSiluAndMul(nn.Module):
    """Module wrapper for NPU-optimized SiLU gating (SwiGLU).

    Used as a drop-in replacement for vllm's SiluAndMul activation
    in models running on Ascend NPUs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return npu_silu_and_mul(x)


class NPUGeluAndMul(nn.Module):
    """Module wrapper for NPU-optimized GELU gating (GeGLU).

    Args:
        approximate: GELU approximation method. Either ``"none"`` (exact)
            or ``"tanh"`` (tanh approximation). Defaults to ``"none"``.
    """

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"approximate must be 'none' or 'tanh', got '{approximate}'"
            )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "tanh":
            return npu_gelu_tanh_and_mul(x)
        return npu_gelu_and_mul(x)
