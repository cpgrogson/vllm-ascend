# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# This file provides NPU-optimized layer normalization operations
# for vllm-ascend, leveraging torch_npu fused kernels where available.

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    import torch_npu
    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


def _check_torch_npu() -> bool:
    """Check whether torch_npu is available for NPU-accelerated ops."""
    return _TORCH_NPU_AVAILABLE


def npu_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """NPU-optimized RMS Layer Normalization.

    Uses torch_npu's fused rms_norm kernel when available,
    falling back to a pure PyTorch implementation otherwise.

    Args:
        x: Input tensor of shape [..., hidden_size].
        weight: Learnable scale parameter of shape [hidden_size].
        epsilon: Small value added to denominator for numerical stability.

    Returns:
        Normalized tensor with the same shape as input.
    """
    if _check_torch_npu():
        # torch_npu.npu_rms_norm returns (output, rstd)
        output, _ = torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)
        return output

    # Pure PyTorch fallback
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + epsilon)
    return x_normed * weight


def npu_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NPU-optimized fused Add + RMS Layer Normalization.

    Fuses the residual addition and RMS norm into a single kernel call
    to reduce memory bandwidth overhead. This is commonly used in
    transformer decoder layers.

    Args:
        x: Input tensor of shape [..., hidden_size].
        residual: Residual tensor of the same shape as x.
        weight: Learnable scale parameter of shape [hidden_size].
        epsilon: Small value added to denominator for numerical stability.

    Returns:
        A tuple of:
          - normed output tensor of shape [..., hidden_size]
          - updated residual (x + residual) of the same shape
    """
    # Accumulate residual first
    residual = x + residual

    if _check_torch_npu():
        output, _ = torch_npu.npu_rms_norm(residual, weight, epsilon=epsilon)
        return output, residual

    # Pure PyTorch fallback
    variance = residual.pow(2).mean(dim=-1, keepdim=True)
    normed = residual * torch.rsqrt(variance + epsilon)
    return normed * weight, residual


class NPURMSNorm(nn.Module):
    """RMS Normalization module backed by NPU-optimized kernels.

    Drop-in replacement for vllm's RMSNorm that routes computation
    through :func:`npu_rms_norm` and :func:`npu_fused_add_rms_norm`.

    Args:
        hidden_size: Dimensionality of the input features.
        eps: Epsilon for numerical stability (default: 1e-6).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional fused residual addition.

        Args:
            x: Input tensor.
            residual: Optional residual tensor. When provided, performs
                fused add + norm and returns both the normed output and
                the updated residual.

        Returns:
            Normed tensor, or a (normed, residual) tuple when residual
            is supplied.
        """
        if residual is not None:
            return npu_fused_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
        return npu_rms_norm(x, self.weight, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"
