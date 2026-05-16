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
                 Default is 1e-6, which is consistent with the original
                 vllm RMSNorm default and most HuggingFace model configs
                 (e.g. LLaMA uses 1e-6, Mistral uses 1e-5).

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
                 Default is 1e-6 to match npu_rms_norm.

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

    Drop-in replacemen