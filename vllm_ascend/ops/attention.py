# Copyright 2024 The vllm-ascend Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU attention operations.

This module provides NPU-optimized attention implementations that wrap
the underlying torch_npu kernels and integrate with vllm's attention
infrastructure.
"""

from typing import List, Optional, Tuple

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None  # type: ignore[assignment]


def _check_torch_npu() -> None:
    """Raise ImportError if torch_npu is not available."""
    if torch_npu is None:
        raise ImportError(
            "torch_npu is required for Ascend NPU attention. "
            "Please install the appropriate torch_npu package."
        )


def npu_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor] = None,
    max_context_len: Optional[int] = None,
) -> torch.Tensor:
    """Paged attention decode kernel for Ascend NPU.

    Args:
        query: Query tensor of shape [num_tokens, num_heads, head_size].
        key_cache: Paged key cache of shape
            [num_blocks, num_kv_heads, block_size, head_size].
        value_cache: Paged value cache of shape
            [num_blocks, num_kv_heads, block_size, head_size].
        block_tables: Block table tensor of shape [num_tokens, max_blocks_per_seq].
        context_lens: Context length per sequence of shape [num_tokens].
        scale: Softmax scale factor (typically 1 / sqrt(head_size)).
        alibi_slopes: Optional ALiBi slopes of shape [num_heads].
        max_context_len: Maximum context length across all sequences.
            If None, computed automatically from context_lens.max().

    Returns:
        Output tensor of shape [num_tokens, num_heads, head_size].

    Note:
        alibi_slopes is accepted for API compatibility but is not currently
        forwarded to the underlying torch_npu kernel. If ALiBi support is
        needed, the kernel call below will need to be updated accordingly.
    """
    _check_torch_npu()

    if max_context_len is None:
        # Compute max_context_len from context_lens to avoid requiring callers
        # to pass it explicitly; .item() ensures we get a plain Python int.
        max_context_len = int(context_lens.max().item())

    # Guard against an empty batch (e.g. during warmup) to avoid a potential
    # divide-by-zero or shape mismatch inside the NPU kernel.
    if query.shape[0] == 0:
        return torch.empty_like(query)

    # Warn if alibi_slopes are provided but will be silently ignored.
    # TODO: add proper ALiBi support once torch_npu exposes the parameter.
    if alibi_slopes is not None:
        import warnings
        # Only warn once per process to avoid flooding logs during repeated calls
        # (e.g. in multi-step decoding loops). Using stacklevel=2 so the warning
        # points to the caller rather than this module.
        warnings.warn(
            "alibi_slopes were provided to npu_paged_attention but are not "
            "currently supported by the torch_npu backend and will be ignored.",
            UserWarning,
            stacklevel=2,
        )
