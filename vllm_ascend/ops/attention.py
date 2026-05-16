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

    Returns:
        Output tensor of shape [num_tokens, num_heads, head_size].
    """
    _check_torch_npu()

    if max_context_len is None:
        max_context_len = int(context_lens.max().item())

    num_tokens, num_heads, head_size = query.shape
    output = torch.empty_like(query)

    # torch_npu paged attention expects contiguous tensors
    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()

    torch_npu.npu_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        output,
    )
    return output


def npu_flash_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flash attention prefill kernel for Ascend NPU.

    Uses torch_npu's fused flash attention operator for the prefill phase,
    which processes full sequences without paging.

    Args:
        query: Query tensor of shape [batch, num_heads, seq_len, head_size].
        key: Key tensor of shape [batch, num_kv_heads, seq_len, head_size].
        value: Value tensor of shape [batch, num_kv_heads, seq_len, head_size].
        scale: Softmax scale factor.
        attn_mask: Optional attention mask (e.g. causal mask).
        alibi_slopes: Optional ALiBi slopes of shape [num_heads].

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_size].
    """
    _check_torch_npu()

    # npu_fusion_attention expects float16 or bfloat16
    if query.dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)

    output, _, _, _ = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query.shape[1],  # num_heads
        input_layout="BNSD",
        pse=None,
        atten_mask=attn_mask,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=0,
        inner_precise=0,
    )
    return output
