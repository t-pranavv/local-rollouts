# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Optional, Tuple

import torch
from einops import rearrange


def apply_rotary_emb(
    x: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
) -> torch.FloatTensor:
    """Apply rotary embeddings to a tensor.

    Args:
        x: Input tensor.
        cos: Cosine positional embeddings.
        sin: Sine positional embeddings.

    Returns:
        The input tensor with rotary embeddings applied.

    """

    _, seqlen, _, head_dim = x.shape
    rotary_seqlen, rotary_dim = cos.shape
    rotary_dim *= 2

    if rotary_dim > head_dim:
        raise ValueError(f"`rotary_dim` must be <= {head_dim}, but got {rotary_dim}.")
    if seqlen > rotary_seqlen:
        raise ValueError(f"`seqlen` must be <= {rotary_seqlen}, but got {seqlen}.")
    if cos.shape != sin.shape:
        raise ValueError(f"`cos` and `sin` must have the same shape, but got {cos.shape} and {sin.shape}.")
    if cos.shape != (rotary_seqlen, rotary_dim // 2):
        raise ValueError(f"`cos` and `sin` must have shape {(rotary_seqlen, rotary_dim // 2)}, but got {cos.shape}.")

    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    x1, x2 = x_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    x1, x2, c, s = [t.to(dtype=torch.float32) for t in [x1, x2, c, s]]

    x_rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).to(x.dtype)

    return torch.cat([x_rot, x_pass], axis=-1)


def apply_rotary_emb_kv(
    kv: torch.FloatTensor,
    cos: torch.FloatTensor,
    sin: torch.FloatTensor,
    cos_k: Optional[torch.FloatTensor] = None,
    sin_k: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Apply rotary embeddings to a key-value tensor.

    Args:
        kv: Key-value tensor.
        cos: Cosine positional embeddings.
        sin: Sine positional embeddings.
        cos_k: Scaled cosine positional embeddings.
        sin_k: Scaled sine positional embeddings.

    Returns:
        The key-value tensor with rotary embeddings applied.

    """

    _, seqlen, two, _, head_dim = kv.shape

    if two != 2:
        raise ValueError(f"`kv` must have 2 heads, but got {two}.")

    rotary_seqlen, rotary_dim = cos.shape
    rotary_dim *= 2

    if rotary_dim > head_dim:
        raise ValueError(f"`rotary_dim` must be <= {head_dim}, but got {rotary_dim}.")
    if seqlen > rotary_seqlen:
        raise ValueError(f"`seqlen` must be <= {rotary_seqlen}, but got {seqlen}.")
    if cos.shape != sin.shape:
        raise ValueError(f"`cos` and `sin` must have the same shape, but got {cos.shape} and {sin.shape}.")
    if cos.shape != (rotary_seqlen, rotary_dim // 2):
        raise ValueError(f"`cos` and `sin` must have shape {(rotary_seqlen, rotary_dim // 2)}, but got {cos.shape}.")

    k_rot = kv[:, :, 0, :, :rotary_dim]
    k_pass = kv[:, :, 0, :, rotary_dim:]

    k1, k2 = k_rot.chunk(2, dim=-1)
    c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")
    k1, k2, c, s = [t.to(dtype=torch.float32) for t in [k1, k2, c, s]]

    k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(kv.dtype)

    return torch.cat(
        [
            torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
            kv[:, :, 1:2, :, :],
        ],
        axis=2,
    )


def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    # Inverse dim formula to find dim based on number of rotations
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
) -> Tuple[float, float]:
    """Find the YaRN correction range based on the low and high rotations.

    Args:
        low_rot: Low rotation.
        high_rot: High rotation.
        dim: Dimension of the embedding
        base: RoPE base frequency.
        max_position_embeddings: Maximum sequence length.

    Returns:
        Low and high correction dimensions.

    """

    # Find dim range bounds based on rotations
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))

    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(min: float, max: float, dim: int, device: Optional[str] = None) -> torch.Tensor:
    """Create a linear ramp mask for YaRN.

    Args:
        min: Minimum value.
        max: Maximum value.
        dim: Dimension of the embedding

    Returns:
        Linear ramp mask.

    """

    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, device=device, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)

    return ramp_func


def yarn_get_mscale(scale: float = 1.0) -> float:
    """Get the YaRN key/query scalar coefficient ``1/sqrt(t)``.

    Args:
        scale: Scale. Ratio between the new sequence length (``L``) and the original sequence length ``L``.

    Returns:
        YaRN key/query scalar coefficient.

    """

    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
