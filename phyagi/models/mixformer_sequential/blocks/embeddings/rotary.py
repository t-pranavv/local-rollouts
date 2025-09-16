# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from phyagi.models.mixformer_sequential.blocks.embeddings.rotary_utils import (
    apply_rotary_emb,
    apply_rotary_emb_kv,
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)
from phyagi.utils.import_utils import is_flash_attn_available


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding.
        https://arxiv.org/pdf/2104.09864.pdf.

    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        flash_rotary: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if scale_base is not None:
            raise NotImplementedError

        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.flash_rotary = flash_rotary
        self.device = device

        # Generate and save the inverse frequency buffer (non-trainable)
        inv_freq = self._compute_inv_freq(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Generate and save the scale buffer (non-trainable)
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

        self._register_rotary_funcs()

    def _register_rotary_funcs(self) -> None:
        self.rotary_func = apply_rotary_emb
        self.rotary_func_kv = apply_rotary_emb_kv

        if self.flash_rotary and is_flash_attn_available():
            from flash_attn.layers.rotary import (
                apply_rotary_emb_func,
                apply_rotary_emb_kv_,
            )

            self.rotary_func = apply_rotary_emb_func
            self.rotary_func_kv = apply_rotary_emb_kv_

    def _compute_inv_freq(self, device: Optional[str] = None) -> torch.FloatTensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(
        self, seqlen: int, device: Optional[str] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        # Tables should be reset if the sequence length has changed, if we are on a new device
        # or if we are switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            # fp32 is preferred since the output of `torch.arange` can be quite large
            # and bf16 would lose a lot of precision
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != device:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            # `torch.outer` is preferred since `torch.einsum` converts from fp32 to fp16 if used with AMP
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

                # Force the scale multiplication to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        seqlen_offset: int = 0,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        seqlen = q.shape[1]

        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        else:
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=q.device, dtype=q.dtype)

        q = self.rotary_func(q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
        kv = self.rotary_func_kv(kv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])

        return q, kv


class YarnEmbedding(nn.Module):
    """YaRN extended rotary positional embedding.

    Reference:
        YaRN: Efficient Context Window Extension of Large Language Models
        https://arxiv.org/abs/2309.00071

    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        max_position_embeddings: int = 2048,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        dynamic: bool = False,
        finetuned: bool = False,
        flash_rotary: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.factor = factor
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = (
            original_max_position_embeddings if original_max_position_embeddings else max_position_embeddings
        )
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.mscale = float(yarn_get_mscale(self.factor) * attn_factor)
        self.dynamic = dynamic
        self.finetuned = finetuned
        self.flash_rotary = flash_rotary

        # Generate and save the inverse frequency buffer (non-trainable)
        if not dynamic:
            self._compute_inv_freq(factor, device)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

        self._register_rotary_funcs()

    def _register_rotary_funcs(self) -> None:
        self.rotary_func = apply_rotary_emb
        self.rotary_func_kv = apply_rotary_emb_kv

        if self.flash_rotary and is_flash_attn_available():
            from flash_attn.layers.rotary import (
                apply_rotary_emb_func,
                apply_rotary_emb_kv_,
            )

            self.rotary_func = apply_rotary_emb_func
            self.rotary_func_kv = apply_rotary_emb_kv_

    @torch.no_grad()
    def _compute_inv_freq(self, factor: float, device: Optional[str] = None) -> None:
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings
        )
        # Get N-dimensional rotation scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, self.dim // 2, device=device)) * self.extrapolation_factor

        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _compute_inv_freq_original(self, device: Optional[str] = None) -> None:
        inv_freq = 1 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _update_cos_sin_cache(
        self, seqlen: int, device: Optional[str] = None, dtype: Optional[torch.dtype] = None
    ) -> None:
        # Tables should be reset if the sequence length has changed, if we are on a new device
        # or if we are switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            if self.dynamic:
                factor = None

                if seqlen <= self.max_position_embeddings:
                    if self.finetuned:
                        factor = self.factor
                else:
                    factor = seqlen / self.original_max_position_embeddings

                if factor:
                    self._compute_inv_freq(factor, device=device)
                    self.mscale = float(yarn_get_mscale(factor) * self.attn_factor)
                else:
                    self._compute_inv_freq_original(device=device)

            # fp32 is preferred since the output of `torch.arange` can be quite large
            # and bf16 would lose a lot of precision
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                if self.inv_freq.dtype != torch.float32 or self.inv_freq.device != device:
                    inv_freq = inv_freq = self.inv_freq.to(torch.float32).to(device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            # `torch.outer` is preferred since `torch.einsum` converts from fp32 to fp16 if used with AMP
            freqs = torch.outer(t, inv_freq)

            # Force the scale multiplication to happen in fp32
            self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        seqlen_offset: int = 0,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        seqlen = q.shape[1]

        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        else:
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=q.device, dtype=q.dtype)

        q = self.rotary_func(q, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
        kv = self.rotary_func_kv(kv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])

        return q, kv
