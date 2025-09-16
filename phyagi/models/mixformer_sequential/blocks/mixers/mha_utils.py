# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any, List, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributed import ProcessGroup

from phyagi.utils.import_utils import (
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_hopper_gpu_available,
)


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, scatter_idx: int, gather_idx: int, group: Any) -> torch.Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.group = group

        world_size = torch.distributed.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        torch.distributed.all_to_all(output_list, input_list, group=group)

        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        return (_SeqAllToAll.apply(*grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.group), None, None, None)


class DistributedAttention(nn.Module):
    """Distributed attention layer."""

    def __init__(
        self,
        local_attn: nn.Module,
        group: Optional[ProcessGroup] = None,
        scatter_idx: int = -2,
        gather_idx: int = 1,
    ) -> None:
        super().__init__()

        self.local_attn = local_attn
        self.group = group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def _offset_distributed_tensor(self, local_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        device = local_tensor.device
        dtype = local_tensor.dtype
        world_size = torch.distributed.get_world_size(self.group)

        # Since `local_tensor` can have different lengths across processes,
        # we need to gather the lengths first to determine the maximum length
        local_len = torch.tensor([local_tensor.shape[0]], device=device)
        lens = [torch.zeros_like(local_len) for _ in range(world_size)]
        torch.distributed.all_gather(lens, local_len, group=self.group)

        lens = [l.item() for l in lens]
        max_len = max(lens)

        # Pad maximum length with repeated last value (safe padding)
        pad_len = max_len - local_tensor.shape[0]
        if pad_len > 0:
            pad_value = local_tensor[-1]
            padding = pad_value.expand(pad_len)
            local_tensor_padded = torch.cat([local_tensor, padding], dim=0)
        else:
            local_tensor_padded = local_tensor

        # Gather padded tensors across all processes
        gathered = [torch.empty(max_len, dtype=dtype, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, local_tensor_padded, group=self.group)

        # Unpad and offset the gathered tensors
        global_values = []
        offset = 0
        for i in range(world_size):
            t = gathered[i][: lens[i]].clone()
            if i > 0:
                t = t[1:]
                t += offset
            global_values.append(t)
            offset = t[-1].item()

        return torch.cat(global_values, dim=0).contiguous()

    def forward(self, q: torch.Tensor, kv: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        q = _SeqAllToAll.apply(q, self.scatter_idx, self.gather_idx, self.group)
        kv = _SeqAllToAll.apply(kv, self.scatter_idx, self.gather_idx, self.group)

        # When using variable sequence lengths, we need to adjust the cumulative sequence lengths
        # and the maximum sequence length since `gather_idx` is the sequence dimension
        world_size = torch.distributed.get_world_size(self.group)
        for key in ["max_seqlen", "max_seqlen_k"]:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = kwargs[key] * world_size
        for key in ["cu_seqlens", "cu_seqlens_k"]:
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = self._offset_distributed_tensor(kwargs[key], world_size)

        attn_output = self.local_attn(q, kv, *args, **kwargs)

        return _SeqAllToAll.apply(attn_output, self.gather_idx, self.scatter_idx, self.group)


class Attention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        causal: bool = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]

        if kv.shape[0] != batch_size:
            raise ValueError(f"`kv` must have the same batch size as `q`, but got {kv.shape[0]} and {batch_size}.")
        if kv.shape[4] != q.shape[3]:
            raise ValueError(f"`kv` must have the same head dimension as `q`, but got {kv.shape[4]} and {q.shape[3]}.")

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        # Autocast is manually disabled to avoid `torch.einsum` performing the operation
        # using float16, which might lead to overflow
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen_k), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            rows = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
            cols = torch.arange(seqlen_k, device=k.device, dtype=torch.long)
            causal_mask = cols > rows + seqlen_k - seqlen_q

            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        attention = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


class FlashAttention(nn.Module):
    """Flash-Attention layer."""

    def __init__(
        self,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        alibi_slopes: Optional[List[float]] = None,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        if not is_flash_attn_available() and not (is_flash_attn_3_available() and is_hopper_gpu_available()):
            raise ImportError("`FlashAttention` is not available. Please install the `flash-attn` package.")

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

        self._register_flash_attn_funcs()

    def _register_flash_attn_funcs(self) -> None:
        self.flash_attn_version = 3 if is_flash_attn_3_available() and is_hopper_gpu_available() else 2

        if self.flash_attn_version == 3:
            import flash_attn_interface

            self.flash_attn_func = flash_attn_interface.flash_attn_func
            self.flash_attn_varlen_func = flash_attn_interface.flash_attn_varlen_func

        elif self.flash_attn_version == 2:
            import flash_attn

            self.flash_attn_func = flash_attn.flash_attn_func
            self.flash_attn_varlen_func = flash_attn.flash_attn_varlen_func

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        causal: Optional[bool] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        max_seqlen: Optional[int] = None,
        cu_seqlens_k: Optional[torch.IntTensor] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.FloatTensor:
        if q.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f"`q` must be either 'float16' or 'bfloat16', but got '{q.dtype}'.")
        if not q.is_cuda or not kv.is_cuda:
            raise ValueError("Both `q` and `kv` must be CUDA tensors.")

        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None

        # `alibi_slopes` needs to be forced into FP32 due to mixed precision
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)

        flash_attn_kwargs = {
            "causal": causal,
            "softmax_scale": self.softmax_scale,
            "window_size": self.window_size,
            "deterministic": self.deterministic,
        }
        if self.flash_attn_version == 2:
            flash_attn_kwargs.update(
                {"alibi_slopes": self.alibi_slopes, "dropout_p": self.drop.p if self.training else 0.0}
            )

        if unpadded:
            if cu_seqlens is None:
                raise ValueError("`cu_seqlens` must be provided when using variable sequence lengths.")
            if cu_seqlens.dtype != torch.int32:
                raise ValueError(f"`cu_seqlens` must be of type 'int32', but got '{cu_seqlens.dtype}'.")
            if max_seqlen is None or not isinstance(max_seqlen, int):
                raise ValueError(f"`max_seqlen` must be an integer, but got '{max_seqlen}'.")
            if cu_seqlens_k is None:
                raise ValueError("`cu_seqlens_k` must be provided when using variable sequence lengths.")
            if cu_seqlens_k.dtype != torch.int32:
                raise ValueError(f"`cu_seqlens_k` must be of type 'int32', but got '{cu_seqlens_k.dtype}'.")
            if max_seqlen_k is None or not isinstance(max_seqlen_k, int):
                raise ValueError(f"`max_seqlen_k` must be an integer, but got '{max_seqlen_k}'.")

            k, v = kv.unbind(dim=1)

            output = self.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                **flash_attn_kwargs,
            )

            return output[0] if self.flash_attn_version == 3 else output

        batch_size, _ = q.shape[0], q.shape[1]
        if kv.shape[0] != batch_size:
            raise ValueError(f"`kv` must have the same batch size as `q`, but got {kv.shape[0]}.")
        if kv.shape[4] != q.shape[3]:
            raise ValueError(f"`kv` must have the same head dimension as `q`, but got {kv.shape[4]} and {q.shape[3]}.")

        k, v = kv.unbind(dim=2)

        output = self.flash_attn_func(q, k, v, **flash_attn_kwargs)

        return output[0] if self.flash_attn_version == 3 else output
