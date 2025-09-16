# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
from transformers import Cache, PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.embeddings.rotary import (
    RotaryEmbedding,
    YarnEmbedding,
)
from phyagi.models.mixformer_sequential.blocks.mixers.mha_utils import (
    Attention,
    DistributedAttention,
)
from phyagi.models.mixformer_sequential.blocks.norms import get_norm
from phyagi.models.parallel_utils import ShardColwiseParallel
from phyagi.utils.import_utils import (
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_fused_dense_lib_available,
    is_hopper_gpu_available,
)
from phyagi.utils.logging_utils import get_logger

pad_input, unpad_input = None, None
FusedDense = None
if is_flash_attn_available():
    from flash_attn.bert_padding import pad_input, unpad_input

    if is_fused_dense_lib_available():
        from flash_attn.ops.fused_dense import FusedDense

FlashAttention = None
if is_flash_attn_available() or (is_flash_attn_3_available() and is_hopper_gpu_available()):
    from phyagi.models.mixformer_sequential.blocks.mixers.mha_utils import (
        FlashAttention,
    )

logger = get_logger(__name__)


def _get_alibi_slopes(n_head: int) -> List[float]:
    def _get_slopes_power_of_2(n_head: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n_head) - 3)))
        ratio = start

        return [start * ratio**i for i in range(n_head)]

    if math.log2(n_head).is_integer():
        return _get_slopes_power_of_2(n_head)

    closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
    return (
        _get_slopes_power_of_2(closest_power_of_2)
        + _get_alibi_slopes(2 * closest_power_of_2)[0::2][: n_head - closest_power_of_2]
    )


def _update_kv_cache(kv: torch.FloatTensor, past_key_values: Cache, layer_idx: int) -> torch.FloatTensor:
    # `past_key_values` expects shape (batch_size, n_head, seqlen, head_dim) per key/value
    kv = kv.transpose(1, 3)

    k, v = kv.unbind(dim=2)
    k, v = past_key_values.update(k, v, layer_idx)

    # `MHA` expects shape (batch_size, seqlen, 2, n_head, head_dim) per key/value
    kv = torch.stack([k, v], dim=2).transpose(1, 3)

    return kv


class MHA(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
        n_head: Optional[int] = None,
        n_head_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
        flash_attn: bool = True,
        causal: bool = True,
        use_alibi: bool = False,
        softmax_scale: Optional[float] = None,
        window_size: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0,
        fused_dense: bool = True,
        bias: bool = True,
        out_bias: Optional[bool] = None,
        qk_norm: bool = False,
        flash_rotary: bool = True,
        rotary_dim: Optional[int] = None,
        rotary_base: float = 10000.0,
        rotary_scale_base: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Dimensions
        self.n_embd = config.n_embd
        self.n_head = n_head or config.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"`n_embd` must be divisible by `n_head`, but got {self.n_embd} and {self.n_head}.")

        self.n_head_kv = n_head_kv or config.n_head_kv or self.n_head
        if self.n_head % self.n_head_kv != 0:
            raise ValueError(f"`n_head` must be divisible by `n_head_kv`, but got {self.n_head} and {self.n_head_kv}.")

        self.head_dim = head_dim or self.n_embd // self.n_head
        self.op_size = self.head_dim * (self.n_head + 2 * self.n_head_kv)
        self.n_positions = config.n_positions
        self.rotary_dim = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", 0)

        self.cp_size = getattr(config, "cp_size", 1)
        self.tp_size = getattr(config, "tp_size", 1)
        self.tp_mesh = None

        # `n_head` and `n_head_kv` need to be divisible by context and tensor parallelism size
        if self.n_head % (self.cp_size * self.tp_size) != 0:
            raise ValueError(
                f"`n_head` must be divisible by `cp_size * tp_size`, but got {self.n_head} and {self.cp_size * self.tp_size}."
            )
        if self.n_head_kv % (self.cp_size * self.tp_size) != 0:
            raise ValueError(
                f"`n_head_kv` must be divisible by `cp_size * tp_size`, but got {self.n_head_kv} and {self.cp_size * self.tp_size}."
            )

        # Remaining attributes
        self.fused_dense = fused_dense
        self.qk_norm = qk_norm
        self.layer_idx = layer_idx
        self.is_flash_attn = flash_attn and FlashAttention is not None

        # Rotary embedding
        self.rotary_emb = self._init_rope(
            rotary_base,
            rotary_scale_base,
            rope_scaling,
            flash_rotary,
            device,
        )

        # Projections
        self.Wqkv, self.out_proj = self._init_projs(
            fused_dense,
            bias,
            out_bias,
            device,
        )

        # Attention
        self.inner_attn = self._init_attn(
            causal,
            softmax_scale,
            dropout,
            window_size,
            use_alibi,
            device,
        )

        # Normalization (optional)
        if self.qk_norm:
            self.q_norm, self.k_norm = self._init_norms(config.architecture.get("norm", {}), config.layer_norm_epsilon)

        self.register_state_dict_post_hook(self._unshard_state_dict_hook)

    def _init_rope(
        self,
        rotary_base: float,
        rotary_scale_base: Optional[float],
        rope_scaling: Optional[Dict[str, Any]],
        flash_rotary: bool,
        device: Optional[str],
    ) -> Optional[Union[RotaryEmbedding, YarnEmbedding]]:
        if self.rotary_dim < 0:
            return None

        rope_scaling_type = (rope_scaling or {}).pop("rope_type", None)
        if rope_scaling_type == "yarn":
            return YarnEmbedding(
                self.rotary_dim,
                base=rotary_base,
                factor=rotary_scale_base or 1.0,
                max_position_embeddings=self.n_positions,
                original_max_position_embeddings=self.n_positions,
                flash_rotary=flash_rotary,
                **(rope_scaling or {}),
            )

        return RotaryEmbedding(
            self.rotary_dim,
            base=rotary_base,
            scale_base=rotary_scale_base,
            flash_rotary=flash_rotary,
            device=device,
        )

    def _init_projs(
        self,
        fused_dense: bool,
        bias: bool,
        out_bias: Optional[bool],
        device: Optional[str],
    ) -> Tuple[nn.Module, nn.Module]:
        linear_cls = FusedDense if fused_dense and FusedDense is not None else nn.Linear
        Wqkv = linear_cls(self.n_embd, self.op_size, bias=bias, device=device)
        out_proj = linear_cls(
            self.head_dim * self.n_head,
            self.n_embd,
            bias=out_bias if out_bias is not None else bias,
            device=device,
        )

        return Wqkv, out_proj

    def _init_attn(
        self,
        causal: bool,
        softmax_scale: Optional[float],
        dropout: float,
        window_size: Optional[Tuple[int, int]],
        use_alibi: bool,
        device: Optional[str],
    ) -> nn.Module:
        attn_cls = FlashAttention if self.is_flash_attn else Attention
        attn_kwargs = {
            "causal": causal,
            "softmax_scale": softmax_scale,
            "attention_dropout": dropout,
        }

        if self.is_flash_attn:
            if window_size is not None:
                if isinstance(window_size, (int, tuple)):
                    window_size = (window_size, 0) if isinstance(window_size, int) else window_size
                elif isinstance(window_size, str):
                    window_size = eval(window_size)
                attn_kwargs["window_size"] = window_size
            if use_alibi:
                attn_kwargs["alibi_slopes"] = torch.tensor(_get_alibi_slopes(self.n_head), device=device)

        if self.cp_size > 1:
            # ZeRO-2/3 requires module to be defined at initialization, but since we don't have
            # the process group yet, it will be set later
            return DistributedAttention(attn_cls(**attn_kwargs))

        return attn_cls(**attn_kwargs)

    def _init_norms(self, norm_config: Dict[str, Any], eps: float) -> Tuple[nn.Module, nn.Module]:
        q_norm = get_norm(self.head_dim, norm_config=norm_config, eps=eps)
        k_norm = get_norm(self.head_dim, norm_config=norm_config, eps=eps)

        return q_norm, k_norm

    @staticmethod
    def _unshard_state_dict_hook(module, state_dict, prefix, local_metadata) -> None:
        key = prefix + "Wqkv.weight"
        if key not in state_dict:
            return

        # Skip if the module does not have tensor parallelism enabled
        if module.tp_mesh is None:
            return

        shard_sizes = [
            module.tp_size * module.n_head * module.head_dim,
            module.tp_size * module.n_head_kv * module.head_dim,
            module.tp_size * module.n_head_kv * module.head_dim,
        ]

        state_dict[key] = ShardColwiseParallel.unshard_param(state_dict[key], shard_sizes, module.tp_mesh)

    def set_torch_tp(self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, **kwargs) -> None:
        if tp_mesh.size() > 1:
            if self.fused_dense:
                raise ValueError("`fused_dense` is not supported with `set_torch_tp()`.")

            plan = {
                "Wqkv": ShardColwiseParallel(
                    shard_sizes=[
                        self.n_head * self.head_dim,
                        self.n_head_kv * self.head_dim,
                        self.n_head_kv * self.head_dim,
                    ]
                ),
                "out_proj": RowwiseParallel(output_layouts=Shard(1) if enable_sequence_parallel else Replicate()),
            }
            parallelize_module(self, tp_mesh, plan)

            self.tp_mesh = tp_mesh
            self.n_head = self.n_head // self.tp_size
            self.n_head_kv = self.n_head_kv // self.tp_size

    def set_torch_cp(self, cp_mesh: Union[DeviceMesh, ProcessGroup], varlen: bool = False, **kwargs) -> None:
        cp_group = cp_mesh.get_group() if isinstance(cp_mesh, DeviceMesh) else cp_mesh
        if cp_group is not None:
            self.inner_attn.gather_idx = 0 if varlen else 1
            self.inner_attn.group = cp_group

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if attention_mask is not None and torch.any(~attention_mask.bool()):
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        qkv = self.Wqkv(hidden_states)

        q = qkv[..., : self.n_head * self.head_dim]
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        kv = qkv[..., self.n_head * self.head_dim :]
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            kv[:, :, 0] = self.k_norm(kv[:, :, 0])

        seqlen_offset = 0
        if position_ids is not None:
            seqlen_offset = position_ids.min().item()
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)

        causal = None if seqlen_offset == 0 else False

        if self.rotary_dim > 0:
            q, kv = self.rotary_emb(q, kv, seqlen_offset=seqlen_offset)

        if past_key_values is not None:
            kv = _update_kv_cache(kv, past_key_values, self.layer_idx)

        if self.is_flash_attn:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]

            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = None, None, None, None
            mask_type = "attention" if attention_mask is not None else "cu_seqlens" if cu_seqlens is not None else None

            if mask_type == "attention":
                kv, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(kv, attention_mask)

                if seqlen_q == 1:
                    attention_mask = torch.ones(batch_size, 1, device=q.device)
                elif seqlen_q != seqlen_k:
                    attention_mask = attention_mask[:, -seqlen_q:]

                q, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, attention_mask)

            elif mask_type == "cu_seqlens":
                cu_seqlens_q, cu_seqlens_k = cu_seqlens, cu_seqlens
                max_seqlen_q, max_seqlen_k = self.n_positions, self.n_positions

                q = rearrange(q, "b s ... -> (b s) ...")
                kv = rearrange(kv, "b s ... -> (b s) ...")

            attn_output = self.inner_attn(
                q,
                kv,
                causal=causal,
                cu_seqlens=cu_seqlens_q,
                max_seqlen=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
            )
            attn_output = (
                pad_input(attn_output, indices_q, batch_size, seqlen_q)
                if mask_type == "attention"
                else rearrange(attn_output, "(b s) ... -> b s ...", b=batch_size, s=seqlen_q)
                if mask_type == "cu_seqlens"
                else attn_output
            )
        else:
            attn_output = self.inner_attn(q, kv, key_padding_mask=attention_mask, causal=causal)

        output = rearrange(attn_output, "... h d -> ... (h d)")
        output = self.out_proj(output)

        return output
