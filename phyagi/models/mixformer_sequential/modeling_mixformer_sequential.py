# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import Cache, DynamicCache, GenerationMixin
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from phyagi.models.mixformer_sequential.blocks import BLOCKS, get_block
from phyagi.models.mixformer_sequential.blocks.embeddings import get_embedding
from phyagi.models.mixformer_sequential.blocks.heads import get_head, get_loss
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.modeling_utils import PreTrainedModel


def _maybe_get_cu_seqlens(input_ids: torch.LongTensor, cross_sample_token_id: int) -> Optional[torch.LongTensor]:
    if not (input_ids == cross_sample_token_id).any():
        return None

    # If the last token in the sequence is not an `eos_token`, we manually set it
    input_ids[input_ids[:, -1] != cross_sample_token_id, -1] = cross_sample_token_id

    # Calculates the cumulative sequence lengths for the entire batch, with the caveat that
    # Flash-Attention expects the first element to be 0
    cu_seqlens = torch.where(input_ids.reshape(-1) == cross_sample_token_id)[0] + 1

    return torch.cat([torch.zeros(1).to(cu_seqlens.device), cu_seqlens], axis=0).int()


class MixFormerSequentialPreTrainedModel(PreTrainedModel):
    """MixFormer (sequential) pre-trained model."""

    config_class = MixFormerSequentialConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = [v.__name__ for v in BLOCKS.values()]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def _from_config(
        cls: MixFormerSequentialPreTrainedModel, config: MixFormerSequentialConfig, **kwargs
    ) -> MixFormerSequentialPreTrainedModel:
        # `transformers` is unable to apply `torch_dtype` to an `nn.Sequential` model,
        # so we need to apply it manually after loading the model
        torch_dtype = getattr(config, "torch_dtype", None)

        model = super()._from_config(config, **kwargs)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        return model

    @classmethod
    def from_pretrained(
        cls: MixFormerSequentialPreTrainedModel,
        pretrained_model_name_or_path: str,
        *args,
        **kwargs,
    ) -> MixFormerSequentialPreTrainedModel:
        # `transformers` is unable to apply `torch_dtype` to an `nn.Sequential` model,
        # so we need to apply it manually after loading the model
        torch_dtype = kwargs.get("torch_dtype", None)

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if torch_dtype is not None:
            model = model.to(torch_dtype)

        return model

    def post_init(self) -> None:
        self.init_weights()

        if self.base_model is self:
            self._tp_plan = self.config.base_model_tp_plan


class MixFormerSequentialForCausalLM(MixFormerSequentialPreTrainedModel, GenerationMixin):
    """MixFormer (sequential) for Causal Language Modeling."""

    def __init__(self, config: MixFormerSequentialConfig) -> None:
        super().__init__(config)

        head = config.architecture.get("head", None) if isinstance(config.architecture, dict) else None
        head = head if head is not None else {"head_cls": "causal_lm"}

        if head["head_cls"] != "causal_lm":
            raise ValueError(f"`head_cls` must be 'causal_lm', but got '{head['head_cls']}'.")

        loss = config.architecture.get("loss", None) if isinstance(config.architecture, dict) else None
        loss = loss if loss is not None else {"loss_cls": "causal_lm"}

        if loss["loss_cls"] != "causal_lm":
            raise ValueError(f"`loss_cls` must be 'causal_lm', but got '{loss['loss_cls']}'.")

        layers = [get_embedding(config, embedding_config={"embedding_cls": config.embd_layer})]
        layers += get_block(config, block_config=config.architecture)
        layers.append(get_head(config, head_config=head))

        self.layers = nn.Sequential(*layers)
        self.loss = get_loss(loss_config=loss)

        self.gradient_checkpointing = config.gradient_checkpointing
        self.cross_sample_token_id = config.cross_sample_token_id

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.layers[0].wte = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.layers[-1].linear

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.layers[-1].linear = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.training:
            if use_cache:
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if cu_seqlens is None and self.cross_sample_token_id is not None:
            # If no cross-sample separator is present, cu_seqlens will still be None
            cu_seqlens = _maybe_get_cu_seqlens(input_ids, self.cross_sample_token_id)

        hidden_states = self.layers[0](input_ids, position_ids=position_ids, input_embeds=input_embeds)
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers[1:-1]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    layer, hidden_states, attention_mask, position_ids, past_key_values, cu_seqlens, use_reentrant=True
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cu_seqlens=cu_seqlens,
                )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.layers[-1](hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MixFormerSequentialForSequenceClassification(MixFormerSequentialPreTrainedModel):
    """MixFormer (sequential) for Sequence Classification."""

    def __init__(self, config: MixFormerSequentialConfig) -> None:
        super().__init__(config)

        head = config.architecture.get("head", None) if isinstance(config.architecture, dict) else None
        head = head if head is not None else {"head_cls": "seq_cls"}

        if head["head_cls"] != "seq_cls":
            raise ValueError(f"`head_cls` must be 'seq_cls', but got '{head['head_cls']}'.")

        loss = config.architecture.get("loss", None) if isinstance(config.architecture, dict) else None
        loss = loss if loss is not None else {"loss_cls": "seq_cls"}

        if loss["loss_cls"] != "seq_cls":
            raise ValueError(f"`loss_cls` must be 'seq_cls', but got '{loss['loss_cls']}'.")

        layers = [get_embedding(config, embedding_config={"embedding_cls": config.embd_layer})]
        layers += get_block(config, block_config=config.architecture)
        layers.append(get_head(config, head_config=head))

        self.layers = nn.Sequential(*layers)
        self.loss = get_loss(config, loss_config=loss)

        self.gradient_checkpointing = config.gradient_checkpointing
        self.cross_sample_token_id = config.cross_sample_token_id

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.layers[0].wte = value

    def get_output_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_output_embeddings(self, value: nn.Embedding) -> None:
        self.layers[0].wte = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if cu_seqlens is None and self.cross_sample_token_id is not None:
            # If no cross-sample separator is present, cu_seqlens will still be None
            cu_seqlens = _maybe_get_cu_seqlens(input_ids, self.cross_sample_token_id)

        hidden_states = self.layers[0](input_ids, position_ids=position_ids, input_embeds=input_embeds)
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers[1:-1]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    layer, hidden_states, attention_mask, position_ids, past_key_values, cu_seqlens, use_reentrant=True
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cu_seqlens=cu_seqlens,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.layers[-1](hidden_states)

        batch_size, seq_lens = input_ids.shape[0], -1
        if batch_size > 1 and self.config.pad_token_id is None:
            raise ValueError("`pad_token_id` must be defined if `batch_size > 1`, but got None.")

        if self.config.pad_token_id is not None:
            seq_lens = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lens]

        loss = None
        if labels is not None:
            loss = self.loss(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
