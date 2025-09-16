# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    PhiConfig,
    PhiForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


def convert_llama_to_phi3(model: LlamaForCausalLM) -> Phi3ForCausalLM:
    """Convert a Llama model to a Phi-3 model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    if model.config.attention_bias is True:
        raise ValueError(f"`attention_bias` must be False, but got {model.config.attention_bias}.")

    config_dict = {
        "attention_dropout": model.config.attention_dropout,
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "hidden_act": model.config.hidden_act,
        "hidden_size": model.config.hidden_size,
        "initializer_range": model.config.initializer_range,
        "intermediate_size": model.config.intermediate_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_key_value_heads": model.config.num_key_value_heads,
        "original_max_position_embeddings": model.config.max_position_embeddings,
        "pad_token_id": model.config.pad_token_id,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_scaling": model.config.rope_scaling,
        "rope_theta": model.config.rope_theta,
        "vocab_size": model.config.vocab_size,
    }

    config = Phi3Config(**config_dict)
    converted_model = Phi3ForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP (down projection), attention, layer normalization and head
    num_hidden_layers = converted_model.config.num_hidden_layers
    mapping_dict = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"model.layers.{i}.{key}": f"model.layers.{i}.{converted_key}"
            for key, converted_key in [
                ("input_layernorm.weight", "input_layernorm.weight"),
                ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
                ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
                ("mlp.down_proj.weight", "mlp.down_proj.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP (`gate_up_proj`) needs to be concatenated into a single tensor
    for i in range(num_hidden_layers):
        gate = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        converted_state_dict[f"model.layers.{i}.mlp.gate_up_proj.weight"].copy_(torch.cat((gate, up), dim=0))

    # Queries, keys and values need to be concatenated into a single tensor
    for i in range(num_hidden_layers):
        q = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        k = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        v = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        converted_state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"].copy_(torch.cat([q, k, v], dim=0))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def _convert_mixformer_sequential_to_llama_or_mistral(
    model: MixFormerSequentialForCausalLM, model_type: str
) -> Union[LlamaForCausalLM, MistralForCausalLM]:
    if model_type not in ["llama", "mistral"]:
        raise ValueError(f"`model_type` must be 'llama' or 'mistral', but got '{model_type}'.")
    if model.config.architecture["block_cls"] != "sequential":
        raise ValueError(f"`block_cls` must be 'sequential', but got '{model.config.architecture['block_cls']}'.")
    if model.config.architecture["mlp"]["act_fn"] != "silu":
        raise ValueError(f"`act_fn` must be 'silu', but got '{model.config.architecture['mlp']['act_fn']}'.")
    if model.config.architecture["mlp"]["mlp_cls"] != "glu":
        raise ValueError(f"`mlp_cls` must be 'glu', but got '{model.config.architecture['mlp']['mlp_cls']}'.")
    if "rms" not in model.config.architecture["norm"]["norm_cls"]:
        raise ValueError(
            f"`norm_cls` should have 'rms' in it, but got '{model.config.architecture['norm']['norm_cls']}'."
        )

    attn_bias = model.config.architecture["mixer"].get("bias", False)
    attn_out_bias = model.config.architecture["mixer"].get("out_bias", attn_bias)

    rope_scaling = model.config.architecture["mixer"].get("rope_scaling", None)
    if rope_scaling is not None:
        if "rotary_scale_base" not in model.config.architecture["mixer"]:
            raise ValueError(
                f"`rope_scaling` must have `rotary_scale_base`, but got {model.config.architecture['mixer']}."
            )
        rope_scaling["factor"] = model.config.architecture["mixer"]["rotary_scale_base"]

    config_dict = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.n_embd,
        "attention_bias": attn_bias,
        "attention_dropout": model.config.architecture["mixer"].get("dropout", 0.0),
        "initializer_range": model.config.initializer_range,
        "intermediate_size": model.config.architecture["mlp"]["n_inner"],
        "num_hidden_layers": model.config.n_layer,
        "num_attention_heads": model.config.n_head,
        "num_key_value_heads": model.config.n_head_kv or model.config.architecture["mixer"].get("n_head_kv", None),
        "rope_theta": model.config.architecture["mixer"].get("rotary_base", 10000.0),
        "rope_scaling": rope_scaling,
        "max_position_embeddings": model.config.n_positions,
        "rms_norm_eps": model.config.layer_norm_epsilon,
    }

    if model_type == "llama":
        config_cls, model_cls = LlamaConfig, LlamaForCausalLM
    elif model_type == "mistral":
        config_cls, model_cls = MistralConfig, MistralForCausalLM

        if model.config.architecture["mixer"].get("window_size", None) is None:
            logger.warning("Model does not have `window_size`, setting it to `(-1, -1)`.")
            model.config.architecture["mixer"]["window_size"] = "(-1, -1)"

        sliding_window = eval(model.config.architecture["mixer"]["window_size"])
        config_dict["sliding_window"] = abs(sliding_window[0] - sliding_window[1])

    config = config_cls(**config_dict)
    converted_model = model_cls(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP (down projection), attention (output projection), layer normalization and head
    num_hidden_layers = converted_model.config.num_hidden_layers
    if f"{num_hidden_layers}.linear.bias" in state_dict:
        logger.warning("Model does not use `lm_head.bias`. Outputs will be different.")

    mapping_dict = {
        "layers.0.wte.weight": "model.embed_tokens.weight",
        f"layers.{num_hidden_layers+1}.ln.weight": "model.norm.weight",
        f"layers.{num_hidden_layers+1}.linear.weight": "lm_head.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"layers.{i+1}.{key}": f"model.layers.{i}.{converted_key}"
            for key, converted_key in [
                ("ln_1.weight", "input_layernorm.weight"),
                ("ln_2.weight", "post_attention_layernorm.weight"),
                ("attn.out_proj.weight", "self_attn.o_proj.weight"),
                ("mlp.fc2.weight", "mlp.down_proj.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP (`fc1`) needs to be split into two parts (up and gate projections)
    for i in range(num_hidden_layers):
        up, gate = torch.chunk(state_dict[f"layers.{i+1}.mlp.fc1.weight"], 2, dim=0)
        converted_state_dict[f"model.layers.{i}.mlp.up_proj.weight"].copy_(up)
        converted_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].copy_(gate)

    # Queries, keys and values need to be dealt separately because they are concatenated in MixFormer
    num_attention_heads = converted_model.config.num_attention_heads
    num_key_value_heads = converted_model.config.num_key_value_heads or num_attention_heads
    head_dim = converted_model.config.hidden_size // converted_model.config.num_attention_heads

    if attn_bias:
        attn_param_names = ["weight", "bias"]
    else:
        attn_param_names = ["weight"]

    for i in range(num_hidden_layers):
        for attn_param in attn_param_names:
            key = f"layers.{i+1}.attn.Wqkv.{attn_param}"
            old_param = state_dict[key]

            projs = {
                "q": old_param[: num_attention_heads * head_dim],
                "k": old_param[
                    num_attention_heads * head_dim : num_attention_heads * head_dim + num_key_value_heads * head_dim
                ],
                "v": old_param[num_attention_heads * head_dim + num_key_value_heads * head_dim :],
            }

            for key, value in projs.items():
                converted_key = f"model.layers.{i}.self_attn.{key}_proj.{attn_param}"
                converted_state_dict[converted_key].copy_(value)

    if attn_bias is True and attn_out_bias is False:
        logger.warning("Llama does not support out_bias=False, so the bias weights are being zeroed out.")
        for name, param in converted_state_dict.items():
            if "self_attn.o_proj.bias" in name:
                converted_state_dict[name].copy_(torch.zeros_like(param))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def convert_mixformer_sequential_to_llama(model: MixFormerSequentialForCausalLM) -> LlamaForCausalLM:
    """Convert a MixFormer (Sequential) model to a Llama model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    return _convert_mixformer_sequential_to_llama_or_mistral(model, "llama")


def convert_mixformer_sequential_to_mistral(model: MixFormerSequentialForCausalLM) -> MistralForCausalLM:
    """Convert a MixFormer (Sequential) model to a Mistral model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    return _convert_mixformer_sequential_to_llama_or_mistral(model, "mistral")


def convert_mixformer_sequential_to_phi(model: MixFormerSequentialForCausalLM) -> PhiForCausalLM:
    """Convert a MixFormer (Sequential) model to a Phi model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    if model.config.architecture["block_cls"] != "parallel":
        raise ValueError(f"`block_cls` must be 'parallel', but got '{model.config.architecture['block_cls']}'.")
    if model.config.tie_word_embeddings:
        raise ValueError(f"`tie_word_embeddings` must be False, but got {model.config.tie_word_embeddings}.")

    config = PhiConfig(
        embd_pdrop=model.config.embd_pdrop,
        hidden_act=model.config.activation_function,
        hidden_size=model.config.n_embd,
        initializer_range=model.config.initializer_range,
        intermediate_size=model.config.n_embd * 4,
        layer_norm_eps=model.config.layer_norm_epsilon,
        max_position_embeddings=model.config.n_positions,
        num_attention_heads=model.config.n_head,
        num_hidden_layers=model.config.n_layer,
        partial_rotary_factor=model.config.rotary_dim / (model.config.n_embd / model.config.n_head),
        resid_pdrop=model.config.resid_pdrop,
        vocab_size=model.config.vocab_size,
    )
    converted_model = PhiForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP, attention (output projection), layer normalization and head
    num_hidden_layers = converted_model.config.num_hidden_layers
    mapping_dict = {
        "layers.0.wte.weight": "model.embed_tokens.weight",
        f"layers.{num_hidden_layers+1}.ln.weight": "model.final_layernorm.weight",
        f"layers.{num_hidden_layers+1}.ln.bias": "model.final_layernorm.bias",
        f"layers.{num_hidden_layers+1}.linear.weight": "lm_head.weight",
        f"layers.{num_hidden_layers+1}.linear.bias": "lm_head.bias",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"layers.{i+1}.{key}": f"model.layers.{i}.{converted_key}"
            for key, converted_key in [
                ("mlp.mlp.fc1.weight", "mlp.fc1.weight"),
                ("mlp.mlp.fc1.bias", "mlp.fc1.bias"),
                ("mlp.mlp.fc2.weight", "mlp.fc2.weight"),
                ("mlp.mlp.fc2.bias", "mlp.fc2.bias"),
                ("ln.weight", "input_layernorm.weight"),
                ("ln.bias", "input_layernorm.bias"),
                ("mixer.out_proj.weight", "self_attn.dense.weight"),
                ("mixer.out_proj.bias", "self_attn.dense.bias"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # Queries, keys and values need to be dealt separately because they are individual tensors in Phi
    num_attention_heads = converted_model.config.num_attention_heads
    num_key_value_heads = converted_model.config.num_key_value_heads or num_attention_heads
    head_dim = converted_model.config.hidden_size // converted_model.config.num_attention_heads

    for i in range(num_hidden_layers):
        for attn_param in ["weight", "bias"]:
            key = f"layers.{i+1}.mixer.Wqkv.{attn_param}"
            old_param = state_dict[key]

            projs = {
                "q": old_param[: num_attention_heads * head_dim],
                "k": old_param[
                    num_attention_heads * head_dim : num_attention_heads * head_dim + num_key_value_heads * head_dim
                ],
                "v": old_param[num_attention_heads * head_dim + num_key_value_heads * head_dim :],
            }

            for key, value in projs.items():
                converted_key = f"model.layers.{i}.self_attn.{key}_proj.{attn_param}"
                converted_state_dict[converted_key].copy_(value)

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def _convert_mixformer_sequential_to_phi3_or_phi4(
    model: MixFormerSequentialForCausalLM, model_type: str
) -> Phi3ForCausalLM:
    attn_bias = model.config.architecture["mixer"].get("bias", False)
    head_bias = model.config.architecture["head"].get("use_bias", False)

    if attn_bias is True:
        raise ValueError(f"`attn_bias` must be False, but got {attn_bias}.")
    if head_bias is True:
        raise ValueError(f"`head_bias` must be False, but got {head_bias}.")
    if model_type not in ["phi3", "phi4"]:
        raise ValueError(f"`model_type` must be 'phi3' or 'phi4', but got '{model_type}'.")
    if model.config.architecture["block_cls"] != "sequential":
        raise ValueError(f"`block_cls` must be 'sequential', but got '{model.config.architecture['block_cls']}'.")
    if model.config.architecture["mlp"]["act_fn"] != "silu":
        raise ValueError(f"`act_fn` must be 'silu', but got '{model.config.architecture['mlp']['act_fn']}'.")
    if model.config.architecture["mlp"]["mlp_cls"] != "glu":
        raise ValueError(f"`mlp_cls` must be 'glu', but got '{model.config.architecture['mlp']['mlp_cls']}'.")
    if "rms" not in model.config.architecture["norm"]["norm_cls"]:
        raise ValueError(
            f"`norm_cls` should have 'rms' in it, but got '{model.config.architecture['norm']['norm_cls']}'."
        )
    if "rope_scaling" in model.config.architecture["mixer"]:
        raise ValueError(
            f"`rope_scaling` must not be in `architecture.mixer`, but got {model.config.architecture['mixer']}."
        )

    sliding_window = model.config.architecture["mixer"].get("window_size", None)
    if isinstance(sliding_window, str):
        sliding_window = eval(sliding_window)
        if sliding_window == (-1, -1):
            sliding_window = None
        else:
            sliding_window = abs(sliding_window[0] - sliding_window[1])

    config_dict = {
        "attention_dropout": model.config.architecture["mixer"].get("dropout", 0.0),
        "bos_token_id": model.config.bos_token_id,
        "embd_pdrop": model.config.embd_pdrop,
        "eos_token_id": model.config.eos_token_id,
        "hidden_act": model.config.architecture["mlp"]["act_fn"],
        "hidden_size": model.config.n_embd,
        "initializer_range": model.config.initializer_range,
        "intermediate_size": model.config.architecture["mlp"]["n_inner"],
        "max_position_embeddings": model.config.n_positions,
        "num_attention_heads": model.config.n_head,
        "num_hidden_layers": model.config.n_layer,
        "num_key_value_heads": model.config.n_head_kv or model.config.architecture["mixer"].get("n_head_kv", None),
        "original_max_position_embeddings": model.config.n_positions,
        "pad_token_id": model.config.pad_token_id,
        "partial_rotary_factor": model.config.rotary_dim / (model.config.n_embd / model.config.n_head),
        "resid_pdrop": model.config.resid_pdrop,
        "rms_norm_eps": model.config.layer_norm_epsilon,
        "rope_scaling": None,
        "rope_theta": model.config.architecture["mixer"].get("rotary_base", 10000),
        "sliding_window": sliding_window,
        "vocab_size": model.config.vocab_size,
        "tie_word_embeddings": model.config.tie_word_embeddings,
    }

    config = Phi3Config(**config_dict)
    converted_model = Phi3ForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP, attention, layer normalization and head
    num_hidden_layers = converted_model.config.num_hidden_layers
    if f"{num_hidden_layers}.linear.bias" in state_dict:
        logger.warning("Model does not use `lm_head.bias`. Outputs will be different.")

    mapping_dict = {
        "layers.0.wte.weight": "model.embed_tokens.weight",
        f"layers.{num_hidden_layers+1}.ln.weight": "model.norm.weight",
        f"layers.{num_hidden_layers+1}.linear.weight": "lm_head.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"layers.{i+1}.{key}": f"model.layers.{i}.{converted_key}"
            for key, converted_key in [
                ("ln_1.weight", "input_layernorm.weight"),
                ("ln_2.weight", "post_attention_layernorm.weight"),
                ("attn.Wqkv.weight", "self_attn.qkv_proj.weight"),
                ("attn.out_proj.weight", "self_attn.o_proj.weight"),
                ("mlp.fc2.weight", "mlp.down_proj.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP (`fc1`) needs to be split into two parts (up and gate projections) and
    # re-ordered to match the Phi-3 or Phi-4 model
    for i in range(num_hidden_layers):
        up, gate = torch.chunk(state_dict[f"layers.{i+1}.mlp.fc1.weight"], 2, dim=0)
        converted_state_dict[f"model.layers.{i}.mlp.gate_up_proj.weight"].copy_(torch.cat((gate, up), dim=0))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def convert_mixformer_sequential_to_phi3(model: MixFormerSequentialForCausalLM) -> Phi3ForCausalLM:
    """Convert a MixFormer (Sequential) model to a Phi-3 model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    return _convert_mixformer_sequential_to_phi3_or_phi4(model, "phi3")


def convert_mixformer_sequential_to_phi4(model: MixFormerSequentialForCausalLM) -> Phi3ForCausalLM:
    """Convert a MixFormer (Sequential) model to a Phi-4 model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    return _convert_mixformer_sequential_to_phi3_or_phi4(model, "phi4")


def convert_phi3_to_mixformer_sequential(model: Phi3ForCausalLM) -> MixFormerSequentialForCausalLM:
    """Convert a Phi-3 model to a MixFormer (Sequential) model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    window_size = model.config.sliding_window
    if window_size is not None:
        window_size = f"({window_size}, 0)"

    # TODO: act_function not silu is not handled gracefully

    config_dict = {
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.num_hidden_layers,
        "n_embd": model.config.hidden_size,
        "n_head": model.config.num_attention_heads,
        "n_positions": model.config.max_position_embeddings,
        "initializer_range": model.config.initializer_range,
        "layer_norm_epsilon": model.config.rms_norm_eps,
        "resid_pdrop": model.config.resid_pdrop,
        "embd_pdrop": model.config.embd_pdrop,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "rotary_dim": int(
            model.config.hidden_size // model.config.num_attention_heads * model.config.partial_rotary_factor
        ),
        "architecture": {
            "block_cls": "sequential",
            "head": {"head_cls": "causal_lm", "use_bias": False},
            "mixer": {
                "bias": False,
                "dropout": model.config.attention_dropout,
                "n_head_kv": model.config.num_key_value_heads,
                "rotary_base": model.config.rope_theta,
                "mixer_cls": "mha",
                "window_size": window_size,
            },
            "mlp": {"act_fn": "silu", "mlp_cls": "glu", "n_inner": model.config.intermediate_size},
            "norm": {"norm_cls": "flash_rms"},
        },
    }

    config = MixFormerSequentialConfig(**config_dict)
    converted_model = MixFormerSequentialForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP (down projection), attention, layer normalization and head
    num_hidden_layers = converted_model.config.n_layer
    mapping_dict = {
        "model.embed_tokens.weight": "layers.0.wte.weight",
        "lm_head.weight": f"layers.{num_hidden_layers+1}.linear.weight",
        "model.norm.weight": f"layers.{num_hidden_layers+1}.ln.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"model.layers.{i}.{key}": f"layers.{i+1}.{converted_key}"
            for key, converted_key in [
                ("input_layernorm.weight", "ln_1.weight"),
                ("post_attention_layernorm.weight", "ln_2.weight"),
                ("self_attn.qkv_proj.weight", "attn.Wqkv.weight"),
                ("self_attn.o_proj.weight", "attn.out_proj.weight"),
                ("mlp.down_proj.weight", "mlp.fc2.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP (`gate_up_proj`) needs to be split into two parts (gate and up projections)
    # and re-ordered to match the MixFormer model
    for i in range(num_hidden_layers):
        gate, up = torch.chunk(state_dict[f"model.layers.{i}.mlp.gate_up_proj.weight"], 2, dim=0)
        converted_state_dict[f"layers.{i+1}.mlp.fc1.weight"].copy_(torch.cat((up, gate), dim=0))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def convert_qwen2_to_mixformer_sequential(model: Qwen2ForCausalLM) -> MixFormerSequentialForCausalLM:
    """Convert a Qwen2 model to a MixFormer (Sequential) model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    if model.config.tie_word_embeddings:
        logger.info("`tie_word_embeddings=True` causes a mismatch in the real number of parameters.")

    config_dict = {
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.num_hidden_layers,
        "n_embd": model.config.hidden_size,
        "n_head": model.config.num_attention_heads,
        "n_positions": model.config.max_position_embeddings,
        "initializer_range": model.config.initializer_range,
        "layer_norm_epsilon": model.config.rms_norm_eps,
        "rotary_dim": model.config.hidden_size // model.config.num_attention_heads,
        "architecture": {
            "block_cls": "sequential",
            "head": {"head_cls": "causal_lm", "use_bias": False},
            "mixer": {
                "mixer_cls": "mha",
                "n_head_kv": model.config.num_key_value_heads,
                "dropout": model.config.attention_dropout,
                "bias": True,
                "out_bias": False,
                "rotary_base": model.config.rope_theta,
            },
            "mlp": {"act_fn": "silu", "mlp_cls": "glu", "n_inner": model.config.intermediate_size},
            "norm": {"norm_cls": "flash_rms"},
        },
    }

    config = MixFormerSequentialConfig(**config_dict)
    converted_model = MixFormerSequentialForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP (down projection), attention (output projection), layer normalization and head
    num_hidden_layers = converted_model.config.n_layer
    mapping_dict = {
        "model.embed_tokens.weight": "layers.0.wte.weight",
        "lm_head.weight": f"layers.{num_hidden_layers+1}.linear.weight",
        "model.norm.weight": f"layers.{num_hidden_layers+1}.ln.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"model.layers.{i}.{key}": f"layers.{i+1}.{converted_key}"
            for key, converted_key in [
                ("input_layernorm.weight", "ln_1.weight"),
                ("post_attention_layernorm.weight", "ln_2.weight"),
                ("self_attn.o_proj.weight", "attn.out_proj.weight"),
                ("mlp.down_proj.weight", "mlp.fc2.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP gate and up projections needs to be re-ordered to match the MixFormer model
    for i in range(num_hidden_layers):
        gate = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        converted_state_dict[f"layers.{i+1}.mlp.fc1.weight"].copy_(torch.cat((up, gate), dim=0))

    # Queries, keys and values need to be concatenated into a single tensor
    for i in range(num_hidden_layers):
        for attn_param in ["weight", "bias"]:
            q = state_dict[f"model.layers.{i}.self_attn.q_proj.{attn_param}"]
            k = state_dict[f"model.layers.{i}.self_attn.k_proj.{attn_param}"]
            v = state_dict[f"model.layers.{i}.self_attn.v_proj.{attn_param}"]
            converted_state_dict[f"layers.{i+1}.attn.Wqkv.{attn_param}"].copy_(torch.cat([q, k, v], dim=0))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model


def convert_qwen3_to_mixformer_sequential(model: Qwen3ForCausalLM) -> MixFormerSequentialForCausalLM:
    """Convert a Qwen3 model to a MixFormer (Sequential) model.

    Args:
        model: Model to be converted.

    Returns:
        Converted model.

    """

    if model.config.tie_word_embeddings:
        logger.info("`tie_word_embeddings=True` causes a mismatch in the real number of parameters.")

    config_dict = {
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.num_hidden_layers,
        "n_embd": model.config.hidden_size,
        "n_head": model.config.num_attention_heads,
        "n_positions": model.config.max_position_embeddings,
        "initializer_range": model.config.initializer_range,
        "layer_norm_epsilon": model.config.rms_norm_eps,
        "rotary_dim": model.config.head_dim,
        "architecture": {
            "block_cls": "sequential",
            "head": {"head_cls": "causal_lm", "use_bias": False},
            "mixer": {
                "mixer_cls": "mha",
                "n_head_kv": model.config.num_key_value_heads,
                "head_dim": model.config.head_dim,
                "dropout": model.config.attention_dropout,
                "bias": False,
                "out_bias": False,
                "qk_norm": True,
                "rotary_base": model.config.rope_theta,
            },
            "mlp": {"act_fn": "silu", "mlp_cls": "glu", "n_inner": model.config.intermediate_size},
            "norm": {"norm_cls": "rms"},
        },
    }

    config = MixFormerSequentialConfig(**config_dict)
    converted_model = MixFormerSequentialForCausalLM(config)

    state_dict = model.state_dict()
    converted_state_dict = converted_model.state_dict()

    # Initial mapping dictionary
    # Embedding, MLP (down projection), attention (output projection), layer normalization and head
    num_hidden_layers = converted_model.config.n_layer
    mapping_dict = {
        "model.embed_tokens.weight": "layers.0.wte.weight",
        "lm_head.weight": f"layers.{num_hidden_layers+1}.linear.weight",
        "model.norm.weight": f"layers.{num_hidden_layers+1}.ln.weight",
    }
    for i in range(num_hidden_layers):
        layer_mappings = {
            f"model.layers.{i}.{key}": f"layers.{i+1}.{converted_key}"
            for key, converted_key in [
                ("input_layernorm.weight", "ln_1.weight"),
                ("post_attention_layernorm.weight", "ln_2.weight"),
                ("self_attn.q_norm.weight", "attn.q_norm.weight"),
                ("self_attn.k_norm.weight", "attn.k_norm.weight"),
                ("self_attn.o_proj.weight", "attn.out_proj.weight"),
                ("mlp.down_proj.weight", "mlp.fc2.weight"),
            ]
        }
        mapping_dict.update(layer_mappings)

    for key, converted_key in mapping_dict.items():
        converted_key_shape, key_shape = converted_state_dict[converted_key].shape, state_dict[key].shape
        if converted_key_shape == key_shape:
            converted_state_dict[converted_key].copy_(state_dict[key])
        else:
            raise ValueError(f"'{converted_key}' shape must be {key_shape}, but got {converted_key_shape}.")

    # MLP gate and up projections needs to be re-ordered to match the MixFormer model
    for i in range(num_hidden_layers):
        gate = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        up = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        converted_state_dict[f"layers.{i+1}.mlp.fc1.weight"].copy_(torch.cat((up, gate), dim=0))

    # Queries, keys and values need to be concatenated into a single tensor
    for i in range(num_hidden_layers):
        for attn_param in ["weight"]:
            q = state_dict[f"model.layers.{i}.self_attn.q_proj.{attn_param}"]
            k = state_dict[f"model.layers.{i}.self_attn.k_proj.{attn_param}"]
            v = state_dict[f"model.layers.{i}.self_attn.v_proj.{attn_param}"]
            converted_state_dict[f"layers.{i+1}.attn.Wqkv.{attn_param}"].copy_(torch.cat([q, k, v], dim=0))

    converted_model.load_state_dict(converted_state_dict)

    return converted_model
