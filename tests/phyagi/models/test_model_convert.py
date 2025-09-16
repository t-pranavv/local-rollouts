# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.models.model_convert import (
    convert_llama_to_phi3,
    convert_mixformer_sequential_to_llama,
    convert_mixformer_sequential_to_mistral,
    convert_mixformer_sequential_to_phi,
    convert_mixformer_sequential_to_phi3,
    convert_mixformer_sequential_to_phi4,
    convert_phi3_to_mixformer_sequential,
    convert_qwen2_to_mixformer_sequential,
)


def dummy_llama_model(attention_bias=False):
    config = LlamaConfig(
        attention_bias=attention_bias,
        attention_dropout=0.1,
        bos_token_id=3,
        eos_token_id=4,
        hidden_act="silu",
        hidden_size=16,
        initializer_range=0.01,
        intermediate_size=64,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        pad_token_id=1,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=9000,
        vocab_size=64,
    )
    return LlamaForCausalLM(config)


def dummy_mixformer_sequential():
    config = MixFormerSequentialConfig(
        vocab_size=64,
        n_layer=2,
        n_embd=16,
        n_head=2,
        n_positions=32,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        rotary_dim=8,
        pad_token_id=1,
        architecture={
            "block_cls": "sequential",
            "head": {"head_cls": "causal_lm", "use_bias": False},
            "mixer": {
                "bias": False,
                "out_bias": False,
                "dropout": 0.1,
                "n_head_kv": None,
                "mixer_cls": "mha",
                "rotary_base": 10000,
                "window_size": "(-1, -1)",
            },
            "mlp": {"act_fn": "silu", "mlp_cls": "glu", "n_inner": 64},
            "norm": {"norm_cls": "flash_rms"},
        },
    )
    return MixFormerSequentialForCausalLM(config)


def dummy_mixformer_sequential_parallel():
    config = MixFormerSequentialConfig(
        vocab_size=64,
        n_layer=2,
        n_embd=16,
        n_head=2,
        n_positions=32,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        rotary_dim=8,
        pad_token_id=0,
        architecture={
            "block_cls": "parallel",
            "head": {"head_cls": "causal_lm", "use_bias": False},
            "mixer": {
                "bias": False,
                "out_bias": False,
                "dropout": 0.1,
                "n_head_kv": None,
                "mixer_cls": "mha",
                "rotary_base": 10000,
                "window_size": "(-1, -1)",
            },
            "mlp": {"act_fn": "silu", "mlp_cls": "glu", "n_inner": 64},
            "norm": {"norm_cls": "flash_rms"},
        },
    )
    return MixFormerSequentialForCausalLM(config)


def dummy_phi3():
    config = Phi3Config(
        attention_dropout=0.1,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=16,
        initializer_range=0.01,
        intermediate_size=64,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        original_max_position_embeddings=32,
        pad_token_id=0,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=9000,
        sliding_window=None,
        vocab_size=64,
    )
    return Phi3ForCausalLM(config)


def dummy_qwen2():
    config = Qwen2Config(
        vocab_size=64,
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        max_position_embeddings=32,
        initializer_range=0.01,
        rms_norm_eps=1e-5,
        rope_theta=9000,
        attention_dropout=0.1,
        num_key_value_heads=4,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)


def test_convert_llama_to_phi3():
    llama_model_with_attn_bias = dummy_llama_model(attention_bias=True)
    with pytest.raises(ValueError):
        convert_llama_to_phi3(llama_model_with_attn_bias)

    llama_model = dummy_llama_model()
    phi3_model = convert_llama_to_phi3(llama_model)
    assert isinstance(phi3_model, Phi3ForCausalLM)
    for attr in [
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
    ]:
        assert getattr(phi3_model.config, attr) == getattr(llama_model.config, attr)


@pytest.mark.parametrize(
    ("converter_fn, expected_cls"),
    [
        pytest.param(convert_mixformer_sequential_to_llama, LlamaForCausalLM, id="to_llama"),
        pytest.param(convert_mixformer_sequential_to_mistral, MistralForCausalLM, id="to_mistral"),
    ],
)
def test_convert_mixformer_sequential_to_llama_or_mistral(converter_fn, expected_cls):
    mixformer_model = dummy_mixformer_sequential()
    converted_model = converter_fn(mixformer_model)
    assert isinstance(converted_model, expected_cls)


@pytest.mark.parametrize(
    ("converter_fn, expected_exception"),
    [
        pytest.param(convert_mixformer_sequential_to_llama, ValueError, id="llama_with_attn_bias"),
        pytest.param(convert_mixformer_sequential_to_phi, KeyError, id="phi_with_attn_bias"),
    ],
)
def test_convert_parallel_mixformer_fails(converter_fn, expected_exception):
    model = dummy_mixformer_sequential_parallel()
    with pytest.raises(expected_exception):
        converter_fn(model)


@pytest.mark.parametrize(
    ("converter_fn"),
    [
        pytest.param(convert_mixformer_sequential_to_phi3, id="to_phi3"),
        pytest.param(convert_mixformer_sequential_to_phi4, id="to_phi4"),
    ],
)
def test_convert_mixformer_sequential_to_phi_variants(converter_fn):
    mixformer_model = dummy_mixformer_sequential()
    phi_model = converter_fn(mixformer_model)
    assert isinstance(phi_model, Phi3ForCausalLM)


def test_convert_phi3_to_mixformer_sequential():
    phi3_model = dummy_phi3()

    mixformer_sequential_model = convert_phi3_to_mixformer_sequential(phi3_model)
    assert isinstance(mixformer_sequential_model, MixFormerSequentialForCausalLM)
    assert mixformer_sequential_model.config.vocab_size == phi3_model.config.vocab_size
    assert mixformer_sequential_model.config.n_layer == phi3_model.config.num_hidden_layers


def test_convert_qwen2_to_mixformer_sequential():
    qwen2_model = dummy_qwen2()

    mixformer_sequential_model = convert_qwen2_to_mixformer_sequential(qwen2_model)
    assert isinstance(mixformer_sequential_model, MixFormerSequentialForCausalLM)
    assert mixformer_sequential_model.config.architecture["mixer"]["bias"] is True
    assert mixformer_sequential_model.config.architecture["mixer"]["out_bias"] is False
