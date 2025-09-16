# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import pytest
import torch
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
    MixFormerSequentialForSequenceClassification,
)


def test_mixformer_sequential_config():
    config = MixFormerSequentialConfig()

    assert config.vocab_size == int(math.ceil(50304 / 64) * 64)
    assert config.n_positions == 2048
    assert config.n_embd == 1024
    assert config.n_layer == 20
    assert config.n_inner is None
    assert config.n_head == 16
    assert config.n_head_kv is None
    assert config.rotary_dim == 1024 // 16
    assert config.activation_function == "gelu_new"
    assert config.embd_layer == "default"
    assert config.architecture == MixFormerSequentialConfig.default_arch
    assert config.embd_pdrop == 0.0
    assert config.resid_pdrop == 0.0
    assert config.layer_norm_epsilon == 1e-5
    assert config.initializer_range == 0.02
    assert config.tie_word_embeddings is False
    assert config.pad_vocab_size_multiple == 64
    assert config.gradient_checkpointing is False
    assert config.cross_sample_token_id is None
    assert config.cp_size == 1
    assert config.tp_size == 1
    assert config.use_cache is True


def test_mixformer_sequential_config_custom():
    vocab_size = 50000
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    n_head_kv = 8
    rotary_dim = 64
    activation_function = "relu"
    embd_layer = "custom_embd_layer"
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    layer_norm_epsilon = 1e-6
    initializer_range = 0.01
    tie_word_embeddings = True
    pad_vocab_size_multiple = 128
    gradient_checkpointing = True
    cross_sample_token_id = 1
    cp_size = 2
    tp_size = 1
    use_cache = False

    config = MixFormerSequentialConfig(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_head_kv=n_head_kv,
        rotary_dim=rotary_dim,
        activation_function=activation_function,
        embd_layer=embd_layer,
        embd_pdrop=embd_pdrop,
        resid_pdrop=resid_pdrop,
        layer_norm_epsilon=layer_norm_epsilon,
        initializer_range=initializer_range,
        tie_word_embeddings=tie_word_embeddings,
        pad_vocab_size_multiple=pad_vocab_size_multiple,
        gradient_checkpointing=gradient_checkpointing,
        cross_sample_token_id=cross_sample_token_id,
        cp_size=cp_size,
        tp_size=tp_size,
        use_cache=use_cache,
    )

    adjusted_vocab_size = int(math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
    adjusted_rotary_dim = min(rotary_dim, n_embd // n_head)

    assert config.vocab_size == adjusted_vocab_size
    assert config.n_positions == n_positions
    assert config.n_embd == n_embd
    assert config.n_layer == n_layer
    assert config.n_head == n_head
    assert config.n_head_kv == n_head_kv
    assert config.rotary_dim == adjusted_rotary_dim
    assert config.activation_function == activation_function
    assert config.embd_layer == embd_layer
    assert config.embd_pdrop == embd_pdrop
    assert config.resid_pdrop == resid_pdrop
    assert config.layer_norm_epsilon == layer_norm_epsilon
    assert config.initializer_range == initializer_range
    assert config.tie_word_embeddings == tie_word_embeddings
    assert config.pad_vocab_size_multiple == pad_vocab_size_multiple
    assert config.gradient_checkpointing == gradient_checkpointing
    assert config.cross_sample_token_id == cross_sample_token_id
    assert config.cp_size == cp_size
    assert config.tp_size == tp_size
    assert config.use_cache == use_cache


@pytest.mark.is_torch_gpu
@pytest.mark.parametrize(
    ("model_cls, output_cls, extra_config, expected_shape_fn"),
    [
        pytest.param(
            MixFormerSequentialForCausalLM,
            CausalLMOutputWithPast,
            {},
            lambda bs, seq, cfg: (bs, seq, cfg.vocab_size),
            id="causal_lm",
        ),
        pytest.param(
            MixFormerSequentialForSequenceClassification,
            SequenceClassifierOutputWithPast,
            {"num_classes": 2, "problem_type": "single_label_classification", "pad_token_id": 1},
            lambda bs, seq, cfg: (bs, cfg.num_classes),
            id="sequence_classification",
        ),
    ],
)
def test_mixformer_sequential(model_cls, output_cls, extra_config, expected_shape_fn):
    batch_size = 2
    sequence_length = 5

    config = MixFormerSequentialConfig(**extra_config)
    model = model_cls(config).half().to("cuda")

    input_ids = torch.randint(config.vocab_size, (batch_size, sequence_length)).to("cuda")
    outputs = model(input_ids=input_ids)

    expected_shape = expected_shape_fn(batch_size, sequence_length, config)
    assert outputs.logits.shape == expected_shape
    assert isinstance(outputs, output_cls)
