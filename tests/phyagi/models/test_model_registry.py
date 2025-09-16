# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerBase

from phyagi.models.registry import get_model, get_tokenizer


def test_get_model():
    pretrained_model_name = "gpt2"
    model_from_pretrained = get_model(pretrained_model_name_or_path=pretrained_model_name)
    assert isinstance(model_from_pretrained, torch.nn.Module)
    assert isinstance(model_from_pretrained, GPT2LMHeadModel)

    kwargs = {
        "model_type": "gpt2",
        "vocab_size": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
    }
    model_from_config = get_model(pretrained_model_name_or_path=None, **kwargs)
    assert isinstance(model_from_config, torch.nn.Module)
    assert isinstance(model_from_config, GPT2LMHeadModel)
    for key, value in kwargs.items():
        assert getattr(model_from_config.config, key) == value


def test_get_tokenizer():
    pretrained_tokenizer_name = "gpt2"
    additional_special_tokens = ["[SPECIAL1]", "[SPECIAL2]"]

    tokenizer = get_tokenizer(pretrained_tokenizer_name, additional_special_tokens=additional_special_tokens)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    added_tokens = tokenizer.get_added_vocab()
    for special_token in additional_special_tokens:
        assert special_token in added_tokens
