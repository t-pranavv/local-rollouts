# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.models.mixformer_sequential.blocks.embeddings.common import (
    Embedding,
    PositionalEmbedding,
)
from phyagi.models.mixformer_sequential.blocks.embeddings.rotary import (
    RotaryEmbedding,
    YarnEmbedding,
)
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)


def test_embedding():
    config = MixFormerSequentialConfig()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    emb = Embedding(config)

    output = emb(input_ids)
    assert output.shape == (2, 3, 1024)


def test_positional_embedding():
    config = MixFormerSequentialConfig()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
    emb = PositionalEmbedding(config)

    output = emb(input_ids, position_ids)
    assert output.shape == (2, 3, 1024)


def test_rotary_embedding():
    rotary_embedding = RotaryEmbedding(dim=16, flash_rotary=False)

    q = torch.randn(2, 4, 8, 16).to(dtype=torch.float16)
    kv = torch.randn(2, 4, 2, 4, 16).to(dtype=torch.float16)

    q, kv = rotary_embedding(q, kv)
    assert q.shape == (2, 4, 8, 16)
    assert kv.shape == (2, 4, 2, 4, 16)


@pytest.mark.is_flash_attn
@pytest.mark.is_torch_gpu
def test_rotary_embedding_flash():
    rotary_embedding = RotaryEmbedding(dim=16)

    q = torch.randn(2, 4, 8, 16).to(dtype=torch.float16, device="cuda")
    kv = torch.randn(2, 4, 2, 4, 16).to(dtype=torch.float16, device="cuda")

    q, kv = rotary_embedding(q, kv)
    assert q.shape == (2, 4, 8, 16)
    assert kv.shape == (2, 4, 2, 4, 16)


def test_yarn_embedding():
    yarn_embedding = YarnEmbedding(dim=16, flash_rotary=False)

    q = torch.randn(2, 4, 8, 16).to(dtype=torch.float16)
    kv = torch.randn(2, 4, 2, 4, 16).to(dtype=torch.float16)

    q, kv = yarn_embedding(q, kv)
    assert q.shape == (2, 4, 8, 16)
    assert kv.shape == (2, 4, 2, 4, 16)


@pytest.mark.is_flash_attn
@pytest.mark.is_torch_gpu
def test_yarn_embedding_flash():
    yarn_embedding = YarnEmbedding(dim=16)

    q = torch.randn(2, 4, 8, 16).to(dtype=torch.float16, device="cuda")
    kv = torch.randn(2, 4, 2, 4, 16).to(dtype=torch.float16, device="cuda")

    q, kv = yarn_embedding(q, kv)
    assert q.shape == (2, 4, 8, 16)
    assert kv.shape == (2, 4, 2, 4, 16)
