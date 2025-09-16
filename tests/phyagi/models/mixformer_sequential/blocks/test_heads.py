# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from phyagi.models.mixformer_sequential.blocks.heads.causal_lm import (
    CausalLMHead,
    CausalLMLoss,
)
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)


def test_causal_lm_head():
    config = MixFormerSequentialConfig()
    causal_lm_head = CausalLMHead(config)
    hidden_states = torch.randn(2, 3, 1024)

    logits = causal_lm_head(hidden_states)
    assert logits.shape == (2, 3, 50304)
    assert torch.all(logits == logits.to(torch.float32))


def test_causal_lm_loss():
    causal_lm_loss = CausalLMLoss()
    logits = torch.randn(2, 3, 50304)
    labels = torch.randint(0, 50304, (2, 3))

    loss = causal_lm_loss(logits, labels)
    assert loss.shape == torch.Size([])
    assert torch.all(loss == loss.to(torch.float32))
