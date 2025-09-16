# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.nn import functional as F

from phyagi.models.mixformer_sequential.blocks.norms.low_precision import LPLayerNorm
from phyagi.models.mixformer_sequential.blocks.norms.rms import RMSLayerNorm


def test_lp_layer_norm():
    lp_layer_norm = LPLayerNorm(normalized_shape=(768,), eps=1e-5, elementwise_affine=True)
    x = torch.randn(2, 3, 768)

    y = lp_layer_norm(x)
    expected_y = F.layer_norm(x, (768,))

    assert y.shape == (2, 3, 768)
    assert torch.allclose(y, expected_y, atol=1e-7)


def test_rms_layer_norm():
    rms_layer_norm = RMSLayerNorm(normalized_shape=(768,), eps=1e-5)
    x = torch.randn(2, 3, 768)

    y = rms_layer_norm(x)
    expected_y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)) * torch.ones(768)

    assert y.shape == (2, 3, 768)
    assert torch.allclose(y, expected_y, atol=1e-7)
