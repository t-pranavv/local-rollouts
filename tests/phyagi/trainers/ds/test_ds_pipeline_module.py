# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import deepspeed
import pytest
from deepspeed.pipe import LayerSpec
from torch.nn import Linear

from phyagi.trainers.ds.ds_pipeline_module import DsPipelineModule


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
def test_partition_layers():
    os.environ["LOCAL_RANK"] = "0"

    deepspeed.init_distributed()
    layers = [
        LayerSpec(Linear, 10, 20),
        LayerSpec(Linear, 20, 30),
        LayerSpec(Linear, 30, 40),
        LayerSpec(Linear, 40, 50),
        LayerSpec(Linear, 50, 60),
        LayerSpec(Linear, 60, 70),
        LayerSpec(Linear, 70, 80),
        LayerSpec(Linear, 80, 90),
    ]

    module = DsPipelineModule(layers, num_stages=1)
    module._partition_layers("uniform")
    assert module.parts == [0, 8]

    module = DsPipelineModule(layers, num_stages=1)
    module._partition_layers([0, 9])
    assert module.parts == [0, 9]

    module = DsPipelineModule(layers, num_stages=1)
    with pytest.raises(ValueError):
        module._partition_layers([0, 3, 6])
    with pytest.raises(ValueError):
        module._partition_layers([0, 3, 6, 9])
