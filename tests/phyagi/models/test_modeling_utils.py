# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path

import torch
from transformers import BertConfig

from phyagi.models.modeling_utils import PreTrainedModel


def test_pretrained_model_from_pretrained():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = BertConfig(vocab_size=1000, hidden_size=32, num_hidden_layers=2)
        model = PreTrainedModel(config)
        torch.save({"module": model.state_dict(), "config": config}, Path(tmp_dir) / "mp_rank_00_model_states.pt")

        loaded_model = torch.load(Path(tmp_dir) / "mp_rank_00_model_states.pt", weights_only=False)
        assert model.state_dict() == loaded_model["module"]
        assert set(model.config.to_dict().keys()) == set(loaded_model["config"].to_dict().keys())
