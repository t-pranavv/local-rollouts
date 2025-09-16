# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import tempfile
import types
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

from phyagi.utils.checkpoint import (
    CheckpointManager,
    convert_ds_zero_checkpoint,
    convert_pl_fsdp_checkpoint,
    convert_ray_actor_checkpoint,
)


def test_checkpoint_manager(monkeypatch):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "ckpt_sync"

        with mock.patch("phyagi.utils.checkpoint.dcp.save") as mock_save, mock.patch(
            "phyagi.utils.checkpoint.dcp.load"
        ) as mock_load:
            checkpoint_manager = CheckpointManager(mode="sync")
            checkpoint_manager.save(ckpt_path, model, optimizer, overwrite=True)

            mock_save.assert_called_once()
            _, save_kwargs = mock_save.call_args
            assert save_kwargs["checkpoint_id"] == str(ckpt_path)

            checkpoint_manager.load(ckpt_path, model, optimizer)
            mock_load.assert_called_once()
            _, load_kwargs = mock_load.call_args
            assert load_kwargs["checkpoint_id"] == str(ckpt_path)

        fake_dist = types.SimpleNamespace()
        fake_dist.is_initialized = lambda: True
        fake_dist.new_group = lambda backend=None: "fake_pg"
        monkeypatch.setattr(torch, "distributed", fake_dist, raising=False)

        dummy_future = mock.Mock()
        dummy_future.result.return_value = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "ckpt_async"

            with mock.patch("phyagi.utils.checkpoint.dcp.async_save", return_value=dummy_future) as mock_async_save:
                checkpoint_manager = CheckpointManager(mode="async")

                checkpoint_manager.save(ckpt_path, model, optimizer)
                mock_async_save.assert_called_once()
                assert checkpoint_manager._async_future is dummy_future

                checkpoint_manager._wait_for_previous()
                dummy_future.result.assert_called_once()
                assert checkpoint_manager._async_future is None


@pytest.mark.parametrize(
    ("convert_fn, input_config_name"),
    [
        pytest.param(convert_ds_zero_checkpoint, "config.json", id="convert_ds_zero_checkpoint"),
        pytest.param(convert_pl_fsdp_checkpoint, "config.json", id="convert_pl_fsdp_checkpoint"),
        pytest.param(convert_ray_actor_checkpoint, "actor_config.json", id="convert_ray_actor_checkpoint"),
    ],
)
def test_checkpoint_converters(convert_fn, input_config_name):
    with tempfile.TemporaryDirectory() as tmp_dir, tempfile.TemporaryDirectory() as output_dir:
        config_path = Path(tmp_dir) / input_config_name
        with open(config_path, "w") as f:
            json.dump({"mock": "config"}, f)

        dummy_state_dict = OrderedDict({"layer.weight": torch.randn(2, 2), "layer.bias": torch.randn(2)})
        dummy_checkpoint = {
            "model": dummy_state_dict,
            "optimizer_0": {"param_groups": []},
        }

        with mock.patch(
            "phyagi.utils.checkpoint._get_fp32_state_dict_from_zero_checkpoint", return_value=dummy_state_dict
        ), mock.patch("phyagi.utils.checkpoint._load_distributed_checkpoint", return_value=dummy_checkpoint):
            convert_fn(tmp_dir, output_dir)

        assert (Path(output_dir) / "config.json").exists()
        assert any(f.endswith(".safetensors") for f in os.listdir(output_dir))
