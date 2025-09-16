# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
from transformers import PreTrainedModel as HfPreTrainedModel

from phyagi.utils.hf_utils import to_device_map
from phyagi.utils.logging_utils import get_logger
from phyagi.version import __version__

logger = get_logger(__name__)


def _is_ds_checkpoint(checkpoint_dir: Union[str, Path]) -> bool:
    checkpoint_dir = Path(checkpoint_dir)

    zero_pp_checkpoint = checkpoint_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    if zero_pp_checkpoint.exists():
        raise ValueError("ZeRO-2/3 has been detected. Run `convert_ds_zero_checkpoint()` before loading the model.")

    checkpoint = checkpoint_dir / "mp_rank_00_model_states.pt"
    return checkpoint.exists()


def _load_ds_state_dict(state_dict_dir: Union[str, Path], dump_unused_keys: bool = False) -> OrderedDict:
    def _count_files(path: Union[str, Path], pattern: str) -> int:
        file_regex = re.compile(pattern)
        return len([f for f in os.listdir(path) if file_regex.match(f)])

    state_dict_dir = Path(state_dict_dir)
    full_state_dict = torch.load(state_dict_dir / "mp_rank_00_model_states.pt", map_location="cpu", weights_only=False)

    n_layer_files = _count_files(state_dict_dir, r"layer_\d+-model_states.pt")
    n_rank_files = _count_files(state_dict_dir, r"mp_rank_\d+_model_states.pt")

    if n_rank_files == 1:
        # If no model parallelism is available, we load the initial states from the first rank file
        model_state_dict = full_state_dict.pop("module", full_state_dict)
    else:
        # If model parallelism is available, we load the states from all ranks and produce a
        # single state dictionary
        for i in range(1, n_rank_files):
            rank_state_dict = torch.load(
                state_dict_dir / f"mp_rank_{i:02d}_model_states.pt", map_location="cpu", weights_only=False
            )
            for k, v in rank_state_dict.items():
                # For each variable type, we handle the update differently
                full_state_dict_value = full_state_dict.get(k, None)
                if full_state_dict_value is None:
                    continue
                if isinstance(full_state_dict_value, list):
                    full_state_dict[k].extend(v)
                if isinstance(full_state_dict_value, (dict, set)):
                    full_state_dict[k].update(v)

        model_state_dict = full_state_dict.pop("module", full_state_dict)

    if model_state_dict is None and n_layer_files == 0:
        raise ValueError("`module` must be in the state dictionary if no layer files are found.")

    if model_state_dict is None:
        model_state_dict = {}
        for i in range(n_layer_files):
            layer_dict = torch.load(
                state_dict_dir / f"layer_{i:02d}-model_states.pt", map_location="cpu", weights_only=False
            )
            for k, v in layer_dict.items():
                model_state_dict[f"layers.{i}.{k}"] = v

    if dump_unused_keys:
        torch.save(full_state_dict, state_dict_dir / "unused_keys.pt")

    return model_state_dict


class PreTrainedModel(HfPreTrainedModel):
    """Pre-trained model that supports loading DeepSpeed checkpoints."""

    @classmethod
    def from_pretrained(
        cls: PreTrainedModel, pretrained_model_name_or_path: Union[str, Path], *args, **kwargs
    ) -> PreTrainedModel:
        ds_device_map = None

        if _is_ds_checkpoint(pretrained_model_name_or_path):
            # When passing `state_dict`, we need to disable `low_cpu_mem_usage`
            kwargs["low_cpu_mem_usage"] = False
            kwargs["state_dict"] = _load_ds_state_dict(pretrained_model_name_or_path)

            # When using `from_pretrained()` outside of `AutoModel` classes, we need to force
            # loading the configuration to avoid having unused keyword arguments
            if kwargs.get("config", None) is None:
                kwargs["return_unused_kwargs"] = True
                config, unused_kwargs = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

                # `unused_kwargs` and `config` should be inputted back as `kwargs`
                kwargs = {**unused_kwargs, "config": config}

            # Since `low_cpu_mem_usage` is disabled, the model will be loaded on the CPU,
            # and we will need to move it to the correct device afterwards
            ds_device_map = kwargs.pop("device_map", None)

            # `pretrained_model_name_or_path` should be None since we are passing `state_dict`
            pretrained_model_name_or_path = None

        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Override the `phyagi_version` since the model is loaded in a different environment
        if hasattr(model.config, "phyagi_version"):
            model.config.phyagi_version = __version__

        # If the model has been loaded from DeepSpeed, we need to ensure that it is placed
        # on the correct device
        if ds_device_map is not None:
            model = to_device_map(model, ds_device_map)

        return model
