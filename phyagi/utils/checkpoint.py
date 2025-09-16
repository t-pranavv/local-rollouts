# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gc
import re
import shutil
from argparse import Namespace
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from deepspeed.checkpoint.ds_to_universal import (
    main as _save_universal_from_ds_checkpoint,
)
from deepspeed.utils.zero_to_fp32 import _get_fp32_state_dict_from_zero_checkpoint
from huggingface_hub import split_torch_state_dict_into_shards
from lightning.fabric.utilities.load import _load_distributed_checkpoint
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from phyagi.utils.config import load_config
from phyagi.utils.file_utils import save_json_file


class _ModelState(Stateful):
    def __init__(self, model: nn.Module, full_state_dict: bool = False, cpu_offload: bool = True) -> None:
        if model is None:
            raise ValueError("`model` must be initialized before saving/loading the state.")

        self._model = model
        self._full_state_dict = full_state_dict
        self._cpu_offload = cpu_offload

    def state_dict(self) -> Dict[str, Any]:
        return get_model_state_dict(
            self._model, options=StateDictOptions(full_state_dict=self._full_state_dict, cpu_offload=self._cpu_offload)
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(self._model, model_state_dict=state_dict)


class _OptimizerState(Stateful):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        full_state_dict: bool = False,
        cpu_offload: bool = True,
    ) -> None:
        if model is None:
            raise ValueError("`model` must be initialized before saving/loading the state.")
        if optimizer is None:
            raise ValueError("`optimizer` must be initialized before saving/loading the state.")

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._full_state_dict = full_state_dict
        self._cpu_offload = cpu_offload

    def state_dict(self) -> Dict[str, Any]:
        optimizer_state_dict = get_optimizer_state_dict(
            self._model,
            self._optimizer,
            options=StateDictOptions(full_state_dict=self._full_state_dict, cpu_offload=self._cpu_offload),
        )

        state_dict = {"optimizer": optimizer_state_dict}
        if self._scheduler is not None:
            state_dict["scheduler"] = self._scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(self._model, self._optimizer, optim_state_dict=state_dict["optimizer"])
        if self._scheduler is not None and "scheduler" in state_dict:
            self._scheduler.load_state_dict(state_dict["scheduler"])


def _get_fp32_state_dict_from_universal_checkpoint(checkpoint_dir: Path) -> OrderedDict:
    model_states = torch.load(
        checkpoint_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt", map_location="cpu", weights_only=False
    )
    all_param_shapes = model_states["param_shapes"]

    model_state_dict = OrderedDict()
    for param_shapes in all_param_shapes:
        for param_name, param_shape in param_shapes.items():
            param_folder = checkpoint_dir / "zero" / param_name
            if not param_folder.is_dir():
                raise FileNotFoundError(f"'{param_folder}' must be a directory.")

            param_file = param_folder / "fp32.pt"
            if not param_file.is_file():
                raise FileNotFoundError(f"'{param_file}' must be a file.")

            flat_tensor = torch.load(param_file, map_location="cpu")
            reshaped_tensor = flat_tensor.view(param_shape)

            model_state_dict[param_name] = reshaped_tensor

    return model_state_dict


def _save_state_dict_with_safetensors(state_dict: OrderedDict, output_dir: Path, max_shard_size: str = "5GB") -> None:
    def _to_torch_tensor(state_dict: OrderedDict, return_empty_tensor: bool = False) -> OrderedDict:
        torch_state_dict = {}
        converted_tensors = {}

        for name, tensor in state_dict.items():
            tensor_id = id(tensor)
            if tensor_id in converted_tensors:
                shared_tensor = torch_state_dict[converted_tensors[tensor_id]]
                torch_state_dict[name] = shared_tensor.clone().detach().contiguous()
            else:
                converted_tensors[tensor_id] = name
                if return_empty_tensor:
                    torch_state_dict[name] = torch.empty(tensor.shape, dtype=tensor.dtype)
                else:
                    torch_state_dict[name] = tensor.contiguous()

        return torch_state_dict

    # Shard the state dictionary if necessary
    weights_name = "model.safetensors"
    if max_shard_size is not None:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        empty_state_dict = _to_torch_tensor(state_dict, return_empty_tensor=True)
        state_dict_split = split_torch_state_dict_into_shards(
            empty_state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
    else:
        StateDictSplit = namedtuple("StateDictSplit", ["is_sharded", "filename_to_tensors"])
        state_dict_split = StateDictSplit(is_sharded=False, filename_to_tensors={weights_name: list(state_dict.keys())})

    # Save the state dictionary shards
    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in tqdm(filename_to_tensors):
        shard_state_dict = {t: state_dict[t] for t in tensors}
        shard_state_dict = _to_torch_tensor(shard_state_dict)

        save_file(shard_state_dict, output_dir / shard_file, metadata={"format": "pt"})

        # Clean up the memory of current shard
        for t in list(shard_state_dict.keys()):
            del state_dict[t]
            del shard_state_dict[t]
        del shard_state_dict

        gc.collect()

    # If the state dictionary has been sharded, save the index file
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

        index_file_path = output_dir / "model.safetensors.index.json"
        save_json_file(index, index_file_path, indent=2, sort_keys=True)


class CheckpointManager:
    """Checkpoint manager for saving and loading checkpoints."""

    def __init__(self, mode: str = "sync") -> None:
        """Initialize the checkpoint manager.

        Args:
            mode: Mode of the checkpoint manager, either ``sync`` or ``async``.

        """

        if mode not in {"sync", "async"}:
            raise ValueError(f"`mode` must be 'sync' or 'async', but got '{mode}'.")

        self.mode = mode

        self._pg = None
        self._async_future = None

    def _wait_for_previous(self) -> None:
        if self._async_future is not None:
            self._async_future.result()
            self._async_future = None

    def save(
        self,
        checkpoint_id: Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        overwrite: bool = True,
        full_state_dict: bool = False,
        cpu_offload: bool = True,
    ) -> None:
        """Save a checkpoint.

        Args:
            checkpoint_id: Path to the checkpoint.
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Learning rate scheduler to save.
            overwrite: Whether to overwrite the checkpoint if it already exists.
            full_state_dict: Whether to save the full state dictionary.
            cpu_offload: Whether to offload the model state to CPU.

        """

        self._wait_for_previous()

        state = {"model": _ModelState(model, full_state_dict=full_state_dict, cpu_offload=cpu_offload)}
        if optimizer is not None:
            state["optimizer"] = _OptimizerState(
                model, optimizer, scheduler, full_state_dict=full_state_dict, cpu_offload=cpu_offload
            )

        if self.mode == "sync":
            writer = FileSystemWriter(str(checkpoint_id), overwrite=overwrite)
            dcp.save(state, storage_writer=writer, checkpoint_id=str(checkpoint_id))

            return

        if self.mode == "async":
            if torch.distributed.is_initialized() and self._pg is None:
                self._pg = torch.distributed.new_group(backend="gloo")

            writer = FileSystemWriter(str(checkpoint_id), overwrite=overwrite)
            self._async_future = dcp.async_save(
                state,
                storage_writer=writer,
                checkpoint_id=str(checkpoint_id),
                process_group=self._pg,
            )

            return

    def load(
        self,
        checkpoint_id: Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        full_state_dict: bool = False,
        cpu_offload: bool = True,
    ) -> None:
        """Load a distributed checkpoint.

        Args:
            checkpoint_id: Path to the checkpoint.
            model: Model to load the state into.
            optimizer: Optimizer to load the state into.
            scheduler: Learning rate scheduler to load the state into.
            full_state_dict: Whether to load the full state dictionary.
            cpu_offload: Whether to offload the model state to CPU.

        """

        state = {"model": _ModelState(model, full_state_dict=full_state_dict, cpu_offload=cpu_offload)}
        if optimizer is not None:
            state["optimizer"] = _OptimizerState(
                model, optimizer, scheduler, full_state_dict=full_state_dict, cpu_offload=cpu_offload
            )

        dcp.load(state, checkpoint_id=str(checkpoint_id))
        gc.collect()


def convert_ds_zero_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_shard_size: str = "5GB",
    exclude_frozen_parameters: bool = False,
    use_universal_checkpoint: bool = False,
) -> None:
    """Convert a DeepSpeed ZeRO-2/3 checkpoint to a safe serialized checkpoint.

    The converted checkpoint is based on ``*.safetensors`` files and can be loaded
    directly with ``AutoModel.from_pretrained()``.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        output_dir: Path to the output directory.
        max_shard_size: Maximum shard size for the converted checkpoint.
        exclude_frozen_parameters: Whether to exclude frozen parameters from the converted checkpoint.
        use_universal_checkpoint: Whether to use a universal-based checkpoint conversion.
            This is a hard-requirement if the checkpoint is split across multiple ranks, e.g., tensor parallelism.

    """

    checkpoint_dir = Path(checkpoint_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check whether configuration file is available
    config_file_path = checkpoint_dir / "config.json"
    if not config_file_path.exists():
        raise FileNotFoundError(f"'config.json' must be available in '{checkpoint_dir}'.")
    shutil.copy(config_file_path, output_dir / "config.json")

    if use_universal_checkpoint:
        # To prevent any failure when converting the checkpoint to a universal checkpoint,
        # we import the script directly from DeepSpeed and pass the arguments to it
        output_folder = output_dir / "universal_checkpoint"
        args = Namespace(
            input_folder=str(checkpoint_dir),
            output_folder=str(output_folder),
            num_extract_workers=4,
            num_merge_workers=2,
            keep_temp_folder=False,
            strict=True,
            inject_missing_state=False,
        )
        _save_universal_from_ds_checkpoint(args)

        # Since universal checkpoints still keep the model parallelism,
        # we need to convert it to a state dictionary
        state_dict = _get_fp32_state_dict_from_universal_checkpoint(output_folder)
    else:
        # Convert the checkpoint to a state dictionary
        state_dict = _get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, exclude_frozen_parameters)

    # Save the state dictionary to a safe serialized checkpoint
    _save_state_dict_with_safetensors(state_dict, output_dir, max_shard_size)


def convert_pl_fsdp_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_shard_size: str = "5GB",
    dump_unused_keys: bool = False,
) -> None:
    """Convert a PyTorch Lightning FSDP checkpoint to a safe serialized checkpoint.

    The converted checkpoint is based on ``*.safetensors`` files and can be loaded
    directly with ``AutoModel.from_pretrained()``.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        output_dir: Path to the output directory.
        max_shard_size: Maximum shard size for the converted checkpoint.
        dump_unused_keys: Whether to dump unused keys in the checkpoint.

    """

    def _format_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = checkpoint.pop("model", None)
        if state_dict is not None:
            checkpoint["state_dict"] = state_dict

        optimizer_keys = [key for key in checkpoint if re.match("optimizer_[0-9]+", key)]
        if not optimizer_keys:
            return checkpoint

        checkpoint["optimizer_states"] = [
            checkpoint.pop(f"optimizer_{opt_idx}") for opt_idx in range(len(optimizer_keys))
        ]

        # Since the model is wrapped by a `model` attribute in `PlLightningModule`,
        # we need to remove the `model` prefix
        checkpoint["state_dict"] = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

        return checkpoint

    checkpoint_dir = Path(checkpoint_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check whether configuration file is available
    config_file_path = checkpoint_dir / "config.json"
    if not config_file_path.exists():
        raise FileNotFoundError(f"'config.json' must be available in '{checkpoint_dir}'.")
    shutil.copy(config_file_path, output_dir / "config.json")

    checkpoint = _load_distributed_checkpoint(checkpoint_dir)
    checkpoint = _format_checkpoint(checkpoint)

    state_dict = checkpoint.pop("state_dict")
    _save_state_dict_with_safetensors(state_dict, output_dir, max_shard_size)

    if dump_unused_keys:
        torch.save(checkpoint, output_dir / "unused_keys.pt")


def convert_ray_actor_checkpoint(
    checkpoint_dir: Union[str, Path], output_dir: Union[str, Path], max_shard_size: str = "5GB"
) -> None:
    """Convert a Ray actor checkpoint to a safe serialized checkpoint.

    The converted checkpoint is based on ``*.safetensors`` files and can be loaded
    directly with ``AutoModel.from_pretrained()``.

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        output_dir: Path to the output directory.
        max_shard_size: Maximum shard size for the converted checkpoint.

    """

    checkpoint_dir = Path(checkpoint_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check whether configuration file is available
    actor_config_file_path = checkpoint_dir / "actor_config.json"
    if not actor_config_file_path.exists():
        raise FileNotFoundError(f"'actor_config.json' file is not available in '{checkpoint_dir}'.")

    model_config = load_config(actor_config_file_path).get("model", {})
    save_json_file(model_config, output_dir / "config.json")

    # Convert the checkpoint to a state dictionary and save it
    model_state_dict = _load_distributed_checkpoint(checkpoint_dir)["model"]

    # Save the state dictionary shards
    _save_state_dict_with_safetensors(model_state_dict, output_dir, max_shard_size)
