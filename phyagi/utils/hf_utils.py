# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


class AzureStorageRotateCheckpointMixin:
    """Mixin to rotate checkpoints and cache them to Azure Storage."""

    def _rotate_checkpoints(self, use_mtime: bool = False, output_dir: Optional[str] = None) -> None:
        """Rotate checkpoints and cache them to Azure Storage.

        The ``use_mtime`` argument is always set to ``False`` to avoid having
        multiple checkpoints with the same timestamp when retrieving them
        from Azure Storage. This is because Azure Storage does not support
        sub-second precision for file timestamps.

        Args:
            use_mtime: Whether to use mtime to sort the checkpoints.
            output_dir: Output directory where checkpoints are rotated.

        """

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Enforce `use_mtime = False` to avoid identical timestamps
        # when retrieving files from Azure Storage
        use_mtime = False

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If `save_total_limit=1` with `load_best_model_at_end=True`, we could delete the
        # last checkpoint, which we do not do to allow resuming
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint, ignore_errors=True)


def to_device_map(
    model: torch.nn.Module,
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, Path]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
) -> torch.nn.Module:
    """Move a model to a given ``device_map``.

    This function allows the layers of the model to be spread across multiple devices,
    offloaded on the CPU or even the disk.

    Args:
        model: Model to move.
        device_map: Mapping of layers to devices.
            If ``auto``, the layers will be automatically assigned to devices based on the memory constraints.
        max_memory: Maximum memory to use for each device.
            If ``None``, the memory will be automatically assigned based on the model's parameters.
        no_split_module_classes: List of module classes that should not be split across devices.
        offload_folder: Folder to use for offloading layers on the disk.
        offload_buffers: Whether to offload the buffers of the layers on the disk.
        dtype: Data type to use for the layers.
            If ``None``, the data type of the model will be used.
        skip_keys: Keys of the model state dict to skip when moving the model.
        preload_module_classes: List of module classes to preload on the CPU.

    Returns:
        Model with the layers moved to the appropriate devices.

    """

    if device_map is None:
        return model

    AVAILABLE_DEVICE_MAPS = ["auto", "balanced", "balanced_low_0", "cpu", "cuda", "sequential"]
    if not isinstance(device_map, str) or device_map not in AVAILABLE_DEVICE_MAPS:
        raise ValueError(f"`device_map` should be one of {AVAILABLE_DEVICE_MAPS}, but got '{device_map}'.")

    if device_map == "cpu":
        device_map = {"": "cpu"}
    elif device_map == "cuda":
        device_map = {"": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}
    else:
        no_split_module_classes = no_split_module_classes or getattr(model, "_no_split_modules", [])
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=dtype
        )

    skip_keys = skip_keys or getattr(model, "_skip_keys_device_placement", [])
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        skip_keys=skip_keys,
        preload_module_classes=preload_module_classes,
    )
