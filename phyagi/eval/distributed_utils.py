# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List

import torch


def get_rank() -> int:
    """Get the rank of the current process.

    If the distributed environment is not initialized, the rank is 0.

    Returns:
        Rank of the current process.

    """

    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_world_size() -> int:
    """Get the world size.

    If the distributed environment is not initialized, the world size is 1.

    Returns:
        World size.

    """

    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def is_main_process() -> bool:
    """Check if the current process is the main process.

    If the distributed environment is not initialized, the current process is the main process.

    Returns:
        ``True`` if the current process is the main process, ``False`` otherwise.

    """

    return torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True


def is_main_local_process() -> bool:
    """Check if the current process is the main local process.

    If the distributed environment is not initialized, the current process is the main local process.

    Returns:
        ``True`` if the current process is the main local process, ``False`` otherwise.

    """

    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def all_reduce_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a dictionary of values across all processes.

    Args:
        input_dict: Dictionary of input values.

    Returns:
        Dictionary of reduced values.

    """

    input_dict = {key: torch.tensor(value) for key, value in input_dict.items()}

    for key, value in input_dict.items():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        input_dict[key] = value.item() / get_world_size()

    return input_dict


def all_gather_list(input_list: List[Any]) -> List[Any]:
    """Gather a list across all processes.

    Args:
        input_list: List of input values.

    Returns:
        List of gathered values.

    """

    gather_list = [None] * get_world_size()
    torch.distributed.all_gather_object(gather_list, input_list)

    return [item for sublist in gather_list for item in sublist]
