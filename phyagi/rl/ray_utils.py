# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from logging import Logger
from typing import Any, Callable, List

import numpy as np
import ray
import wandb

from phyagi.utils.logging_handlers import WandbHandler
from phyagi.utils.logging_utils import get_logger


def chunk_equal(array: List[Any], splits: int) -> List[List[Any]]:
    """Chunk a list into a specified number of approximately equal-sized chunks.

    Args:
        array: List to be chunked.
        splits: Number of chunks to split the list into.

    Returns:
        List of lists, where each sublist is a chunk of the original list.

    """

    return [chunk.tolist() for chunk in np.array_split(np.array(array, dtype=object), splits)]


def get_ray_logger(name: str, **wandb_kwargs) -> Logger:
    """Get a logger for Ray with optional WandB integration.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.

    """

    logger = get_logger(name, rank_filter=False, append_gpu_memory_stats=True)
    if wandb_kwargs:
        logger.addHandler(WandbHandler(**wandb_kwargs))

    return logger


def ray_wandb_graceful_shutdown(f: Callable) -> Callable:
    """Gracefully shutdown WandB when using Ray.

    This decorator properly handles the shutdown despite any exceptions raised during the function execution
    or when the Ray actor is killed.

    Args:
        f: Function to decorate.

    Returns:
        Decorated function.

    """

    def _ray_wandb_graceful_shutdown(*args, **kwargs) -> Callable:
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if wandb.run is not None:
                wandb.run.finish(exit_code=1)
            ray.shutdown()

    return _ray_wandb_graceful_shutdown
