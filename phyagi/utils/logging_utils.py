# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import sys
import warnings
from logging import Filter, Formatter, Logger, LogRecord
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from phyagi.utils.logging_handlers import StreamHandlerWithGPUMemory

FORMATTER = Formatter("[phyagi] [%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
LOG_LEVEL = logging.DEBUG
LOCAL_RANK = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or os.environ.get("LOCAL_RANK", 0))

# Suppress specific information messages and warnings from installed packages
logging.getLogger("absl").setLevel(logging.WARNING)
logging.getLogger("DeepSpeed").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"flash_attn\.ops\.fused_dense")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.autograd\.function")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.autograd\.graph")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.distributed\.checkpoint\.filesystem")


class _RankFilter(Filter):
    def __init__(self, rank: int) -> None:
        self._rank = rank

    def filter(self, record: LogRecord) -> bool:
        return self._rank == 0


def _get_console_handler(append_gpu_memory_stats: bool = False) -> StreamHandlerWithGPUMemory:
    console_handler = StreamHandlerWithGPUMemory(stream=sys.stdout, append_gpu_memory_stats=append_gpu_memory_stats)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def _get_timed_file_handler(file_path: Optional[str] = None) -> TimedRotatingFileHandler:
    file_path = file_path or "phyagi.log"

    file_handler = TimedRotatingFileHandler(file_path, delay=True, when="midnight", encoding="utf-8")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(
    logger_name: str, file_path: Optional[str] = None, rank_filter: bool = True, append_gpu_memory_stats: bool = False
) -> Logger:
    """Get a logger with the specified name and default settings.

    Args:
        logger_name: Name of the logger.
        file_path: Path to the file to which log messages will be written.
        rank_filter: Whether to apply a filter that prints only to the GPU with ``local_rank=0``.
        append_gpu_memory_stats: Whether to append GPU memory stats to the log messages.

    Returns:
        :class:`logging.Logger` instance with the specified name and default settings.

    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)

    logger.addHandler(_get_console_handler(append_gpu_memory_stats=append_gpu_memory_stats))
    logger.addHandler(_get_timed_file_handler(file_path=file_path))

    if rank_filter:
        logger.addFilter(_RankFilter(LOCAL_RANK))

    logger.propagate = False

    return logger
