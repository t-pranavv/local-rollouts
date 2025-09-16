# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import queue
from logging import Handler, LogRecord, StreamHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


class StreamHandlerWithGPUMemory(StreamHandler):
    """Stream handler that supports appending GPU memory stats to log messages."""

    def __init__(self, stream: Optional[Any] = None, append_gpu_memory_stats: bool = False) -> None:
        """Initialize the handler.

        Args:
            stream: Stream to which log messages will be written.
            append_gpu_memory_stats: Whether to append GPU memory stats to the log messages.

        """

        super().__init__(stream=stream)

        self._append_gpu_memory_stats = append_gpu_memory_stats
        if not torch.cuda.is_available():
            self._append_gpu_memory_stats = False

    def _calculate_gpu_memory_stats(self, msg: str) -> str:
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = (total_mem - free_mem) / 1024**3
        used_fraction = round(1 - free_mem / total_mem, 2) * 100

        return f"{msg} [GPU memory allocated: {used_mem:.2f} GB ({used_fraction}% of device)]"

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        Args:
            record: Log record.

        """

        if isinstance(record.msg, str) and self._append_gpu_memory_stats:
            record.msg = self._calculate_gpu_memory_stats(record.msg)

        return super().emit(record)


class MlflowHandler(Handler):
    """MLflow logging handler."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        step_key: str = "train/step",
    ) -> None:
        """Initialize the handler.

        Args:
            run_id: Run identifier.
            experiment_id: Experiment identifier.
            run_name: Run name.
            nested: Whether new run should be nested to an active run.
            tags: Additional tags.
            description: Run description.
            step_key: Key for the step in the log record message.

        """

        super().__init__()

        try:
            mlflow.start_run(
                run_id=run_id,
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags,
                description=description,
            )
        except Exception:
            pass

        self._step_key = step_key

    def _parse(self, record: LogRecord) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        metrics = record.msg
        if isinstance(metrics, dict):
            step = metrics.get(self._step_key, None)
            return metrics, step

        return None, None

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        Args:
            record: Log record.

        """

        metrics, step = self._parse(record)
        if metrics is not None:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def end(self) -> None:
        """End the run."""

        mlflow.end_run()


class WandbHandler(Handler):
    """Weights & Biases logging handler."""

    def __init__(
        self,
        key: Optional[str] = None,
        host: Optional[str] = None,
        project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        step_key: str = "train/step",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        """Initialize the handler.

        Args:
            key: WandB API key.
                If not provided, it will attempt to read from the environment variable `WANDB_API_KEY`.
            host: Name of the host.
                If not provided, it will attempt to read from the environment variable `WANDB_HOST`.
            project: Name of the project.
            config: Configuration dictionary.
            step_key: Key for the step in the log record message.
            output_dir: Directory where output files will be stored (used to store the run identifier).

        """

        super().__init__()

        key = key or os.environ.get("WANDB_API_KEY", None)
        host = host or os.environ.get("WANDB_HOST", None)

        # Needed by the A100 cluster since it fails to connect to WandB's API
        os.environ["WANDB_DISABLE_SERVICE"] = "True"

        try:
            wandb.login(key=key, host=host, timeout=0)
        except Exception:
            pass

        # If `wandb_run_id` is available, use it to resume the run
        wandb_run_id = None
        if output_dir is not None and "id" not in kwargs and "resume" not in kwargs:
            wandb_run_id_path = Path(output_dir) / "wandb_run_id.txt"
            if wandb_run_id_path.exists():
                with open(wandb_run_id_path, "r") as f:
                    wandb_run_id = f.read().strip()

            # If it is not available, generate a new one
            if not wandb_run_id:
                wandb_run_id = wandb.util.generate_id()
                with open(wandb_run_id_path, "w") as f:
                    f.write(wandb_run_id)

            kwargs["id"] = wandb_run_id
            kwargs["resume"] = "allow"

        try:
            wandb.init(project=project, config=config, **kwargs)
        except Exception:
            pass

        self._step_key = step_key
        self._defined_metrics = set()

    def _parse(self, record: LogRecord) -> Tuple[Optional[Dict[str, Any]], Optional[Union[str, int]]]:
        metrics = record.msg
        if isinstance(metrics, dict):
            step = metrics.get(self._step_key, None)
            return metrics, step

        return None, None

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        Args:
            record: Log record.

        """

        metrics, _ = self._parse(record)
        if metrics is not None:
            for k in set(metrics.keys()).difference(self._defined_metrics):
                try:
                    wandb.define_metric(k, step_metric=self._step_key, summary="last")
                except Exception:
                    pass
                self._defined_metrics.add(k)

            try:
                wandb.log(metrics)
            except Exception:
                pass

    def end(self) -> None:
        """End the run."""

        wandb.finish()


class TensorBoardHandler(Handler):
    """TensorBoard logging handler."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = "runs", step_key: str = "train/step", **kwargs) -> None:
        """Initialize the handler.

        Args:
            log_dir: Directory where TensorBoard logs will be saved.
            step_key: Key for the step in the log record message.

        """

        super().__init__()

        self._log_dir = str(log_dir)
        self._step_key = step_key
        self._writer = SummaryWriter(log_dir=self._log_dir, **kwargs)

    def _parse(self, record: LogRecord) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        metrics = record.msg
        if isinstance(metrics, dict):
            step = metrics.get(self._step_key, None)
            return metrics, step

        return None, None

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        Args:
            record: Log record.

        """

        metrics, step = self._parse(record)
        if metrics is not None:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(key, value, step)
                    elif isinstance(value, (list, tuple)):
                        pass
            except Exception:
                pass

    def end(self) -> None:
        """End the run."""

        self._writer.close()


class QueueHandler(Handler):
    """Queue handler for non-blocking logging."""

    def __init__(self, queue: queue.Queue, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._queue = queue

    def emit(self, record: LogRecord) -> None:
        """Emit a record.

        Args:
            record: Log record.

        """

        self._queue.put_nowait(record)

    def end(self) -> None:
        """End the handler."""

        self.close()
