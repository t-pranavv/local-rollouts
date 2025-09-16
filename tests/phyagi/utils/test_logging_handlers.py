# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import queue
import shutil
import tempfile
from io import StringIO
from logging import LogRecord
from unittest.mock import MagicMock, call, patch

from phyagi.utils.logging_handlers import (
    MlflowHandler,
    QueueHandler,
    StreamHandlerWithGPUMemory,
    TensorBoardHandler,
    WandbHandler,
)


def test_stream_handler_with_gpu_memory():
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)
    ):

        stream = StringIO()
        handler = StreamHandlerWithGPUMemory(stream=stream, append_gpu_memory_stats=True)

        logger = logging.getLogger("test_gpu_logger")
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]

        logger.info("Training step complete")

        output = stream.getvalue()
        assert "Training step complete" in output
        assert "GPU memory allocated: 4.00 GB (50.0% of device)" in output

    with patch("torch.cuda.is_available", return_value=False):
        stream = StringIO()
        handler = StreamHandlerWithGPUMemory(stream=stream, append_gpu_memory_stats=True)

        logger = logging.getLogger("test_no_gpu_logger")
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]

        logger.info("No GPU info expected")

        output = stream.getvalue()
        assert "No GPU info expected" in output
        assert "GPU memory allocated" not in output


def test_mlflow_handler():
    with patch("mlflow.start_run"), patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.end_run"
    ) as mock_end_run:
        handler = MlflowHandler()

        log_record = LogRecord(
            name="test", level=1, pathname="", lineno=0, msg={"train/step": 1, "loss": 0.5}, args=None, exc_info=None
        )
        handler.emit(log_record)
        mock_log_metrics.assert_called_with({"train/step": 1, "loss": 0.5}, step=1)

        handler.end()
        mock_end_run.assert_called_once()


def test_wandb_handler():
    with patch("wandb.login"), patch("wandb.init"), patch("wandb.define_metric") as mock_define_metric, patch(
        "wandb.log"
    ) as mock_log, patch("wandb.finish") as mock_finish:
        handler = WandbHandler(step_key="step")

        log_record = LogRecord(
            name="test", level=1, pathname="", lineno=0, msg={"loss": 0.5, "step": 1}, args=None, exc_info=None
        )
        handler.emit(log_record)
        mock_log.assert_called_with({"loss": 0.5, "step": 1})
        mock_define_metric.assert_has_calls(
            calls=[call("step", step_metric="step", summary="last"), call("loss", step_metric="step", summary="last")],
            any_order=True,
        )

        mock_define_metric.reset_mock()
        handler.emit(log_record)
        mock_define_metric.assert_not_called()

        handler.end()
        mock_finish.assert_called_once()


def test_tensorboard_handler():
    temp_dir = tempfile.mkdtemp()

    try:
        handler = TensorBoardHandler(log_dir=temp_dir, step_key="train/step")
        log_record = LogRecord(
            name="test",
            level=1,
            pathname="",
            lineno=0,
            msg={"train/step": 10, "loss": 0.42, "accuracy": 0.97},
            args=None,
            exc_info=None,
        )

        handler.emit(log_record)
        handler.end()

        event_files = [f for f in os.listdir(temp_dir) if f.startswith("events.out.tfevents")]
        assert len(event_files) > 0

    finally:
        shutil.rmtree(temp_dir)


def test_queue_handler():
    log_queue = queue.Queue()
    mock_record = MagicMock(spec=LogRecord)

    handler = QueueHandler(log_queue)
    handler.emit(mock_record)

    assert not log_queue.empty()
    queued_record = log_queue.get_nowait()
    assert queued_record is mock_record
