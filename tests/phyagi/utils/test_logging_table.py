# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import time
from unittest import mock

import pytest

from phyagi.utils.logging_table import LocalTableLogger, TableLogger, WandbTableLogger


def test_table_logger():
    class DummyTableLogger(TableLogger):
        pass

    logger = DummyTableLogger(name="test", columns=["col1", "col2"])
    with pytest.raises(NotImplementedError):
        logger.add_data([[1, 2]])


def test_local_table_logger(tmp_path):
    columns = ["a", "b"]
    path = tmp_path / "logfile"

    logger = LocalTableLogger(str(path), columns)

    rows = [[1, 2], [3, 4]]
    logger.add_data(rows)

    expected_data = [{col: val for col, val in zip(columns, row)} for row in rows]
    with open(path.with_suffix(".jsonl"), "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert lines == expected_data

    with pytest.raises(TypeError):
        logger.add_data("not a list")
    with pytest.raises(ValueError):
        logger.add_data([[1]])


def test_wandb_table_logger(monkeypatch):
    mock_table = mock.MagicMock()
    mock_table.columns = ["x", "y"]
    mock_table.data = []

    monkeypatch.setattr("wandb.Table", lambda columns, data=None: mock_table)
    monkeypatch.setattr("wandb.log", lambda x: x)

    logger = WandbTableLogger(name="wandb_test", columns=["x", "y"])
    logger.add_data([[1, 2], [3, 4]])
    assert mock_table.add_data.call_count == 2

    with pytest.raises(TypeError):
        logger.add_data("not a list")
    with pytest.raises(ValueError):
        logger.add_data([[1]])
