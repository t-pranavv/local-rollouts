# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Union

import wandb


class TableLogger:
    """Abstract class for table loggers.

    This class serves as a base for implementing different table loggers, such as
    local file loggers or cloud-based loggers like WandB. It enforces the implementation
    of the :meth:`add_data` method, which is responsible for adding data to the table.

    Examples:
        >>> class MyTableLogger(TableLogger):
        >>>     def __init__(self, name: str, columns: List[str]) -> None:
        >>>         super().__init__(name, columns)
        >>>
        >>>     def add_data(self, rows: List[List[Any]]) -> None:
        >>>         # Implement the logic to add data to the table
        >>>         pass

    """

    def __init__(self, name: Union[str, Path], columns: List[str]) -> None:
        """Initialize the table logger.

        Args:
            name: Name of the table.
            columns: List of column names.

        """

        self._name = Path(name)
        self._data = []
        self._columns = columns

    def add_data(self, rows: List[List[Any]]) -> None:
        """Add data to the table.

        Args:
            rows: List of rows to add to the table, where each row is a list of values.

        """

        raise NotImplementedError("`TableLogger` must implement `add_data()`.")


class LocalTableLogger(TableLogger):
    """Table logger that logs data to a local ``.jsonl`` file."""

    def __init__(self, path: Union[str, Path], columns: List[str]) -> None:
        """Initialize the table logger.

        Args:
            path: Path to the ``.jsonl` file.
            columns: List of column names.

        """

        super().__init__(Path(path).with_suffix(".jsonl"), columns)

        if not self._name.exists():
            self._name.parent.mkdir(parents=True, exist_ok=True)

    def add_data(self, rows: List[List[Any]]) -> None:
        if not all(isinstance(row, list) for row in rows):
            raise TypeError(f"`rows` must be a list of lists, but got '{type(rows)}'.")
        if not all(len(row) == len(self._columns) for row in rows):
            raise ValueError(f"All rows must have {len(self._columns)} columns, but got {len(rows[0])}.")

        self._data.extend(rows)

        with open(self._name, "a", encoding="utf-8") as f:
            for row in rows:
                data_dict = {col: val for col, val in zip(self._columns, row)}
                f.write(json.dumps(data_dict) + "\n")


class WandbTableLogger(TableLogger):
    """Table logger that logs data to WandB."""

    def __init__(self, name: Union[str, Path], columns: List[str]) -> None:
        """Initialize the table logger.

        Args:
            name: Name of the table.
            columns: List of column names.

        """

        super().__init__(name, columns)

        self._table = wandb.Table(columns=self._columns)

    def add_data(self, rows: List[List[Any]]) -> None:
        if not all(isinstance(row, list) for row in rows):
            raise TypeError(f"`rows` must be a list of lists, but got {[type(row) for row in rows]}.")
        if not all(len(row) == len(self._table.columns) for row in rows):
            raise ValueError(f"All rows must have {len(self._columns)} columns, but got {len(rows[0])}.")

        # Need to create a new table every `add_data` call
        # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        self._table = wandb.Table(columns=self._table.columns, data=self._table.data)
        for row in rows:
            self._table.add_data(*row)

        wandb.log({str(self._name): self._table})
