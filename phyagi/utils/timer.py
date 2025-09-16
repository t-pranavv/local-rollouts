# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import time
from typing import Any, Dict


class Timer:
    """Timer to measure the time taken for different parts of a process."""

    def __init__(self) -> None:
        """Initialize the timer."""

        self._timings = {}
        self._start_time = None
        self._tag = None

    def __enter__(self) -> None:
        self._timings[self._tag] = time.time()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self._timings[self._tag] = time.time() - self._timings[self._tag]
        self._tag = None

        return False

    def start(self) -> None:
        """Start the timer."""

        self._start_time = time.time()

    def end(self) -> Dict[str, float]:
        """End the timer.

        Returns:
            Dictionary containing the total time taken and the timings for each tag.

        """

        return self.metrics(end=True)

    def metrics(self, total_key: str = "total", end: bool = False) -> Dict[str, float]:
        """Get the metrics.

        Args:
            total_key: Key for the total time taken.
            end: Whether to end the timer.

        Returns:
            Dictionary containing the total time taken and the timings for each tag.

        """

        timings = {total_key: time.time() - self._start_time if self._start_time else 0, **self._timings}

        if end:
            self._start_time = None
            self._timings = {}

        return timings

    def measure(self, tag: str) -> Timer:
        """Measure the time taken for a specific tag.

        Args:
            tag: Tag for the measurement.

        Returns:
            :class:`Timer` instance with the specified tag.

        """

        self._tag = tag

        return self
