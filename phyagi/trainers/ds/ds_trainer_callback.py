# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List

from deepspeed import DeepSpeedEngine

from phyagi.trainers.ds.ds_training_args import DsTrainingArguments


class DsTrainerCallback:
    """DeepSpeed trainer callback.

    The callback is used for customizing the DeepSpeed pipeline. It is called at
    specific points during the training and evaluation process.

    """

    def on_evaluate(self, engine: DeepSpeedEngine, args: DsTrainingArguments, state: Dict[str, Any], **kwargs) -> None:
        """Event called after evaluation.

        Args:
            engine: DeepSpeed engine.
            args: Training arguments.
            state: Client state.

        """

        pass

    def on_save(self, engine: DeepSpeedEngine, args: DsTrainingArguments, state: Dict[str, Any], **kwargs) -> None:
        """Event called after checkpoint has been saved.

        Args:
            engine: DeepSpeed engine.
            args: Training arguments.
            state: Client state.

        """

        pass


class DsCallbackHandler:
    """DeepSpeed callback handler."""

    def __init__(self, callbacks: List[DsTrainerCallback]) -> None:
        """Initializes the callback handler.

        Args:
            callbacks: List of callbacks.

        """

        self._callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback: DsTrainerCallback) -> None:
        """Add a callback to the handler and prevent it from being added twice.

        Args:
            callback: Callback to add.

        """

        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__

        if cb_class in [c.__class__ for c in self._callbacks]:
            raise RuntimeError("Callback already exists.")

        self._callbacks.append(cb)

    def on_evaluate(self, engine: DeepSpeedEngine, args: DsTrainingArguments, state: Dict[str, Any], **kwargs) -> None:
        """Event called after evaluation.

        Args:
            engine: DeepSpeed engine.
            args: Training arguments.
            state: Client state.

        """

        for callback in self._callbacks:
            callback.on_evaluate(engine, args, state, **kwargs)

    def on_save(self, engine: DeepSpeedEngine, args: DsTrainingArguments, state: Dict[str, Any], **kwargs) -> None:
        """Event called after checkpoint has been saved.

        Args:
            engine: DeepSpeed engine.
            args: Training arguments.
            state: Client state.

        """

        for callback in self._callbacks:
            callback.on_save(engine, args, state, **kwargs)
