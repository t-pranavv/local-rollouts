# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict

import pytest

from phyagi.trainers.ds.ds_trainer_callback import DsCallbackHandler, DsTrainerCallback
from phyagi.trainers.ds.ds_training_args import DsTrainingArguments


class MockDeepSpeedEngine:
    pass


@pytest.mark.is_mpi
def test_ds_trainer_callback():
    callback = DsTrainerCallback()
    engine = MockDeepSpeedEngine()
    args = DsTrainingArguments("test_tmp")
    state = {}

    callback.on_save(engine, args, state)


@pytest.mark.is_mpi
def test_ds_callback_handler():
    class MockDsTrainerCallback(DsTrainerCallback):
        def __init__(self):
            self.called = False

        def on_save(
            self, engine: MockDeepSpeedEngine, args: DsTrainingArguments, state: Dict[str, Any], **kwargs
        ) -> None:
            self.called = True

    callback = MockDsTrainerCallback()
    handler = DsCallbackHandler([callback])

    engine = MockDeepSpeedEngine()
    args = DsTrainingArguments("test_tmp")
    state = {}

    handler.on_save(engine, args, state)
    assert callback.called
