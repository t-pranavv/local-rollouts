# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import Trainer

from phyagi.utils.hf_utils import AzureStorageRotateCheckpointMixin


class HfTrainer(AzureStorageRotateCheckpointMixin, Trainer):
    """Hugging Face trainer."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the trainer."""

        super().__init__(*args, **kwargs)
