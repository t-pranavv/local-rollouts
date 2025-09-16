# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from trl import DPOTrainer, GRPOTrainer, SFTTrainer

from phyagi.utils.hf_utils import AzureStorageRotateCheckpointMixin


class HfSFTTuner(AzureStorageRotateCheckpointMixin, SFTTrainer):
    """Hugging Face supervised fine-tuning tuner."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the tuner."""

        super().__init__(*args, **kwargs)


class HfDPOTuner(AzureStorageRotateCheckpointMixin, DPOTrainer):
    """Hugging Face direct preference optimization tuner."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the tuner."""

        super().__init__(*args, **kwargs)


class HfGRPOTuner(AzureStorageRotateCheckpointMixin, GRPOTrainer):
    """Hugging Face group relative policy optimization tuner."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the tuner."""

        super().__init__(*args, **kwargs)
