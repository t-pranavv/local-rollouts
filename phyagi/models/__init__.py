# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from phyagi.models.mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
    MixFormerSequentialConfig,
    MixFormerSequentialForCausalLM,
    MixFormerSequentialForSequenceClassification,
)

# Register configurations and models at a top-level package to make them available for auto-based classes
AutoConfig.register(MIXFORMER_SEQUENTIAL_MODEL_TYPE, MixFormerSequentialConfig)
AutoModelForCausalLM.register(MixFormerSequentialConfig, MixFormerSequentialForCausalLM)
AutoModelForSequenceClassification.register(MixFormerSequentialConfig, MixFormerSequentialForSequenceClassification)
