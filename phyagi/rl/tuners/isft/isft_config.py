# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field

from phyagi.rl.tuners.ray_worker_config import RayWorkerConfig


@dataclass
class RayISFTConfig(RayWorkerConfig):
    """Ray ISFT configuration.

    Args:
        generation_interval: Number of steps between rollout generations.
            This parameter will define how much data is generated and how often the model weights are
            synchronized with the rollouts. During a generation step (that occurs after every ``generation_interval`` steps),
            the trainer will generate ``generation_interval * train_batch_size * group_size`` completions that
            will be used to perform the next ``generation_interval`` actor model updates."

    """

    generation_interval: int = field(default=1, metadata={"help": "Number of steps between rollout generations."})
