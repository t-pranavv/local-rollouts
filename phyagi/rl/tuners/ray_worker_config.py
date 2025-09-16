# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import torch

from phyagi.rl.models.actor_config import ActorConfig
from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig
from phyagi.utils.file_utils import get_full_path


@dataclass
class RayWorkerConfig:
    """Ray worker configuration.

    Args:
        output_dir: Output folder where checkpoints and states will be written.
        n_nodes: Number of training nodes.
        n_gpus_per_node: Number of GPUs per node.
        do_final_eval: Whether to run last step evaluation on the validation set.
        eval_before_training: Whether to evaluate the model before training.
        epochs: Number of training epochs.
            If ``None``, ``max_steps`` must be set.
        max_steps: Maximum number of training steps.
            If ``None``, ``epochs`` must be set.
        log_n_eval_completions: Number of completions for questions from the validation set to save on each evaluation step.
            If ``-1``, all completions are saved.
        save_steps: Number of steps between checkpoints.
            If set to range ``[0, 1)``, will be interpreted as a ratio of total training steps.
        save_final_checkpoint: Whether to save last step checkpoint.
        eval_steps: Number of steps between evaluations.
            If set to range ``[0, 1)``, will be interpreted as a ratio of total training steps.
        seed: Random seed.
        group_size: Group size (completions per question).
        train_batch_size: Number of questions used in each round.
            Must be divisible by ``world_size``.
        train_max_micro_batch_size_per_gpu: Maximum micro-batch size of packed documents per GPU.
            If ``None``, tries to fit the entire batch (``train_batch_size * group_size / [n_nodes * n_gpus_per_node]`` documents in the worst case) on each GPU.
        normalize_advantage_std: Whether to normalize the advantages standard deviation.
        actor: Actor configuration.
        rollout: Rollout configuration.
        checkpoint_mode: Checkpointing mode.
        dataloader_shuffle: Whether to shuffle the data loader (training).
        dataloader_num_workers: Number of subprocesses to use for data loading.
        reward_num_workers: Number of reward workers.
        wandb: WandB configuration.

    """

    output_dir: Union[str, Path] = field(
        metadata={"help": "Output folder where checkpoints and states will be written."}
    )

    n_nodes: int = field(default=1, metadata={"help": "Number of training nodes."})

    n_gpus_per_node: int = field(default=1, metadata={"help": "Number of GPUs per node."})

    do_final_eval: bool = field(
        default=True, metadata={"help": "Whether to run last step evaluation on the validation set."}
    )

    eval_before_training: bool = field(
        default=False, metadata={"help": "Whether to evaluate the model before training."}
    )

    epochs: Optional[int] = field(default=None, metadata={"help": "Number of training epochs."})

    max_steps: Optional[int] = field(default=None, metadata={"help": "Maximum number of training steps."})

    log_n_eval_completions: int = field(
        default=20,
        metadata={
            "help": "Number of completions for questions from the validation set to save on each evaluation step."
        },
    )

    save_steps: Union[int, float] = field(
        default=-1,
        metadata={"help": "Number of steps between checkpoints."},
    )

    save_final_checkpoint: bool = field(default=True, metadata={"help": "Whether to save last step checkpoint."})

    eval_steps: Union[int, float] = field(
        default=0,
        metadata={"help": "Number of steps between evaluations."},
    )

    seed: int = field(default=1, metadata={"help": "Random seed."})

    group_size: int = field(default=1, metadata={"help": "Group size (completions per question)."})

    train_batch_size: int = field(default=1, metadata={"help": "Number of questions used in each round."})

    train_max_micro_batch_size_per_gpu: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum micro-batch size of packed documents per GPU."},
    )

    normalize_advantage_std: bool = field(
        default=False, metadata={"help": "Whether to normalize the advantage standard deviation."}
    )

    actor: ActorConfig = field(default_factory=ActorConfig, metadata={"help": "Actor configuration."})

    rollout: VLLMWorkerConfig = field(
        default_factory=VLLMWorkerConfig,
        metadata={"help": "Rollout configuration."},
    )

    checkpoint_mode: Literal["sync", "async"] = field(default="sync", metadata={"help": "Checkpointing mode."})

    dataloader_shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the data loader (training)."})

    dataloader_num_workers: int = field(default=1, metadata={"help": "Number of subprocesses to use for data loading."})

    reward_num_workers: int = field(default=1, metadata={"help": "Number of reward workers."})

    wandb: Dict[str, Any] = field(default_factory=lambda: {}, metadata={"help": "WandB configuration."})

    def _init_ray_actor_config(self) -> None:
        if isinstance(self.actor, ActorConfig):
            return
        self.actor = ActorConfig(**self.actor)

    def _init_vllm_worker_config(self) -> None:
        if isinstance(self.rollout, VLLMWorkerConfig):
            return
        self.rollout = VLLMWorkerConfig(**self.rollout)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)

        self.output_dir = get_full_path(self.output_dir, create_folder=True)

        self._init_ray_actor_config()
        self._init_vllm_worker_config()

        if self.epochs is None and self.max_steps is None:
            raise ValueError(f"Either `epochs` or `max_steps` must be set, but got {self.epochs} and {self.max_steps}.")

    def to_dict(self, json_serialize: bool = False) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Args:
            json_serialize: Whether to serialize non-compatible types into native types supported by JSON.

        Returns:
            Attributes encoded as a dictionary.

        """

        args = copy.deepcopy(asdict(self))

        if json_serialize:
            args["output_dir"] = str(self.output_dir)
            args["actor"] = self.actor.to_dict()
            args["rollout"] = self.rollout.to_dict()

        return args
