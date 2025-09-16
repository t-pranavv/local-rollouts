# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Union


@dataclass
class ActorConfig:
    """Actor model (with FSDP support) configuration.

    Args:
        model: Model configuration.
        use_meta_tensor: Whether to initialize the model using meta-tensors.
        optimizer: Optimizer configuration.
        scheduler: Learning rate scheduler configuration.
        gradient_clipping: Gradient clipping value.
        manual_offload: Whether to manually offload the model to CPU.
        fsdp_offload: Whether to use FSDP-based CPU offload.
        activation_checkpointing: Whether to use activation checkpointing.
        dtype: Precision of the model.
        adam_8bit: Whether to use 8-bit AdamW optimizer.
        context_parallel_size: Size of the context parallel group.
        tensor_parallel_size: Size of the tensor parallel group.
        tp_loss_parallel: Whether to use loss parallelism in tensor parallelism.

    """

    model: Dict[str, Any] = field(default_factory=lambda: {}, metadata={"help": "Model configuration."})

    use_meta_tensor: bool = field(
        default=False, metadata={"help": "Whether to initialize the model using meta-tensors."}
    )

    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-6,
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
        },
        metadata={"help": "Optimizer configuration."},
    )

    scheduler: Dict[str, Union[int, float, bool, None]] = field(
        default_factory=lambda: {
            "warmup_num_steps": 500,
        },
        metadata={"help": "Learning rate scheduler configuration."},
    )

    gradient_clipping: float = field(default=1.0, metadata={"help": "Gradient clipping value."})

    manual_offload: bool = field(
        default=False,
        metadata={"help": "Whether to manually offload the model to CPU."},
    )

    fsdp_offload: bool = field(
        default=False,
        metadata={"help": "Whether to use FSDP-based CPU offload."},
    )

    activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use activation checkpointing."},
    )

    dtype: str = field(default="bfloat16", metadata={"help": "Precision of the model."})

    adam_8bit: bool = field(default=False, metadata={"help": "Whether to use 8-bit AdamW optimizer."})

    context_parallel_size: int = field(default=1, metadata={"help": "Size of the context parallel group."})

    tensor_parallel_size: int = field(default=1, metadata={"help": "Size of the tensor parallel group."})

    tp_loss_parallel: bool = field(
        default=False, metadata={"help": "Whether to use loss parallelism in tensor parallelism."}
    )

    def __post_init__(self) -> None:
        if self.use_meta_tensor:
            raise NotImplementedError("`use_meta_tensor` has not been implemented yet.")

        if self.manual_offload and self.fsdp_offload:
            raise ValueError("`manual_offload` and `fsdp_offload` cannot be used together.")

        cp_size = self.model.get("cp_size", 1)
        if cp_size != self.context_parallel_size:
            raise ValueError(
                f"`model.cp_size` must be equal to `context_parallel_size`, but got {cp_size} and {self.context_parallel_size}."
            )

        tp_size = self.model.get("tp_size", 1)
        if tp_size != self.tensor_parallel_size:
            raise ValueError(
                f"`model.tp_size` must be equal to `tensor_parallel_size`, but got {tp_size} and {self.tensor_parallel_size}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Returns:
            Attributes encoded as a dictionary.

        """

        return asdict(self)
