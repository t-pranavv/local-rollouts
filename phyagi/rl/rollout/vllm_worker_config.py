# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VLLMWorkerConfig:
    """vLLM worker configuration.

    Args:
        prompt_length: Maximum prompt length accepted.
        response_length: Maximum length of the generated response.
        tensor_parallel_size: Number of GPUs to use for distributed execution with tensor parallelism.
        offload: Whether to offload the model to CPU when not in use.
        dtype: Data type for the model weights and activations.
            If ``auto``, we use the ``torch_dtype`` attribute specified in the model configuration file.
            If the ``torch_dtype`` in the configuration is ``float32``, we will use ``float16`` instead.
        gpu_memory_utilization: Ratio of GPU memory to reserve for the model weights, activations, and KV-cache.
        swap_space: Size (GiB) of CPU memory per GPU to use as swap space.
        enforce_eager: Whether to enforce eager execution.
        hf_overrides: Arguments forwarded to the Hugging Face configuration.
        enable_chunked_prefill: Whether prefill requests can be chunked based on the ``max_num_batched_tokens``.
        enable_prefix_caching: Whether to enable prefix caching.
        preemption_mode: Preemption mode for KV cache.
        max_num_batched_tokens: If ``enable_chunked_prefill`` is set, prefill requests will be chunked based on ``max_num_batched_tokens``.
        max_num_seqs: Maximum number of sequences in a batch.
        kv_cache_dtype: Data type for kv cache storage.
            If ``auto``, will use model data type.
        sampling_params: Sampling parameters to be passed to ``vllm.SamplingParams``.
        eval_sampling_params: Sampling parameters to be used during evaluation.
        disable_log_stats: Whether to disable logging of vLLM statistics.
        extra_kwargs: Additional keyword arguments for the ``vllm.LLM`` constructor.

    """

    prompt_length: int = field(default=1024, metadata={"help": "Maximum prompt length accepted."})

    response_length: int = field(default=8192, metadata={"help": "Maximum length of the generated response."})

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs to use for distributed execution with tensor parallelism."}
    )

    offload: bool = field(default=True, metadata={"help": "Whether to offload the model to CPU when not in use."})

    dtype: str = field(
        default="auto",
        metadata={"help": "Data type for the model weights and activations."},
    )

    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "Ratio of GPU memory to reserve for the model weights, activations, and KV-cache."},
    )

    swap_space: float = field(
        default=64.0,
        metadata={"help": "Size (GiB) of CPU memory per GPU to use as swap space."},
    )

    enforce_eager: Optional[bool] = field(default=None, metadata={"help": "Whether to enforce eager execution."})

    hf_overrides: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Arguments forwarded to the Hugging Face configuration."}
    )

    enable_chunked_prefill: bool = field(
        default=True,
        metadata={"help": "Whether prefill requests can be chunked based on the `max_num_batched_tokens`."},
    )

    enable_prefix_caching: bool = field(default=False, metadata={"help": "Whether to enable prefix caching."})

    preemption_mode: Optional[str] = field(default=None, metadata={"help": "Preemption mode for KV cache."})

    max_num_batched_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "If `enable_chunked_prefill` is set, prefill requests will be chunked based on `max_num_batched_tokens`."
        },
    )

    max_num_seqs: Optional[int] = field(default=None, metadata={"help": "Maximum number of sequences in a batch."})

    kv_cache_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Data type for kv cache storage."},
    )

    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {"temperature": 1.0},
        metadata={"help": "Sampling parameters to be passed to `vllm.SamplingParams`."},
    )

    eval_sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {},
        metadata={"help": "Sampling parameters to be used during evaluation."},
    )

    disable_log_stats: bool = field(default=True, metadata={"help": "Whether to disable logging of vLLM statistics."})

    extra_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Additional keyword arguments for the `vllm.LLM` constructor."}
    )

    def __post_init__(self) -> None:
        if self.gpu_memory_utilization < 0 or self.gpu_memory_utilization > 1:
            raise ValueError(f"`gpu_memory_utilization` must be in [0, 1], but got {self.gpu_memory_utilization}.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Returns:
            Attributes encoded as a dictionary.

        """

        return asdict(self)
