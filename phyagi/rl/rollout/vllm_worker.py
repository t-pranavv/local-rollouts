# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
from torch.distributed._tensor import DTensor
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.models.model_convert import convert_mixformer_sequential_to_llama
from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig


def _maybe_gather(tensor: torch.Tensor, device: torch.device, world_size: int = 1) -> torch.Tensor:
    if isinstance(tensor, DTensor) and world_size > 1:
        return tensor.to(device, non_blocking=True).full_tensor()

    if isinstance(tensor, DTensor) and world_size == 1:
        return tensor.full_tensor()

    return tensor


def _yield_fsdp_weights_for_rollout(
    state_dict: Dict[str, Union[torch.Tensor, DTensor]],
    num_hidden_layers: int,
    device: torch.device,
    world_size: int = 1,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    def _push(out_key: str, in_key: str) -> Generator[Tuple[str, torch.Tensor], None, None]:
        yielded_keys.add(in_key)
        tensor = _maybe_gather(state_dict[in_key], device, world_size)
        yield (out_key, tensor)

    yielded_keys = set()

    mapping = {
        "model.layers.{}.input_layernorm.weight": "layers.{}.ln_1.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ln_2.weight",
        "model.layers.{}.self_attn.qkv_proj.weight": "layers.{}.attn.Wqkv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.out_proj.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.fc1.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.fc2.weight",
    }
    if "layers.1.attn.Wqkv.bias" in state_dict:
        mapping["model.layers.{}.self_attn.qkv_proj.bias"] = "layers.{}.attn.Wqkv.bias"

    yield from _push("model.embed_tokens.weight", "layers.0.wte.weight")

    for i in range(num_hidden_layers):
        for vllm_key, mix_key in mapping.items():
            yield from _push(vllm_key.format(i), mix_key.format(i + 1))

    yield from _push("model.norm.weight", f"layers.{num_hidden_layers + 1}.ln.weight")
    yield from _push("lm_head.weight", f"layers.{num_hidden_layers + 1}.linear.weight")

    missing = set(state_dict.keys()) - yielded_keys
    if missing:
        raise ValueError(f"{missing} weights were not yielded.")


def _convert_mixformer_sequential_to_llama_for_vllm(
    output_dir: Union[str, Path],
    pretrained_model_name_or_path: Union[str, MixFormerSequentialForCausalLM],
    pretrained_tokenizer_name_or_path: Union[str, PreTrainedTokenizer, None] = None,
    overwrite: bool = False,
) -> str:
    output_dir = Path(output_dir)
    if (output_dir / "config.json").exists() and not overwrite:
        return output_dir

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        model = (
            MixFormerSequentialForCausalLM.from_pretrained(pretrained_model_name_or_path)
            if isinstance(pretrained_model_name_or_path, (str, Path))
            else pretrained_model_name_or_path
        )
        tokenizer = (
            AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
            if isinstance(pretrained_tokenizer_name_or_path, (str, Path))
            else pretrained_tokenizer_name_or_path
        )

        converted_model = convert_mixformer_sequential_to_llama(model)

        converted_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return str(output_dir)


class VLLMWorker:
    """vLLM worker.

    This worker is designed to handle model loading, GPU offloading, distributed setup,
    and generation via vLLM.

    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        config: VLLMWorkerConfig,
        distributed_executor_backend: Optional[str] = None,
    ) -> None:
        """Initialize the worker.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model or checkpoint.
            config: Worker configuration.
            distributed_executor_backend: Distributed executor backend to use.
                If ``None``, it will be set to "external_launcher" if ``torch.distributed`` is initialized.

        """

        self.config = config
        self.tp_size = self.config.tensor_parallel_size
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.tp_group = self.rank // self.tp_size

        if distributed_executor_backend is None and torch.distributed.is_initialized():
            distributed_executor_backend = "external_launcher"

        self._llm = LLM(
            model=pretrained_model_name_or_path,
            enable_sleep_mode=True,
            tensor_parallel_size=self.tp_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=self.config.dtype,
            enforce_eager=self.config.enforce_eager,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.config.prompt_length + self.config.response_length,
            disable_log_stats=self.config.disable_log_stats,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            swap_space=self.config.swap_space,
            kv_cache_dtype=self.config.kv_cache_dtype,
            calculate_kv_scales=(self.config.kv_cache_dtype == "fp8"),
            enable_prefix_caching=self.config.enable_prefix_caching,
            preemption_mode=self.config.preemption_mode,
            seed=self.tp_group,
            **(self.config.extra_kwargs or {}),
        )

        self.offload = self.config.offload
        self._asleep = False
        if self.offload:
            self._llm.sleep(level=1)
            self._asleep = True

        self.sampling_params = {
            "n": 1,
            "logprobs": None,
            "max_tokens": self.config.response_length,
            **(self.config.sampling_params or {}),
            "detokenize": False,
        }

    @property
    def _module(self) -> MixFormerSequentialForCausalLM:
        return self._llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

    @classmethod
    def from_mixformer_sequential(
        cls: VLLMWorker,
        output_dir: Union[str, Path],
        pretrained_model_name_or_path: Union[str, Path, MixFormerSequentialForCausalLM],
        pretrained_tokenizer_name_or_path: Union[str, PreTrainedTokenizer],
        config: VLLMWorkerConfig,
        distributed_executor_backend: Optional[str] = None,
        overwrite: bool = False,
    ) -> VLLMWorker:
        # TODO: add hashing mechanism to avoid re-loading the wrong model if `overwrite=False`
        converted_model_path = _convert_mixformer_sequential_to_llama_for_vllm(
            output_dir, pretrained_model_name_or_path, pretrained_tokenizer_name_or_path, overwrite
        )

        return cls(str(converted_model_path), config, distributed_executor_backend)

    @contextmanager
    def on_gpu(self) -> Generator[None, None, None]:
        """Context manager to wake up the model for GPU execution.

        Temporarily moves the model to GPU if offloaded, and returns it to sleep after use.

        """

        if not self.offload or not self._asleep:
            yield
        else:
            torch.cuda.empty_cache()
            self._llm.wake_up()
            self._asleep = False

            try:
                yield
            finally:
                self._llm.sleep(level=1)
                self._asleep = True

    def _assert_awake(self) -> None:
        if self.offload:
            if self._asleep:
                raise RuntimeError(f"{self.__class__} is asleep. Use the `on_gpu()` context manager.")

    def sync_weights(self, actor: MixFormerSequentialForCausalLM) -> None:
        """Synchronize weights from an actor model into the worker.

        Args:
            actor: Model whose weights will be synced.

        """

        self._assert_awake()

        world_size = torch.distributed.get_world_size()
        device = torch.cuda.current_device()
        state_dict = actor.state_dict()

        weight_loader = _yield_fsdp_weights_for_rollout(
            state_dict, num_hidden_layers=actor.config.n_layer, world_size=world_size, device=device
        )
        del state_dict

        self._module.load_weights(weight_loader)

    def generate_from_input_ids(
        self, input_ids: List[List[int]], use_tqdm: bool = False, **sampling_kwargs
    ) -> List[RequestOutput]:
        """Generate model outputs given input token identifiers.

        Args:
            input_ids: Batch of token identifier sequences to generate from.
            use_tqdm: Whether to display a progress bar during generation.

        Returns:
            Generated outputs.

        """

        self._assert_awake()

        sampling_params = {**self.sampling_params, **sampling_kwargs}
        sampling_params = SamplingParams(**sampling_params)

        outputs = self._llm.generate(
            prompts=None, sampling_params=sampling_params, prompt_token_ids=input_ids, use_tqdm=use_tqdm
        )

        return outputs
