# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig


def test_vllm_worker_config():
    config = VLLMWorkerConfig()
    assert config.prompt_length == 1024
    assert config.response_length == 8192
    assert config.tensor_parallel_size == 1
    assert config.offload is True
    assert config.dtype == "auto"
    assert config.gpu_memory_utilization == 0.9
    assert config.swap_space == 64.0
    assert config.enforce_eager is None
    assert config.hf_overrides is None
    assert config.enable_chunked_prefill is True
    assert config.enable_prefix_caching is False
    assert config.preemption_mode is None
    assert config.max_num_batched_tokens is None
    assert config.max_num_seqs is None
    assert config.kv_cache_dtype == "auto"
    assert config.sampling_params == {"temperature": 1.0}
    assert config.disable_log_stats is True
    assert config.extra_kwargs is None

    custom_config = VLLMWorkerConfig(
        prompt_length=2048,
        response_length=4096,
        tensor_parallel_size=4,
        offload=False,
        dtype="float16",
        gpu_memory_utilization=0.8,
        swap_space=128.0,
        enforce_eager=True,
        hf_overrides={"use_cache": False},
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        preemption_mode="swap",
        max_num_batched_tokens=512,
        max_num_seqs=128,
        kv_cache_dtype="float16",
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        disable_log_stats=False,
        extra_kwargs={"foo": "bar"},
    )
    assert custom_config.prompt_length == 2048
    assert custom_config.response_length == 4096
    assert custom_config.tensor_parallel_size == 4
    assert custom_config.offload is False
    assert custom_config.dtype == "float16"
    assert custom_config.gpu_memory_utilization == 0.8
    assert custom_config.swap_space == 128.0
    assert custom_config.enforce_eager is True
    assert custom_config.hf_overrides == {"use_cache": False}
    assert custom_config.enable_chunked_prefill is False
    assert custom_config.enable_prefix_caching is True
    assert custom_config.preemption_mode == "swap"
    assert custom_config.max_num_batched_tokens == 512
    assert custom_config.max_num_seqs == 128
    assert custom_config.kv_cache_dtype == "float16"
    assert custom_config.sampling_params == {"temperature": 0.7, "top_p": 0.9}
    assert custom_config.disable_log_stats is False
    assert custom_config.extra_kwargs == {"foo": "bar"}

    with pytest.raises(ValueError):
        VLLMWorkerConfig(gpu_memory_utilization=1.5)
    with pytest.raises(ValueError):
        VLLMWorkerConfig(gpu_memory_utilization=-0.1)
