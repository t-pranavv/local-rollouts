# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import pytest

from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig


@pytest.fixture
def mock_llm(monkeypatch):
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    monkeypatch.setattr("phyagi.rl.rollout.vllm_worker.LLM", mock_class)
    return mock_instance


@pytest.fixture
def dummy_config():
    return VLLMWorkerConfig(
        tensor_parallel_size=1,
        dtype="float16",
        enforce_eager=False,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        prompt_length=16,
        response_length=16,
        max_num_batched_tokens=512,
        enable_chunked_prefill=False,
        swap_space=4,
        kv_cache_dtype="float16",
        enable_prefix_caching=False,
        preemption_mode=None,
        sampling_params={},
        extra_kwargs=None,
        offload=True,
    )


@pytest.mark.is_vllm
def test_vllm_worker(mock_llm, dummy_config):
    from phyagi.rl.rollout.vllm_worker import VLLMWorker

    worker = VLLMWorker("some/path", dummy_config)
    assert worker._asleep is True
    mock_llm.sleep.assert_called_once()


@pytest.mark.is_vllm
def test_vllm_worker_on_gpu(mock_llm, dummy_config):
    from phyagi.rl.rollout.vllm_worker import VLLMWorker

    worker = VLLMWorker("some/path", dummy_config)

    with worker.on_gpu():
        assert worker._asleep is False
        mock_llm.wake_up.assert_called_once()

    assert worker._asleep is True
    assert mock_llm.sleep.call_count == 2


@pytest.mark.is_vllm
def test_vllm_worker_assert_awake(mock_llm, dummy_config):
    from phyagi.rl.rollout.vllm_worker import VLLMWorker

    worker = VLLMWorker("some/path", dummy_config)
    with pytest.raises(RuntimeError):
        worker._assert_awake()


@pytest.mark.is_vllm
def test_vllm_worker_generate_from_input_ids(mock_llm, dummy_config):
    from phyagi.rl.rollout.vllm_worker import VLLMWorker

    dummy_config.offload = False
    mock_llm.generate.return_value = ["generated text"]

    worker = VLLMWorker("some/path", dummy_config)
    output = worker.generate_from_input_ids([[1, 2, 3]])
    assert output == ["generated text"]

    mock_llm.generate.assert_called_once()
