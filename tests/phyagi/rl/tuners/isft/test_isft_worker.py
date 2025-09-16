# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import pytest
import torch

from phyagi.rl.distributed_layout import DistributedLayout


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_configs():
    return MagicMock(fsdp_offload=False, use_meta_tensor=False), MagicMock()


@pytest.fixture
def mock_distributed_layout():
    return DistributedLayout(
        n_nodes=1,
        n_gpus_per_node=1,
        actor_cp_size=1,
        actor_tp_size=1,
        rollout_tp_size=1,
    )


@pytest.fixture
def init_mocks(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)

    mock_mesh = MagicMock()
    mock_mesh.size.return_value = 1
    mock_submesh = MagicMock()
    mock_submesh._flatten.return_value = None
    mock_submesh.size.return_value = 1
    mock_mesh.__getitem__.side_effect = lambda key: (
        mock_submesh
        if key in {"data_parallel", "tensor_parallel", "context_parallel", "data_context_parallel"}
        else MagicMock()
    )
    monkeypatch.setattr("phyagi.rl.distributed_layout.init_device_mesh", lambda *args, **kwargs: mock_mesh)

    monkeypatch.setattr("phyagi.rl.ray_utils.get_ray_logger", lambda name: MagicMock())


@pytest.mark.is_vllm
def test_isft_worker_compute_loss(tmp_path, init_mocks, mock_configs, mock_distributed_layout, mock_tokenizer):
    from phyagi.rl.tuners.isft.isft_worker import RayISFTConfig, RayISFTWorker
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayISFTConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayISFTWorker(
        config=config,
        tokenizer=mock_tokenizer,
        distributed_layout=mock_distributed_layout,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    B, T = 2, 5
    logps = torch.randn(B, T, dtype=torch.float32)
    masks = torch.ones(B, T + 1, dtype=torch.bool)
    advantages = torch.rand(B, T + 1, dtype=torch.float32)

    loss = worker.compute_loss(logps, masks, advantages)
    assert loss.ndim == 0


@pytest.mark.is_vllm
def test_isft_worker_update_actor_policy(tmp_path, init_mocks, mock_configs, mock_distributed_layout, mock_tokenizer):
    from phyagi.rl.tuners.isft.isft_worker import RayISFTConfig, RayISFTWorker
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayISFTConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayISFTWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    packed_batch = MagicMock()
    packed_batch.tokens = torch.randint(0, 10, (2, 5))
    packed_batch.masks = torch.ones(2, 6, dtype=torch.bool)
    packed_batch.cu_seqlens = torch.tensor([0, 5, 10])
    packed_batch.advantages = torch.rand(2, 6)

    worker.actor = MagicMock()
    worker.actor.config.gradient_clipping = 1.0
    worker.actor.model.device = torch.device("cpu")
    worker.actor.model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    worker.actor.compute_logprobs.return_value = {"logprobs": torch.randn(2, 5, requires_grad=True), "entropy": None}
    worker.actor.optimizer = MagicMock()
    worker.actor.lr_scheduler = MagicMock()
    worker.actor.lr_scheduler.get_last_lr.return_value = [0.001]

    result = worker.update_actor_policy([packed_batch])

    assert "train/loss" in result
    assert "train/lr" in result
    assert "train/grad_norm" in result
    assert isinstance(result["train/loss"], float)
    assert isinstance(result["train/lr"], float)
    assert isinstance(result["train/grad_norm"], float)
