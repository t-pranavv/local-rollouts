# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from phyagi.rl.distributed_layout import DistributedLayout


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_configs():
    return MagicMock(fsdp_offload=False, use_meta_tensor=False, checkpoint_mode="sync"), MagicMock()


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
def test_ray_worker(tmp_path, init_mocks, mock_configs, mock_distributed_layout, mock_tokenizer):
    from phyagi.rl.tuners.ray_worker import RayWorker, RayWorkerConfig
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayWorkerConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    assert isinstance(worker.output_dir, Path)
    assert worker.tokenizer == mock_tokenizer


@pytest.mark.is_vllm
def test_ray_worker_configure_models(
    monkeypatch, init_mocks, tmp_path, mock_configs, mock_distributed_layout, mock_tokenizer
):
    from phyagi.rl.tuners.ray_worker import RayWorker, RayWorkerConfig
    from phyagi.utils.checkpoint import CheckpointManager

    monkeypatch.setattr("phyagi.rl.models.actor.Actor", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr("phyagi.rl.models.reference.Reference", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        "phyagi.rl.rollout.vllm_worker.VLLMWorker.from_mixformer_sequential",
        lambda *args, **kwargs: MagicMock(
            on_gpu=MagicMock(return_value=MagicMock(__enter__=lambda x: None, __exit__=lambda x, y, z, a: None))
        ),
    )

    actor_cfg, rollout_cfg = mock_configs

    config = RayWorkerConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    worker.configure_models()
    assert worker.actor is not None
    assert worker.rollout is not None


@pytest.mark.is_vllm
def test_ray_worker_generate_completions(
    monkeypatch, init_mocks, tmp_path, mock_configs, mock_distributed_layout, mock_tokenizer
):
    from phyagi.rl.tuners.ray_worker import RayWorker, RayWorkerConfig
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayWorkerConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    mock_generate = MagicMock(return_value=["output"])
    mock_rollout = MagicMock()
    mock_rollout.on_gpu.return_value.__enter__.return_value = None
    mock_rollout.generate_from_input_ids = mock_generate

    worker.rollout = mock_rollout
    worker.actor = MagicMock()
    worker.actor.model = MagicMock()

    output = worker.generate_completions([101, 102], sync_weights=False)
    assert output == ["output"]
    mock_generate.assert_called_once()


@pytest.mark.is_vllm
def test_ray_worker_save_and_load_actor_model(
    init_mocks, tmp_path, mock_configs, mock_distributed_layout, mock_tokenizer
):
    from phyagi.rl.tuners.ray_worker import RayWorker, RayWorkerConfig
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayWorkerConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    worker.actor = MagicMock()

    worker.save_actor_model(tmp_path)
    worker.actor.save_checkpoint.assert_called_once()

    worker.load_actor_model(tmp_path)
    worker.actor.load_checkpoint.assert_called_once()


@pytest.mark.is_vllm
def test_ray_worker_save_and_load_reference_model(
    init_mocks, tmp_path, mock_configs, mock_distributed_layout, mock_tokenizer
):
    from phyagi.rl.tuners.ray_worker import RayWorker, RayWorkerConfig
    from phyagi.utils.checkpoint import CheckpointManager

    actor_cfg, rollout_cfg = mock_configs

    config = RayWorkerConfig(
        output_dir=tmp_path,
        actor=actor_cfg,
        rollout=rollout_cfg,
        max_steps=1000,
    )

    worker = RayWorker(
        config=config,
        distributed_layout=mock_distributed_layout,
        tokenizer=mock_tokenizer,
        skip_process_group_init=True,
        checkpoint_manager=CheckpointManager(),
    )

    worker.ref = MagicMock()

    worker.save_reference_model(str(tmp_path))
    worker.ref.save_checkpoint.assert_called_once()

    worker.load_reference_model(str(tmp_path))
    worker.ref.load_checkpoint.assert_called_once()
