# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from pathlib import Path

import pytest

from phyagi.rl.models.actor_config import ActorConfig
from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig
from phyagi.rl.tuners.ray_worker_config import RayWorkerConfig


def test_ray_worker_config():
    tmp_dir = Path("temp_test_ray_config")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    config = RayWorkerConfig(output_dir=tmp_dir, epochs=1)
    assert config.n_nodes == 1
    assert config.n_gpus_per_node == 1
    assert config.do_final_eval is True
    assert config.eval_before_training is False
    assert config.epochs == 1
    assert config.max_steps is None
    assert config.log_n_eval_completions == 20
    assert config.save_steps == -1
    assert config.save_final_checkpoint is True
    assert config.eval_steps == 0
    assert config.seed == 1
    assert config.group_size == 1
    assert config.train_batch_size == 1
    assert config.train_max_micro_batch_size_per_gpu is None
    assert config.normalize_advantage_std is False
    assert isinstance(config.actor, ActorConfig)
    assert isinstance(config.rollout, VLLMWorkerConfig)
    assert config.checkpoint_mode == "sync"
    assert config.dataloader_shuffle is True
    assert config.dataloader_num_workers == 1
    assert config.reward_num_workers == 1
    assert config.wandb == {}

    actor = ActorConfig()
    rollout = VLLMWorkerConfig(prompt_length=512)
    custom_config = RayWorkerConfig(
        output_dir=str(tmp_dir),
        n_nodes=2,
        n_gpus_per_node=4,
        do_final_eval=False,
        eval_before_training=True,
        max_steps=1000,
        save_steps=0.2,
        eval_steps=0.3,
        seed=42,
        group_size=8,
        train_batch_size=32,
        train_max_micro_batch_size_per_gpu=4,
        normalize_advantage_std=True,
        actor=actor,
        rollout=rollout,
        checkpoint_mode="async",
        dataloader_shuffle=False,
        dataloader_num_workers=2,
        reward_num_workers=3,
        wandb={"project": "test"},
    )
    assert custom_config.output_dir == tmp_dir.resolve()
    assert custom_config.n_nodes == 2
    assert custom_config.n_gpus_per_node == 4
    assert custom_config.do_final_eval is False
    assert custom_config.eval_before_training is True
    assert custom_config.epochs is None
    assert custom_config.max_steps == 1000
    assert custom_config.save_steps == 0.2
    assert custom_config.eval_steps == 0.3
    assert custom_config.seed == 42
    assert custom_config.group_size == 8
    assert custom_config.train_batch_size == 32
    assert custom_config.train_max_micro_batch_size_per_gpu == 4
    assert custom_config.normalize_advantage_std is True
    assert isinstance(custom_config.actor, ActorConfig)
    assert isinstance(custom_config.rollout, VLLMWorkerConfig)
    assert custom_config.rollout.prompt_length == 512
    assert custom_config.checkpoint_mode == "async"
    assert custom_config.dataloader_shuffle is False
    assert custom_config.dataloader_num_workers == 2
    assert custom_config.reward_num_workers == 3
    assert custom_config.wandb["project"] == "test"
    assert tmp_dir.exists() and tmp_dir.is_dir()

    dict_config = RayWorkerConfig(
        output_dir=tmp_dir,
        epochs=1,
        actor=actor.to_dict(),
        rollout=rollout.to_dict(),
    )
    assert isinstance(dict_config.actor, ActorConfig)
    assert isinstance(dict_config.rollout, VLLMWorkerConfig)

    serialized = dict_config.to_dict(json_serialize=True)
    assert isinstance(serialized["output_dir"], str)
    assert isinstance(serialized["actor"], dict)
    assert isinstance(serialized["rollout"], dict)

    with pytest.raises(ValueError):
        RayWorkerConfig(output_dir=tmp_dir)

    shutil.rmtree(tmp_dir)
