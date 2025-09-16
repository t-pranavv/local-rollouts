# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

from phyagi.rl.models.actor_config import ActorConfig
from phyagi.rl.rollout.vllm_worker_config import VLLMWorkerConfig
from phyagi.rl.tuners.grpo.grpo_config import RayGRPOConfig
from phyagi.utils.file_utils import get_full_path


def test_ray_grpo_config():
    config = RayGRPOConfig(output_dir="output", epochs=1)
    assert config.output_dir == get_full_path("output")
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
    assert config.num_policy_updates_per_batch == 1
    assert config.kl_coeff == 0.001
    assert config.epsilon_low == 0.2
    assert config.epsilon_high == 0.2
    assert isinstance(config.actor, ActorConfig)
    assert isinstance(config.rollout, VLLMWorkerConfig)
    assert config.dataloader_shuffle is True
    assert config.dataloader_num_workers == 1
    assert config.reward_num_workers == 1
    assert isinstance(config.wandb, dict)
    assert config.wandb == {}

    shutil.rmtree("output", ignore_errors=True)

    custom_actor = ActorConfig()
    custom_rollout = VLLMWorkerConfig()

    config = RayGRPOConfig(
        output_dir="custom_output",
        n_nodes=2,
        n_gpus_per_node=4,
        do_final_eval=False,
        eval_before_training=True,
        epochs=3,
        max_steps=100,
        log_n_eval_completions=-1,
        save_steps=0.5,
        save_final_checkpoint=False,
        eval_steps=0.25,
        seed=123,
        group_size=8,
        train_batch_size=64,
        train_max_micro_batch_size_per_gpu=16,
        normalize_advantage_std=False,
        num_policy_updates_per_batch=1,
        kl_coeff=0.005,
        epsilon_low=0.1,
        epsilon_high=0.3,
        actor=custom_actor,
        rollout=custom_rollout,
        dataloader_shuffle=False,
        dataloader_num_workers=8,
        reward_num_workers=2,
        wandb={"project": "test_project"},
    )
    assert config.output_dir == get_full_path("custom_output")
    assert config.n_nodes == 2
    assert config.n_gpus_per_node == 4
    assert config.do_final_eval is False
    assert config.eval_before_training is True
    assert config.epochs == 3
    assert config.max_steps == 100
    assert config.log_n_eval_completions == -1
    assert config.save_steps == 0.5
    assert config.save_final_checkpoint is False
    assert config.eval_steps == 0.25
    assert config.seed == 123
    assert config.group_size == 8
    assert config.train_batch_size == 64
    assert config.train_max_micro_batch_size_per_gpu == 16
    assert config.normalize_advantage_std is False
    assert config.num_policy_updates_per_batch == 1
    assert config.kl_coeff == 0.005
    assert config.epsilon_low == 0.1
    assert config.epsilon_high == 0.3
    assert config.actor == custom_actor
    assert config.rollout == custom_rollout
    assert config.dataloader_shuffle is False
    assert config.dataloader_num_workers == 8
    assert config.reward_num_workers == 2
    assert config.wandb == {"project": "test_project"}

    shutil.rmtree("custom_output", ignore_errors=True)
