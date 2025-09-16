# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import CompletionOutput

from phyagi.datasets.rl.formatting_utils import patch_tokenizer_generation_tag
from phyagi.datasets.rl.packing import PackedBatch, distribute_and_pack_sequences
from phyagi.rl.distributed_layout import DistributedLayout
from phyagi.rl.ray_utils import get_ray_logger, ray_wandb_graceful_shutdown
from phyagi.rl.rewards.reward import Reward
from phyagi.rl.rewards.reward_manager import RewardManager
from phyagi.rl.tuners.isft.isft_config import RayISFTConfig
from phyagi.rl.tuners.isft.isft_worker import RayISFTWorker
from phyagi.trainers.trainer_utils import RepeatingLoader, StatefulDistributedSampler
from phyagi.utils.checkpoint import CheckpointManager
from phyagi.utils.file_utils import (
    is_checkpoint_available,
    load_json_file,
    save_json_file,
)
from phyagi.utils.logging_table import LocalTableLogger, TableLogger, WandbTableLogger
from phyagi.utils.timer import Timer


class RayISFTTuner:
    """Ray ISFT tuner."""

    def __init__(
        self,
        args: Optional[RayISFTConfig] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        rewards: Dict[str, Reward] = None,
        **kwargs,
    ) -> None:
        """Initialize the tuner.

        Args:
            args: ISFT tuning arguments.
            tokenizer: Tokenizer used for encoding the dataset.
            data_collator: Collate function used for creating a batch from ``train_dataset`` and ``eval_dataset``.
            train_dataset: Dataset used for training.
                If set to ``None``, :meth:`train` will not be available.
            eval_dataset: Dataset used for evaluation.
                If set to ``None``, will not perform evaluation.
            rewards: Dictionary of reward functions.

        """

        if not ray.is_initialized():
            ray.init()

        self.args = args
        self.tokenizer = patch_tokenizer_generation_tag(tokenizer) if tokenizer is not None else None
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.rewards = RewardManager(rewards, num_workers=self.args.reward_num_workers)

        wandb_kwargs = self.args.wandb
        wandb_kwargs["config"] = {"tuning_args": self.args.to_dict(json_serialize=True)}

        self.logger = get_ray_logger(__name__, **wandb_kwargs)
        self.table_logger = self._create_completions_table("eval_completions", wandb=bool(self.args.wandb))

        self.checkpoint_manager = CheckpointManager(mode=self.args.checkpoint_mode)
        self.dist_layout = DistributedLayout(
            n_nodes=self.args.n_nodes,
            n_gpus_per_node=self.args.n_gpus_per_node,
            actor_cp_size=self.args.actor.context_parallel_size,
            actor_tp_size=self.args.actor.tensor_parallel_size,
            rollout_tp_size=self.args.rollout.tensor_parallel_size,
        )

        self.workers = None
        self.global_step = 0

        if self.args.epochs is not None:
            self.training_steps = len(train_dataset) // self.args.train_batch_size * self.args.epochs
        if self.args.max_steps is not None:
            if self.args.epochs is not None:
                self.logger.warning(
                    "Both `epochs` and `max_steps` have been set. `max_steps` will be used for training."
                )
            self.training_steps = self.args.max_steps

        self.save_steps = (
            int(self.training_steps * self.args.save_steps)
            if isinstance(self.args.save_steps, float)
            else self.args.save_steps
        )
        self.eval_steps = (
            int(self.training_steps * self.args.eval_steps)
            if isinstance(self.args.eval_steps, float)
            else self.args.eval_steps
        )

        if self.args.train_batch_size % self.dist_layout.world_size != 0:
            raise ValueError(
                f"`train_batch_size` must be divisible by `world_size`, but got {self.args.train_batch_size}"
                f" and {self.dist_layout.world_size}."
            )

        self.logger.info(f"Tuning arguments: {self.args.to_dict(json_serialize=True)}")

    def _create_completions_table(self, name: str, wandb: bool = False) -> TableLogger:
        columns = [
            "step",
            "question_idx",
            "repetition",
            "data_source",
            "prompt",
            "ground_truth",
            "completion",
            "total_reward",
        ]

        if wandb:
            return WandbTableLogger(name, columns)

        return LocalTableLogger(self.args.output_dir / name, columns)

    def _is_eval_step(self, step: int) -> bool:
        if self.eval_dataset is None:
            return False

        do_first_eval = (step == 1) and self.args.eval_before_training
        do_periodic_eval = self.eval_steps > 0 and (step % self.eval_steps) == 0
        do_final_eval = (step == self.training_steps) and self.args.do_final_eval

        return do_first_eval or do_periodic_eval or do_final_eval

    def _is_checkpoint_save_step(self, step: int, resume_step: Optional[int] = None) -> bool:
        # We never save the first step after we just resumed from a checkpoint
        if resume_step and (step == resume_step):
            return False

        do_periodic_save = self.save_steps > 0 and (step % self.save_steps) == 0
        do_final_save = (step == self.training_steps) and self.args.save_final_checkpoint

        return do_periodic_save or do_final_save

    def _is_generation_step(self, step: int) -> bool:
        """Check if the current step is a generation step."""
        return (step - 1) % self.args.generation_interval == 0

    def _init_workers(self) -> None:
        worker_cls = ray.remote(num_gpus=1.0)(RayISFTWorker)
        worker_kwargs = dict(
            config=self.args,
            checkpoint_manager=self.checkpoint_manager,
            distributed_layout=self.dist_layout,
            tokenizer=self.tokenizer,
            total_training_steps=self.training_steps,
        )

        self.workers = [
            worker_cls.options(runtime_env={"env_vars": worker_env}).remote(**worker_kwargs)
            for worker_env in self.dist_layout.get_ray_worker_envs()
        ]

        ray.get([worker.configure_models.remote() for worker in self.workers])

    def get_train_dataloader(self, total_consumed_samples: int = 0, shuffle: bool = True) -> DataLoader:
        """Get the training dataloader.

        Args:
            total_consumed_samples: Total number of samples consumed so far.
            shuffle: Whether to shuffle the dataset.

        Returns:
            Training dataloader.

        """

        # Since `RayISFTTuner` runs on a single process, we manually set the rank to 0
        sampler = StatefulDistributedSampler(
            self.train_dataset,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=self.args.seed,
            total_consumed_samples=total_consumed_samples,
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.args.train_batch_size * self.args.generation_interval,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            collate_fn=self.data_collator,
        )

    def generate_completions(
        self,
        prompt_input_ids: List[List[int]],
        num_repetitions: int = 1,
        sync_weights: bool = True,
        batch_size: Optional[int] = None,
        **generation_kwargs,
    ) -> List[List[CompletionOutput]]:
        """Generate completions for the given prompts.

        Args:
            prompt_input_ids: Input prompts to generate completions for.
            num_repetitions: Number of repetitions for each prompt.
            sync_weights: Whether to synchronize weights with the rollout model.
            batch_size: Batch size for generation.

        Returns:
            List of lists of completion outputs, where each inner list corresponds to a prompt and contains
            the generated completions.

        """

        worker_inputs = self.dist_layout.distribute_data(prompt_input_ids, context="rollout")
        outputs = ray.get(
            [
                worker.generate_completions.remote(
                    worker_input,
                    num_repetitions=num_repetitions,
                    sync_weights=sync_weights,
                    batch_size=batch_size,
                    **generation_kwargs,
                )
                for worker, worker_input in zip(self.workers, worker_inputs)
                if len(worker_input) > 0
            ]
        )

        return [request_output.outputs for request_output in self.dist_layout.gather_data(outputs, context="rollout")]

    def _get_reward_metrics(
        self,
        prefix: str,
        completions: List[List[CompletionOutput]],
        calculated_rewards: List[List[Dict[str, float]]],
        data_sources: Optional[List[Union[str, None]]] = None,
    ) -> Dict[str, float]:
        prefix = prefix or ""
        data_sources = data_sources or [""] * len(completions)
        scores = torch.tensor(
            [
                [repetition["total_reward"] for repetition in question_completions]
                for question_completions in calculated_rewards
            ],
            dtype=torch.float32,
        )

        # Assumes a 0.5 threshold for correctness
        correct_questions = scores > 0.5
        generation_lengths = torch.tensor(
            [
                [len(gen_output.token_ids) for gen_output in question_completions]
                for question_completions in completions
            ],
            dtype=torch.float32,
        )
        # Calculates statistics for each data source
        metrics = {}
        dedup_data_sources = list(dict.fromkeys(data_sources))

        for data_source in dedup_data_sources:
            s_mask = [ds == data_source for ds in data_sources]
            s_prefix = f"{prefix}_reward/{data_source + '-' if data_source else ''}"
            s_scores, s_lengths, s_correct = [
                array[s_mask] for array in [scores, generation_lengths, correct_questions]
            ]

            metrics.update(
                {
                    f"{s_prefix}mean": s_scores.mean().item(),
                    f"{s_prefix}best_of_group": s_scores.max(axis=1).values.mean().item(),
                    f"{s_prefix}accuracy": s_correct.float().mean().item(),
                    f"{s_prefix}avg_response_length": s_lengths.mean().item(),
                    f"{s_prefix}avg_correct_response_length": s_lengths[s_correct].mean().item(),
                    f"{s_prefix}avg_incorrect_response_length": s_lengths[~s_correct].mean().item(),
                }
            )

        return metrics

    def _calculate_advantages(self, calculated_rewards: List[List[Dict[str, float]]]) -> List[List[float]]:
        total_rewards = torch.tensor(
            [
                [repetition["total_reward"] for repetition in question_completions]
                for question_completions in calculated_rewards
            ],
            dtype=torch.float32,
        )

        advantages = total_rewards - total_rewards.mean(axis=1, keepdims=True)
        if self.args.normalize_advantage_std:
            advantages = advantages / (total_rewards.std(axis=1, keepdim=True) + 1e-4)

        return advantages.tolist()

    def _pack_completions(
        self,
        prompt_input_ids: List[List[int]],
        completions: List[List[CompletionOutput]],
        advantages: List[List[float]],
    ) -> Dict[int, List[PackedBatch]]:
        interaction_tokens, interaction_masks = [], []

        for prompt_tokens, gen_outputs in zip(prompt_input_ids, completions):
            for output in gen_outputs:
                interaction_tokens.append(list(prompt_tokens) + list(output.token_ids))
                interaction_masks.append([False] * len(prompt_tokens) + [True] * len(output.token_ids))

        return distribute_and_pack_sequences(
            interaction_tokens,
            interaction_masks,
            list(itertools.chain(*advantages)),
            self.args.rollout.prompt_length + self.args.rollout.response_length,
            dp_size=self.dist_layout.actor_dp_size,
            cp_size=self.dist_layout.actor_cp_size,
            tp_size=self.dist_layout.actor_tp_size,
            pad_token_id=self.tokenizer.pad_token_id,
            micro_batch_size=self.args.train_max_micro_batch_size_per_gpu,
            pad_to_largest_micro_batch=True,
        )

    def _update_actor_policy(
        self, packed_batches: Dict[int, List[PackedBatch]], temperature: float
    ) -> List[Dict[str, float | int]]:
        futures = [
            worker.update_actor_policy.remote(packed_batches[i], temperature=temperature)
            for i, worker in enumerate(self.workers)
        ]
        metrics = ray.get(futures)

        return {
            k: None if metrics[0][k] is None else sum([m[k] for m in metrics]) / len(metrics) for k in metrics[0].keys()
        }

    def save_checkpoint(self, output_dir: Union[str, Path]) -> None:
        """Save the current state of the tuner.

        Args:
            output_dir: Directory to save the checkpoint.

        """

        # Save the current step as the latest
        output_dir = Path(output_dir)
        open(output_dir / "latest", "w").write(str(self.global_step))

        checkpoint_dir = output_dir / str(self.global_step)
        actor_dir = checkpoint_dir / "actor"

        actor_dir.mkdir(exist_ok=True, parents=True)
        ray.get([worker.save_actor_model.remote(actor_dir) for worker in self.workers])

        # `total_consumed_samples` is `batch_size * (step - 1)` because we are saving the checkpoint before
        # actually training on the samples from the current step
        trainer_state = {
            "global_step": self.global_step,
            "total_consumed_samples": (self.global_step - 1) * self.args.train_batch_size,
            "dataloader_shuffle": self.args.dataloader_shuffle,
            "seed": self.args.seed,
        }

        save_json_file(trainer_state, checkpoint_dir / "trainer_state.json")
        save_json_file(self.args.to_dict(json_serialize=True), output_dir / "config.json")

    def load_checkpoint(
        self,
        load_dir: str,
        tag: Optional[int] = None,
        load_optimizer_states: bool = True,
        load_scheduler_states: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint from the specified directory.

        Args:
            load_dir: Directory to load the checkpoint from.
            tag: Tag of the checkpoint to load.
                If set to ``None``, it will load the latest checkpoint available in the directory.
            load_optimizer_states: Whether to load optimizer states from the checkpoint.
            load_scheduler_states: Whether to load learning rate scheduler states from the checkpoint.

        Returns:
            Trainer state dictionary containing the global step, total consumed samples,
            dataloader shuffle state, and seed used for training.

        """

        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise FileNotFoundError(f"'{load_dir}' is not a valid path.")

        if tag is None:
            if not (load_dir / "latest").exists():
                raise FileNotFoundError(f"`tag` has not been provided and no 'latest' file found in '{load_dir}'.")
            tag = int(open(load_dir / "latest").read())

        checkpoint_dir = load_dir / str(tag)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"'{checkpoint_dir}' is not a valid path.")
        trainer_state = load_json_file(checkpoint_dir / "trainer_state.json")

        # Load actor model checkpoint and, possibly, optimizer states
        ray.get(
            [
                worker.load_actor_model.remote(checkpoint_dir / "actor", load_optimizer_states, load_scheduler_states)
                for worker in self.workers
            ]
        )

        return trainer_state

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the evaluation dataset.

        Returns:
            Evaluation metrics containing the rewards and other statistics.

        """

        n_eval_samples = len(self.eval_dataset)
        eval_samples = list(self.eval_dataset)

        raw_prompts = [item["raw_prompt"] for item in eval_samples]
        ground_truth = [item["ground_truth"] for item in eval_samples]
        prompt_input_ids = [item["prompt_input_ids"] for item in eval_samples]

        data_sources = None
        if "dataset_name" in eval_samples[0]:
            data_sources = [item["dataset_name"] for item in eval_samples]

        self.logger.info("Generating evaluation completions...")
        eval_completions = self.generate_completions(
            prompt_input_ids,
            num_repetitions=self.args.group_size,
            batch_size=self.args.train_batch_size * 2,
            sync_weights=True,
            detokenize=True,
            use_tqdm=True,
        )
        self.logger.info("Completions generated.")

        self.logger.info("Calculating evaluation rewards...")
        eval_rewards = self.rewards.score(
            eval_completions,
            ground_truth,
            reward_names=[getattr(self.data_collator, "reward_names"), ""] * n_eval_samples,
            reward_weights=[getattr(self.data_collator, "reward_weights", 1.0)] * n_eval_samples,
        )
        eval_total_rewards = [
            [repetition["total_reward"] for repetition in question_completions] for question_completions in eval_rewards
        ]
        self.logger.info("Rewards calculated.")

        if self.table_logger and self.args.log_n_eval_completions > 0:
            self.logger.info("Logging evaluation completions...")
            self.table_logger.add_data(
                [
                    [
                        self.global_step,
                        question_idx,
                        repetition_idx,
                        data_sources[question_idx] if data_sources else None,
                        raw_prompts[question_idx],
                        ground_truth[question_idx],
                        completion.text,
                        eval_total_rewards[question_idx][repetition_idx],
                    ]
                    for question_idx in range(min(self.args.log_n_eval_completions, n_eval_samples))
                    for repetition_idx, completion in enumerate(eval_completions[question_idx])
                ]
            )
            self.logger.info("Evaluation completions logged.")

        return self._get_reward_metrics("validation", eval_completions, eval_rewards, data_sources=data_sources)

    @ray_wandb_graceful_shutdown
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        checkpoint_tag: Optional[int] = None,
        resume_optimizer_states: bool = True,
        resume_lr_scheduler_states: bool = True,
        resume_dataset_states: bool = True,
    ) -> None:
        """Train a model.

        Args:
            resume_from_checkpoint: Resume training from a specific checkpoint.
                If set to ``None``, training will start from scratch.
                If different than ``None``, training will resume from the checkpoint.
            checkpoint_tag: Resume training from a specific tag/step.
                If set to ``None``, it will resume from the latest checkpoint.
                If different than ``None``, training will resume from that tag.
            resume_optimizer_states: Whether to resume optimizer state from checkpoint.
                Only works if ``resume_from_checkpoint`` is provided.
            resume_lr_scheduler_states: Whether to resume learning rate scheduler state from checkpoint.
                Only works if ``resume_from_checkpoint`` is provided.
            resume_dataset_states: Whether to resume the dataset state from the checkpoint.
                Only works if ``resume_from_checkpoint`` is provided.

        """

        self.logger.info("Starting training...")

        self.global_step = 1
        total_consumed_samples = 0
        timer = Timer()

        self._init_workers()

        if resume_from_checkpoint:
            if not is_checkpoint_available(resume_from_checkpoint, checkpoint_tag):
                self.logger.info(
                    f"Either {checkpoint_tag} or 'latest' checkpoint not found in '{resume_from_checkpoint}'."
                )
            else:
                self.logger.info(f"{checkpoint_tag} or 'latest' checkpoint found in '{resume_from_checkpoint}'.")

                checkpoint_state = self.load_checkpoint(
                    resume_from_checkpoint,
                    tag=checkpoint_tag,
                    load_optimizer_states=resume_optimizer_states,
                    load_scheduler_states=resume_lr_scheduler_states,
                )

                if resume_dataset_states:
                    if checkpoint_state["seed"] != self.args.seed:
                        raise ValueError(
                            f"Checkpoint and configuration `seed` must be equal, but got {checkpoint_state['seed']} and {self.args.seed}."
                        )
                    if checkpoint_state["dataloader_shuffle"] != self.args.dataloader_shuffle:
                        raise ValueError(
                            f"Checkpoint and configuration `dataloader_shuffle` must be equal, but got {checkpoint_state['dataloader_shuffle']} and {self.args.dataloader_shuffle}."
                        )

                    total_consumed_samples = checkpoint_state["total_consumed_samples"]
                    checkpoint_tag = checkpoint_state["global_step"]
                    self.global_step = checkpoint_state["global_step"]

        train_dataloader = self.get_train_dataloader(
            total_consumed_samples=total_consumed_samples, shuffle=self.args.dataloader_shuffle
        )
        train_iterator = RepeatingLoader(train_dataloader, use_batch_tracker=False)
        packed_batches = None

        for step in tqdm(range(self.global_step, self.training_steps + 1)):
            timer.start()

            eval_reward_metrics = {}
            if self._is_eval_step(step):
                with timer.measure("timing/evaluation"):
                    self.logger.info("Evaluating model...")
                    eval_reward_metrics = self.evaluate()
                    self.logger.info("Evaluation done.")

            if self._is_checkpoint_save_step(step, checkpoint_tag):
                with timer.measure("timing/checkpoint"):
                    self.logger.info("Saving checkpoint...")
                    self.save_checkpoint(self.args.output_dir)
                    self.logger.info("Checkpoint saved (or scheduled to be saved).")

            train_reward_metrics = {}
            if self._is_generation_step(step) or packed_batches is None:
                batch = next(train_iterator)

                self.logger.info(f"Generating completions for {self.args.generation_interval} steps...")
                with timer.measure("timing/generation"):
                    completions = self.generate_completions(
                        batch["prompt_input_ids"], self.args.group_size, sync_weights=True, detokenize=True
                    )
                self.logger.info("Completions generated.")

                self.logger.info("Calculating rewards...")
                with timer.measure("timing/reward"):
                    rewards = self.rewards.score(
                        completions, batch["ground_truth"], batch["reward_names"], batch["reward_weights"]
                    )
                    train_reward_metrics = self._get_reward_metrics("train", completions, rewards)
                    advantages = self._calculate_advantages(rewards)
                self.logger.info("Rewards calculated.")

                self.logger.info("Packing completions, masks and advantages...")
                with timer.measure("timing/packing"):
                    packed_batches = self._pack_completions(batch["prompt_input_ids"], completions, advantages)
                self.logger.info("Packing done.")

            # Select the micro-batches for the current step
            sft_step = step % self.args.generation_interval
            sft_micro_batches = len(packed_batches[0]) // self.args.generation_interval

            current_sft_batch = {
                gpu_id: packed_batch[(sft_micro_batches) * sft_step : (sft_micro_batches) * (sft_step + 1)]
                for gpu_id, packed_batch in packed_batches.items()
            }

            self.logger.info("Updating actor model...")
            with timer.measure("timing/actor_update"):
                actor_metrics = self._update_actor_policy(
                    current_sft_batch, temperature=self.args.rollout.sampling_params["temperature"]
                )
            self.logger.info("Actor model updated.")

            self.logger.info(
                {
                    "train/step": step,
                    "train/epoch": step // len(train_dataloader) + 1,
                    **actor_metrics,
                    **train_reward_metrics,
                    **eval_reward_metrics,
                    **timer.metrics(total_key="timing/isft_step", end=True),
                }
            )

            self.global_step += 1

        self.logger.info("Training done.")
