# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import queue
import time
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import deepspeed
import torch
from deepspeed.runtime import lr_schedules
from deepspeed.utils import groups
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from tqdm import tqdm

from phyagi.datasets.concat_dataset import WeightedConcatIterableDataset
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
)
from phyagi.models.mixformer_sequential.parallel_mixformer_sequential import (
    apply_cp_mixformer_sequential,
)
from phyagi.models.parallel_utils import maybe_apply_context_parallel_to_inputs
from phyagi.optimizers.schedulers.warmup_decay import WarmupDecayCooldownLR
from phyagi.trainers.ds import ds_mpu
from phyagi.trainers.ds.ds_pipeline_module import DsPipelineModule
from phyagi.trainers.ds.ds_trainer_callback import DsCallbackHandler, DsTrainerCallback
from phyagi.trainers.ds.ds_training_args import DsTrainingArguments
from phyagi.trainers.flops_utils import estimate_tflops, get_peak_tflops
from phyagi.trainers.trainer_utils import (
    BatchTracker,
    RepeatingLoader,
    StatefulDistributedSampler,
)
from phyagi.utils.file_utils import save_json_file, save_jsonl_file
from phyagi.utils.logging_handlers import (
    MlflowHandler,
    QueueHandler,
    TensorBoardHandler,
    WandbHandler,
)
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import rgetattr

# Register custom scheduler in DeepSpeed
setattr(lr_schedules, "WarmupDecayCooldownLR", WarmupDecayCooldownLR)

logger = get_logger(__name__)


class DsTrainer:
    """DeepSpeed trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[DsTrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        sampler: Optional[Sampler] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_parameters: Optional[Union[Iterable[torch.Tensor], Dict[str, torch.Tensor]]] = None,
        mpu: Optional[Any] = None,
        dist_init_required: Optional[bool] = None,
        callbacks: Optional[List[DsTrainerCallback]] = None,
        optimizers: Optional[Tuple[Optimizer, LRScheduler]] = None,
    ) -> None:
        """Initialize the DeepSpeed trainer.

        The initialization ensure that the training arguments and DeepSpeed configuration
        are properly set up. It also initializes the pipeline parallelism (if needed) and
        the DeepSpeed engine.

        Args:
            model: Model to be trained or evaluated.
            args: DeepSpeed training arguments.
                If set to ``None``, will use a default instance of :class:`phyagi.trainers.ds.ds_training_args.DsTrainingArguments`.
            data_collator: Collate function used for creating a batch from ``train_dataset`` and ``eval_dataset``.
            sampler: Sampler used for sampling ``train_dataset`` and ``eval_dataset``.
                If set to ``None``, will use :class:`phyagi.trainers.trainer_utils.StatefulDistributedSampler`.
            train_dataset: Dataset used for training.
                If set to ``None``, :meth:`train` will not be available.
            eval_dataset: Dataset used for evaluation.
                If set to ``None``, will not perform evaluation.
            model_parameters: Model parameters to be used for training.
                If set to ``None``, will use all trainable parameters in the model.
            mpu: Model parallelism unit object that implements ``get_{model,data}_parallel_{rank,group,world_size}()``.
                If set to ``None``, uses ``model.mpu()``.
            dist_init_required: Auto-initializes the torch distributed if needed.
                If different than ``None``, will force the torch distributed to be initialized or not.
            callbacks: Optional callbacks to be used.
            optimizers: Tuple of ``(optimizer, lr_scheduler)`` to be used for training.
                If set to ``None``, will use the optimizer and scheduler defined in DeepSpeed configuration.

        """

        self._model_config = getattr(model, "config", None)
        self._model_cls = getattr(model, "__class__", None)
        self._model_type = rgetattr(model, "config.model_type", None)
        self._cp_size = rgetattr(model, "config.cp_size", 1)

        args = args or DsTrainingArguments("tmp")
        if not isinstance(args, DsTrainingArguments):
            raise TypeError(f"`args` must be an instance of DsTrainingArguments, but got '{type(args)}'.")
        self._args = args

        self._data_collator = data_collator
        self._sampler = sampler
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset

        self._args.do_eval = self._args.do_eval if self._eval_dataset is not None else False
        self._peak_tflops = get_peak_tflops()

        callbacks = callbacks or []
        if not isinstance(callbacks, list):
            raise TypeError(f"`callbacks` must be a list, but got '{type(callbacks)}'.")
        self._callback_handler = DsCallbackHandler(callbacks)

        optimizers = optimizers or (None, None)
        if not (isinstance(optimizers, tuple) and len(optimizers) == 2):
            raise ValueError(f"`optimizers` must be a tuple of length 2, but got {optimizers}.")
        optimizer, lr_scheduler = optimizers

        # Context Parallelism (SP)
        if self._args.context_parallel_size > 1:
            if self._cp_size != self._args.context_parallel_size:
                raise ValueError(
                    f"`cp_size` must be equal to {self._args.context_parallel_size}, but got {self._cp_size}."
                )
            if self._model_type != MIXFORMER_SEQUENTIAL_MODEL_TYPE:
                raise ValueError(
                    f"`model_type` must be '{MIXFORMER_SEQUENTIAL_MODEL_TYPE}', but got '{self._model_type}'."
                )

            # Use a custom `mpu` implementation to avoid overriding DeepSpeed-based implementations
            # and to ensure that parallelism groups are set up correctly
            mpu = ds_mpu
            mpu.initialize(context_parallel_size=self._args.context_parallel_size)

        # Pipeline Parallelism (PP)
        if self._args.pipe_parallel_size > 1:
            if getattr(model, "loss", None) is None:
                raise ValueError("`model` must have a `loss` layer, but got None.")

            seq_model = None
            for layer in model.modules():
                if isinstance(layer, torch.nn.Sequential):
                    logger.info(f"{layer} is a torch.nn.Sequential instance, setting it as the pipeline model.")
                    seq_model = layer
                    break
            if not isinstance(seq_model, torch.nn.Sequential):
                raise TypeError(f"`model` must be an instance of torch.nn.Sequential, but got '{type(model)}'.")

            model = DsPipelineModule(
                layers=seq_model,
                num_stages=self._args.pipe_parallel_size,
                loss_fn=model.loss,
                partition_method=self._args.pipe_parallel_partition_method,
                activation_checkpoint_interval=self._args.pipe_parallel_activation_checkpoint_steps,
            )

        # Prepare DeepSpeed engine by adjusting the number of steps and applying last-minute changes
        self._convert_epochs_to_max_steps()
        self._convert_steps_to_int()
        self._prepare_ds_config()

        # Initialize the engine and client state
        self.engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters or [p for p in model.parameters() if p.requires_grad],
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            config=OmegaConf.to_object(self._args.ds_config),
        )

        # After engine has been initialized and processes groups are available,
        # ensure any remaining parallelism configuration is set up
        if self._args.context_parallel_size > 1:
            apply_cp_mixformer_sequential(self.engine.module, cp_mesh=self.engine.get_sequence_parallel_group())
        if self._args.pipe_parallel_size > 1:
            self.engine.set_batch_fn(lambda x: tuple(x.values()))

        train_batch_size = (
            self._args.train_batch_size
            if self._args.train_batch_size_init_rampup == 0
            else self._args.train_batch_size_init_rampup
        )
        self.client_state = {
            "global_epoch": 0,
            "train_batch_size": train_batch_size,
            "checkpoint_history": [],
            "log_history": [],
        }

        if self.engine.global_rank == 0:
            config = {
                "model_config": self._model_config.to_dict() if self._model_config else None,
                "training_args": self._args.to_dict(json_serialize=True),
            }

            log_handlers = []
            if self._args.mlflow:
                run_id = os.getenv("MLFLOW_RUN_ID", None)
                experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", None)
                log_handlers.append(MlflowHandler(run_id=run_id, experiment_id=experiment_id, tags=config))
            if self._args.wandb:
                log_handlers.append(
                    WandbHandler(
                        key=self._args.wandb_api_key,
                        host=self._args.wandb_host,
                        config=config,
                        output_dir=self._args.log_dir,
                    )
                )
            if self._args.tensorboard:
                log_handlers.append(TensorBoardHandler(log_dir=self._args.log_dir))

            self._listener = None
            if len(log_handlers) > 0:
                log_queue = queue.Queue(-1)
                logger.addHandler(QueueHandler(log_queue))

                self._listener = QueueListener(log_queue, *log_handlers, respect_handler_level=True)
                self._listener.start()

        logger.info(f"Model: {model}")
        logger.info(f"Training arguments: {self._args.to_dict(json_serialize=True)}")

    @property
    def _data_parallel_world_size(self) -> int:
        """Return the data parallel world size."""

        return groups._get_data_parallel_world_size()

    @property
    def _data_parallel_rank(self) -> int:
        """Return the data parallel rank of the current process."""

        return groups._get_data_parallel_rank()

    @property
    def _context_parallel_world_size(self) -> int:
        """Return the context parallel world size."""

        return groups._get_sequence_parallel_world_size()

    @property
    def _context_parallel_rank(self) -> int:
        """Return the context parallel rank of the current process."""

        return groups._get_sequence_parallel_rank()

    @property
    def seq_len(self) -> int:
        """Return the sequence length used for training."""

        return getattr(self._train_dataset, "seq_len", 1) if self._train_dataset is not None else 1

    def _is_log_step(self, step: int) -> bool:
        do_periodic_log = step % self._args.logging_steps == 0
        do_final_log = step == self._args.max_steps

        return do_periodic_log or do_final_log

    def _is_batch_tracker_save_step(self, step: int) -> bool:
        do_periodic_save = step % self._args.batch_tracker_save_steps == 0
        do_final_save = step == self._args.max_steps

        return do_periodic_save or do_final_save

    def _is_eval_step(self, step: int) -> bool:
        do_periodic_eval = step % self._args.eval_steps == 0
        do_final_eval = step == self._args.max_steps and self._args.do_final_eval

        return (do_periodic_eval or do_final_eval) and self._args.do_eval

    def _is_checkpoint_save_step(self, step: int) -> bool:
        do_periodic_save = step % self._args.save_steps == 0
        do_final_save = step == self._args.max_steps and self._args.save_final_checkpoint

        return do_periodic_save or do_final_save

    def _convert_epochs_to_max_steps(self) -> None:
        if self._args.max_steps < 0 and self._train_dataset:
            if not hasattr(self._train_dataset, "__len__"):
                raise ValueError("`max_steps` must be defined when `train_dataset` does not implement `__len__()`.")

            n_steps_per_epoch = len(self._train_dataset) // self._args.train_batch_size
            self._args.max_steps = int(n_steps_per_epoch * self._args.num_train_epochs)

            logger.warning(
                f"`max_steps` not provided. Setting to: {n_steps_per_epoch} * {self._args.num_train_epochs} = {self._args.max_steps} steps."
            )

    def _convert_steps_to_int(self) -> None:
        if isinstance(self._args.logging_steps, float):
            self._args.logging_steps = math.ceil(self._args.logging_steps * self._args.max_steps)
            logger.warning(f"`logging_steps` is a ratio. Converting to {self._args.logging_steps} steps.")
        if isinstance(self._args.save_steps, float):
            self._args.save_steps = math.ceil(self._args.save_steps * self._args.max_steps)
            logger.warning(f"`save_steps` is a ratio. Converting to {self._args.save_steps} steps.")
        if isinstance(self._args.eval_steps, float):
            self._args.eval_steps = math.ceil(self._args.eval_steps * self._args.max_steps)
            logger.warning(f"`eval_steps` is a ratio. Converting to {self._args.eval_steps} steps.")
        if isinstance(self._args.batch_tracker_save_steps, float):
            self._args.batch_tracker_save_steps = math.ceil(self._args.batch_tracker_save_steps * self._args.max_steps)
            logger.warning(
                f"`batch_tracker_save_steps` is a ratio. Converting to {self._args.batch_tracker_save_steps} steps."
            )

    def _prepare_ds_config(self) -> None:
        # If scheduler `total_num_steps` has not been provided,
        # update it to the value of `max_steps`
        if OmegaConf.select(self._args.ds_config, "scheduler.type") in ["WarmupDecayLR", "WarmupDecayCooldownLR"]:
            if OmegaConf.select(self._args.ds_config, "scheduler.params.total_num_steps") is None:
                OmegaConf.update(self._args.ds_config, "scheduler.params.total_num_steps", self._args.max_steps)
                logger.warning(
                    f"`scheduler.params.total_num_steps` not provided. Setting to {self._args.max_steps} steps."
                )

    def _rampup_train_batch_size(self, step: int) -> None:
        if self._args.train_batch_size - self.client_state["train_batch_size"] > 0:
            if step % self._args.rampup_steps == 0:
                self.client_state["train_batch_size"] += self._args.train_batch_size_per_rampup
                if self.client_state["train_batch_size"] > self._args.train_batch_size:
                    self.client_state["train_batch_size"] = self._args.train_batch_size

                self.engine.set_train_batch_size(self.client_state["train_batch_size"])
                logger.info(f"`train_batch_size` ramping to {self.client_state['train_batch_size']}.")

    def _accumulate_batch_tracker_metrics(
        self, metrics: List[Dict[str, Any]], batch_tracker: BatchTracker, step: int, train_step_time: float, loss: float
    ) -> None:
        step_metrics = {
            "train/global_rank": self.engine.global_rank,
            "train/step": step,
            "train/step_runtime": train_step_time / self.engine.world_size,
            "train/loss": loss,
        }

        # When using Pipeline Parallelism, each pipeline is responsible for retrieving
        # batches from the dataset (instead of each process / GPU)
        if self._args.pipe_parallel_size > 1:
            # Since the same batch can be used across multiple stages, we use the first
            # stage to track the number of samples
            if self.engine.is_first_stage():
                step_metrics.update(
                    {
                        "train/n_pipelines": self.engine.world_size // self._args.pipe_parallel_size,
                        "train/n_samples_per_dataset_per_pipeline": batch_tracker.n_samples_per_dataset,
                    }
                )
        else:
            step_metrics.update(
                {
                    "train/n_samples_per_dataset": batch_tracker.n_samples_per_dataset,
                }
            )

        metrics.append(step_metrics)
        batch_tracker.reset()

    def _calculate_throughput_metrics(self, time: float, label: str = "train") -> Dict[str, float]:
        samples_per_second = self.client_state["train_batch_size"] / time
        tokens_per_second = samples_per_second * self.seq_len
        tflops = estimate_tflops(
            time,
            n_layer=getattr(self._model_config, "n_layer", 1),
            n_embd=getattr(self._model_config, "n_embd", 1),
            vocab_size=getattr(self._model_config, "vocab_size", 1),
            seq_len=self.seq_len,
            batch_size=self.client_state["train_batch_size"],
            activation_checkpointing=getattr(self._model_config, "gradient_checkpointing", False),
        )

        metrics = {
            "step_runtime": time,
            "samples_per_second": samples_per_second,
            "samples_per_second_per_gpu": samples_per_second / self.engine.world_size,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second / self.engine.world_size,
            "tflops": tflops,
            "tflops_per_gpu": tflops / self.engine.world_size,
            "mfu": tflops / self.engine.world_size / self._peak_tflops,
        }

        return {f"{label}/{k}": v for k, v in metrics.items()}

    def get_dataloader(
        self,
        dataset: Dataset,
        sampler: Optional[Sampler] = None,
        shuffle: bool = False,
        epoch: int = 0,
        total_consumed_samples: int = 0,
    ) -> DataLoader:
        """Get a data loader for a given dataset.

        This method ensures that the data loader is properly set up for distributed training. If ``dataset`` is not an
        instance of :class:`torch.utils.data.IterableDataset`, it will use use ``sampler`` as
        :class:`phyagi.trainers.trainer_utils.StatefulDistributedSampler` to ensure that the
        data loader is stateful and shuffled across processes.

        Args:
            dataset: Dataset to be used.
            sampler: Sampler to be used.
            shuffle: Whether to shuffle the dataset.
            epoch: Epoch to be used for the sampler.
            total_consumed_samples: Total number of samples consumed by the model.

        Returns:
            A data loader.

        """

        if sampler is None and not isinstance(dataset, IterableDataset):
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=self._data_parallel_world_size,
                rank=self._data_parallel_rank,
                shuffle=shuffle,
                seed=self._args.seed,
                epoch=epoch,
                total_consumed_samples=total_consumed_samples,
            )

        if isinstance(dataset, WeightedConcatIterableDataset):
            dataset.set_rank_and_world_size(self._data_parallel_rank, self._data_parallel_world_size)
            if self._args.dataloader_num_workers != 0 or self._args.dataloader_prefetch_factor is not None:
                raise ValueError(
                    f"`dataloader_num_workers` must be == 0 and `dataloader_prefetch_factor` must be None, but got {self._args.dataloader_num_workers} and {self._args.dataloader_prefetch_factor}."
                )

        return DataLoader(
            dataset,
            batch_size=self.engine.train_micro_batch_size_per_gpu(),
            sampler=sampler,
            num_workers=self._args.dataloader_num_workers,
            pin_memory=self._args.dataloader_pin_memory if not isinstance(dataset, IterableDataset) else True,
            drop_last=True,
            collate_fn=self._data_collator,
            prefetch_factor=self._args.dataloader_prefetch_factor,
        )

    def load_checkpoint(
        self,
        load_dir: Union[str, Path],
        tag: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
        load_dataset_states: bool = True,
    ) -> None:
        """Load a checkpoint using :meth:`deepspeed.DeepSpeedEngine.load_checkpoint`.

        Args:
            load_dir: Path to the directory holding the checkpoint to be loaded.
            tag: Tag to be used for loading the checkpoint.
                If ``None``, the "latest" file with the checkpoint number is loaded.
            load_optimizer_states: Whether to load optimizer states.
            load_lr_scheduler_states: Whether to load learning rate scheduler states.
            load_dataset_states: Wheter to load dataset states for resuming training.
        """

        load_dir = Path(load_dir)

        # If `tag` is None, try to read from latest file
        if tag is None:
            latest_file_path = load_dir / "latest"
            if latest_file_path.is_file():
                with open(latest_file_path, "r") as f:
                    tag = f.read().strip()

        # Try to load checkpoint from given tag
        # If checkpoint is corrupted, reduce step number and try again
        for i in range(self._args.load_checkpoint_num_tries):
            logger.info(f"Trying to load checkpoint from step {tag}...")
            error_flag = torch.tensor(0, dtype=torch.int32, device=self.engine.device)

            try:
                _, client_state = self.engine.load_checkpoint(
                    load_dir,
                    tag=tag,
                    load_optimizer_states=load_optimizer_states,
                    load_lr_scheduler_states=load_lr_scheduler_states,
                )
            except:
                error_flag = torch.tensor(1, dtype=torch.int32, device=self.engine.device)
                logger.warning(f"Loading checkpoint failed on rank {self.engine.global_rank} for step {tag}.")

            # Check if loading checkpoint was successful on all ranks
            # If not, reduce the step number
            torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.MAX)
            if error_flag.item() > 0:
                client_state = None

                # If `tag` is not None, try to reduce it to the previous checkpoint number
                if tag is not None:
                    new_tag = int(tag) - self._args.save_steps
                    if new_tag > 0 and i < (self._args.load_checkpoint_num_tries - 1):
                        logger.info(f"Reducing tag from {tag} to {new_tag}...")
                        tag = str(new_tag)
                    else:
                        raise Exception(
                            "Maximum number of tries or `tag < 0`. Aborting due to partially loaded checkpoint..."
                        )
                else:
                    raise Exception("Aborting due to partially loaded checkpoint since `tag` is None...")
            else:
                break

        # Load dataset state for streaming datasets
        if isinstance(self._train_dataset, WeightedConcatIterableDataset):
            if client_state is not None and load_dataset_states:
                self._train_dataset.set_rank_and_world_size(self._data_parallel_rank, self._data_parallel_world_size)
                self._train_dataset.load_state(load_dir, tag)

        # If `client_state` is not None, we update the class `client_state`
        if client_state is not None:
            for key in self.client_state:
                if key in client_state:
                    self.client_state[key] = client_state[key]
                else:
                    logger.warning(f"{key} not found in DeepSpeed checkpoint `client_state`.")

    def save_checkpoint(self, step: int, epoch: int = 0) -> None:
        """Save a checkpoint using :meth:`deepspeed.DeepSpeedEngine.save_checkpoint`.

        This method also saves the ``trainer_state``, ``training_args`` and ``model_config``
        (if available) to the output directory.

        Args:
            step: Step to be saved.
            epoch: Epoch to be saved.

        """

        # Save the checkpoint and ensure it is added to the `checkpoint_history`
        self.client_state["global_epoch"] = epoch
        self.engine.save_checkpoint(self._args.output_dir, step, client_state=self.client_state)

        checkpoint_output_dir = self._args.output_dir / str(step)
        self.client_state["checkpoint_history"].append({"path": str(checkpoint_output_dir), "train/step": step})

        # Save dataset state for streaming datasets
        if isinstance(self._train_dataset, WeightedConcatIterableDataset):
            self._train_dataset.save_state(self._args.output_dir, str(step))

        # Prevent non-main processes from saving `trainer_state`, `training_args`,
        # and `model_config`
        if self.engine.global_rank == 0:
            # Parse only the necessary information for `trainer_state`,
            # because `client_state` might contain information that cannot be serialized
            non_serializable_keys = [
                "buffer_names",
                "param_shapes",
                "frozen_param_shapes",
                "shared_params",
                "frozen_param_fragments",
                "ds_config",
                "ds_version",
            ]
            trainer_state = {k: v for k, v in self.client_state.items() if k not in non_serializable_keys}

            save_json_file(trainer_state, self._args.output_dir / "trainer_state.json")
            save_json_file(self._args.to_dict(json_serialize=True), self._args.output_dir / "training_args.json")

            if self._model_config:
                self._model_config.save_pretrained(checkpoint_output_dir)

    def _train_batch_without_pipe_parallel(self, data_iter: Optional[Iterator] = None) -> torch.Tensor:
        gradient_accumulation_steps = self.engine.gradient_accumulation_steps()

        total_loss = torch.tensor(0.0, device=self.engine.device)

        for _ in range(gradient_accumulation_steps):
            batch = next(data_iter)

            batch = {k: v.to(self.engine.device) for k, v in batch.items()}
            batch = maybe_apply_context_parallel_to_inputs(
                batch,
                context_parallel_world_size=self._context_parallel_world_size,
                context_parallel_rank=self._context_parallel_rank,
            )

            outputs = self.engine(**batch)
            loss = outputs.loss.mean()

            self.engine.backward(loss)
            self.engine.step()

            total_loss += loss

        total_loss /= gradient_accumulation_steps

        return total_loss

    def train_step(self, train_iterator: Iterator) -> torch.Tensor:
        """Perform a training step.

        If ``pipe_parallel_size`` is greater than 1, this method will use
        :meth:`deepspeed.runtime.pipe.engine.PipelineEngine.train_batch` to perform a training step.
        Otherwise, it will use :meth:`_train_batch_without_pipe_parallel`.

        Args:
            train_iterator: Training iterator.

        Returns:
            Training loss.

        """

        if self._args.pipe_parallel_size > 1:
            return self.engine.train_batch(data_iter=train_iterator)

        return self._train_batch_without_pipe_parallel(data_iter=train_iterator)

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        checkpoint_tag: Optional[str] = None,
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

        logger.info("Starting training...")

        if resume_from_checkpoint:
            self.load_checkpoint(
                resume_from_checkpoint,
                tag=checkpoint_tag,
                load_optimizer_states=resume_optimizer_states,
                load_lr_scheduler_states=resume_lr_scheduler_states,
                load_dataset_states=resume_dataset_states,
            )

        # Ensure that the initial `train_batch_size` is correctly set
        self.engine.set_train_batch_size(self.client_state["train_batch_size"])
        batch_tracker_metrics = []

        if self._train_dataset is None:
            raise ValueError("`train_dataset` must be defined, but got None.")

        n_steps_per_epoch = 1
        if hasattr(self._train_dataset, "__len__"):
            n_steps_per_epoch = len(self._train_dataset) // self._args.train_batch_size or 1

        train_dataloader = self.get_dataloader(
            self._train_dataset,
            shuffle=self._args.dataloader_shuffle,
            epoch=self.client_state["global_epoch"],
            total_consumed_samples=self.engine.global_samples,
            sampler=self._sampler,
        )
        train_iterator = iter(RepeatingLoader(train_dataloader, use_batch_tracker=self._args.batch_tracker))
        train_time = time.time()

        for step in tqdm(
            range(self.engine.global_steps + 1, self._args.max_steps + 1), disable=not self._args.is_main_process
        ):
            # `step_time` tracks the whole step (including checkpointing, batch tracking, etc)
            # `inner_step_time` tracks the `train_step()` function
            step_time, inner_step_time = time.time(), time.time()
            metrics = {}

            loss = self.train_step(train_iterator)
            inner_step_time = time.time() - inner_step_time

            if self._args.batch_tracker:
                self._accumulate_batch_tracker_metrics(
                    batch_tracker_metrics,
                    train_iterator.batch_tracker,
                    step,
                    inner_step_time,
                    loss.item(),
                )

                if self._is_batch_tracker_save_step(step):
                    output_file_path = self._args.output_dir / f"batch_tracker_rank_{self.engine.global_rank:02d}.jsonl"

                    # Reset `batch_tracker_metrics` to prevent a memory leak
                    save_jsonl_file(batch_tracker_metrics, output_file_path, mode="a")
                    batch_tracker_metrics = []

            # First logging block is used to store inner step throughput metrics
            if self._is_log_step(step):
                metrics.update({**self._calculate_throughput_metrics(inner_step_time, label="step")})

            torch.distributed.reduce(loss, 0)

            # Checkpoint block is used to save the model and store the saving runtime
            if self._is_checkpoint_save_step(step):
                ckp_time = time.time()

                self.save_checkpoint(step, epoch=int(step / n_steps_per_epoch))
                self._callback_handler.on_save(self.engine, self._args, self.client_state)

                metrics.update({"train/checkpoint_runtime": time.time() - ckp_time})

            # Second logging block is used to store main metrics and log everything
            if self._is_log_step(step):
                float_loss = loss.item() / self.engine.world_size
                loss_scale = getattr(self.engine.optimizer, "loss_scale", 1.0)

                gradient_norm = getattr(self.engine.optimizer, "_global_grad_norm", None)
                gradient_norm = gradient_norm.item() if isinstance(gradient_norm, torch.Tensor) else gradient_norm

                metrics.update(
                    {
                        "train/total_runtime": time.time() - train_time,
                        "train/progress": step / self._args.max_steps,
                        "train/epoch": step / n_steps_per_epoch,
                        "train/step": step,
                        "train/loss": float_loss,
                        "train/loss_scale": loss_scale,
                        "train/gradient_norm": gradient_norm,
                        "train/ppl": math.exp(float_loss),
                        "train/learning_rate": self.engine.get_lr()[0],
                        "train/batch_size": self.client_state["train_batch_size"],
                        "train/n_samples": self.engine.global_samples,
                        "train/n_tokens": self.engine.global_samples * self.seq_len,
                        **self._calculate_throughput_metrics(time.time() - step_time),
                    }
                )

                if hasattr(train_iterator, "data_iter") and hasattr(train_iterator.data_iter, "_data_queue"):
                    metrics.update({"train/stream_queue_size": train_iterator.data_iter._data_queue.qsize()})

                self.client_state["log_history"].append(metrics)
                if self.engine.global_rank == 0:
                    logger.info(metrics)

            # Evaluation block is used to perform evaluation and log the evaluation metrics
            if self._is_eval_step(step):
                eval_dataset = (
                    {"0": self._eval_dataset} if not isinstance(self._eval_dataset, dict) else self._eval_dataset
                )

                eval_metrics = {}
                for dn, dataset in eval_dataset.items():
                    eval_loss, _, _, _ = self.evaluate(dataset)
                    eval_metrics.update(
                        {
                            f"eval/{dn}/loss": eval_loss,
                            f"eval/{dn}/ppl": math.exp(eval_loss),
                        }
                    )

                    self._callback_handler.on_evaluate(self.engine, self._args, self.client_state)

                eval_metrics.update({"eval/idx": step // self._args.eval_steps})

                self.client_state["log_history"].append(eval_metrics)
                if self.engine.global_rank == 0:
                    logger.info(eval_metrics)

            # Adjust `train_batch_size` every `rampup_steps` steps until it reaches maximum value
            self._rampup_train_batch_size(step)

        logger.info("Training done.")

        if self.engine.global_rank == 0:
            if self._listener is not None:
                self._listener.stop()

            for handler in logger.handlers:
                if getattr(handler, "end", None) is not None:
                    handler.end()

    def _eval_batch_without_pipe_parallel(self, data_iter: Optional[Iterator] = None) -> torch.Tensor:
        with torch.no_grad():
            gradient_accumulation_steps = self.engine.gradient_accumulation_steps()
            total_loss = 0.0

            for _ in range(gradient_accumulation_steps):
                batch = next(data_iter)
                batch = {k: v.to(self.engine.device) for k, v in batch.items()}

                outputs = self.engine(**batch)
                loss = outputs.loss.mean()

                total_loss += loss

        return total_loss / gradient_accumulation_steps

    def evaluate_step(self, eval_iterator: Iterator) -> torch.Tensor:
        """Perform an evaluation step.

        If ``pipe_parallel_size`` is greater than 1, this method will use
        :meth:`deepspeed.runtime.pipe.engine.PipelineEngine.eval_batch` to perform an evaluation step.
        Otherwise, it will use :meth:`_eval_batch_without_pipe_parallel`.

        Args:
            eval_iterator: Evaluation iterator.

        Returns:
            Evaluation loss.

        """

        if self._args.pipe_parallel_size > 1:
            return self.engine.eval_batch(data_iter=eval_iterator)

        return self._eval_batch_without_pipe_parallel(data_iter=eval_iterator)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[float, float, float, float]:
        """Evaluate a model.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Evaluation loss, time, samples per second and steps per second.

        """

        if eval_dataset is None:
            raise ValueError("`eval_dataset` must be defined, but got None.")

        eval_dataloader = self.get_dataloader(eval_dataset, shuffle=self._args.eval_dataloader_shuffle)
        eval_iterator = iter(RepeatingLoader(eval_dataloader))

        n_eval_steps = self._args.eval_max_steps or len(eval_dataloader)
        eval_loss, eval_time = 0.0, time.time()

        for _ in range(n_eval_steps):
            loss = self.evaluate_step(eval_iterator)
            eval_loss += loss.mean().item()

        eval_loss /= n_eval_steps

        eval_time = time.time() - eval_time
        eval_samples_per_second = (n_eval_steps * self.client_state["train_batch_size"]) / eval_time
        eval_steps_per_second = n_eval_steps / eval_time

        return eval_loss, eval_time, eval_samples_per_second, eval_steps_per_second

    def predict(self) -> None:
        """Predict with a model."""

        raise NotImplementedError
