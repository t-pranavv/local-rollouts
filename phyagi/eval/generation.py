# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import get_world_size, is_main_process
from phyagi.eval.log_likelihood_pipeline import LogLikelihoodPipeline
from phyagi.eval.loss_pipeline import LossPipeline
from phyagi.eval.text_generation_pipeline import TextGenerationPipeline


def example_generator(
    dataset: Dataset,
    mapping_fn: Optional[Callable[[Any], Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    shuffle: bool = False,
    seed: int = 42,
    drop_last: bool = False,
    **mapping_fn_kwargs,
) -> Iterator[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Yield and optionally map examples over a dataset.

    Args:
        dataset: Dataset to generate examples from.
        mapping_fn: Mapping function to apply to each example.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed (applied to sampler and data loader).
        drop_last: Whether to drop the last incomplete sample.

    Yields:
        Iterator over examples.

    """

    sampler = None
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    collate_fn = (lambda x: mapping_fn(x, **mapping_fn_kwargs)) if mapping_fn is not None else None
    dataloader = DataLoader(dataset, batch_size=None, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn)
    for example in dataloader:
        yield example


class GenerationEngine:
    """Abstract class for generation engines.

    This class serves as a base for implementing generation engines that can generate information
    from prompts. It enforces implementation of the :meth:`generate` method.

    Examples:
        >>> class MyGenerationEngine(GenerationEngine):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>
        >>>     def generate(self, prompts: Union[List[Union[str, Dict[str, Any]]], Dataset], **kwargs) -> List[Dict[str, Any]]:
        >>>         return [{"responses": ["Anything"], "metadata": "Metadata"} for _ in prompts]

    """

    def generate(self, prompts: Union[List[Union[str, Dict[str, Any]]], Dataset], **kwargs) -> List[Dict[str, Any]]:
        """Generate information.

        Args:
            prompts: Prompts to generate information from.

        Returns:
            List of dictionaries containing generated information (and metadata).

        """

        raise NotImplementedError("`GenerationEngine` must implement `generate()`.")


class HfGenerationEngine(GenerationEngine):
    """Generation engine using Hugging Face."""

    def generate(
        self,
        prompts: Union[List[Union[str, Dict[str, Any]]], Dataset],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        example_generator_kwargs: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **generate_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate information using Hugging Face ``generate`` method.

        Args:
            prompts: Prompts to generate information from.
            model: Instance of a model for generation.
            tokenizer: Instance of a tokenizer for generation.
            device: Device to use for generation.
            example_generator_kwargs: Keyword arguments for the ``example_generator`` function.
            generation_config: Generation configuration.

        Returns:
            List of dictionaries containing generated information.

        """

        def _default_mapping_fn(prompt: Union[Dict[str, Any], str]) -> Dict[str, torch.Tensor]:
            if isinstance(prompt, str):
                return {"input_ids": tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)}
            return {"input_ids": tokenizer(prompt["text"], return_tensors="pt").input_ids.to(model.device)}

        model.to(device or "cpu")

        example_generator_kwargs = example_generator_kwargs or {"mapping_fn": _default_mapping_fn}
        generation_config = generation_config or {}

        responses = []
        for prompt in tqdm(
            example_generator(prompts, **example_generator_kwargs),
            total=math.ceil(len(prompts) / get_world_size()),
            disable=not is_main_process(),
        ):
            output_ids = model.generate(**prompt, **generation_config, **generate_kwargs)
            responses.append({"responses": [tokenizer.decode(output_ids[0], skip_special_tokens=True)]})

        return responses


class LogLikelihoodPipelineEngine(GenerationEngine):
    """Generation engine using the log likelihood pipeline."""

    def generate(
        self,
        prompts: Union[List[Dict[str, Any]], Dataset],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        example_generator_kwargs: Optional[Dict[str, Any]] = None,
        **pipeline_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate information using the log-likelihood pipeline.

        Args:
            prompts: Prompts to generate information from.
            model: Instance of a model for generation.
            tokenizer: Instance of a tokenizer for generation.
            device: Device to use for generation.
            example_generator_kwargs: Keyword arguments for the ``example_generator`` function.

        """

        pipeline = LogLikelihoodPipeline(model, tokenizer, device=device)
        example_generator_kwargs = example_generator_kwargs or {}

        responses = []
        for response in tqdm(
            pipeline(
                example_generator(prompts, **example_generator_kwargs),
                **pipeline_kwargs,
            ),
            total=math.ceil(len(prompts) / get_world_size()),
            disable=not is_main_process(),
        ):
            responses.append(response)

        return responses


class LossPipelineEngine(GenerationEngine):
    """Generation engine using the loss pipeline."""

    def generate(
        self,
        prompts: Union[List[Dict[str, Any]], Dataset],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        example_generator_kwargs: Optional[Dict[str, Any]] = None,
        shift_labels: bool = True,
        **pipeline_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate information using the loss pipeline.

        Args:
            prompts: Prompts to generate information from.
            model: Instance of a model for generation.
            tokenizer: Instance of a tokenizer for generation.
            device: Device to use for generation.
            example_generator_kwargs: Keyword arguments for the ``example_generator`` function.
            shift_labels: Whether to shift labels for loss calculation.

        """

        pipeline = LossPipeline(model, tokenizer, device=device)
        example_generator_kwargs = example_generator_kwargs or {}

        responses = []
        for response in tqdm(
            pipeline(
                example_generator(prompts, **example_generator_kwargs),
                shift_labels=shift_labels,
                **pipeline_kwargs,
            ),
            total=math.ceil(len(prompts) / get_world_size()),
            disable=not is_main_process(),
        ):
            responses.append(response)

        return responses


class TextGenerationPipelineEngine(GenerationEngine):
    """Generation engine using the text generation pipeline."""

    def generate(
        self,
        prompts: Union[List[Dict[str, Any]], Dataset],
        model: AutoModelForCausalLM = None,
        tokenizer: PreTrainedTokenizerBase = None,
        device: Optional[Union[int, torch.device]] = None,
        example_generator_kwargs: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **pipeline_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate information using the text generation pipeline.

        Args:
            prompts: Prompts to generate information from.
            model: Instance of a model for generation.
            tokenizer: Instance of a tokenizer for generation.
            device: Device to use for generation.
            example_generator_kwargs: Keyword arguments for the ``example_generator`` function.
            generation_config: Generation configuration.

        """

        pipeline = TextGenerationPipeline(model, tokenizer, device=device)
        example_generator_kwargs = example_generator_kwargs or {}
        generation_config = generation_config or {}

        responses = []
        for response in tqdm(
            pipeline(
                example_generator(prompts, **example_generator_kwargs),
                **generation_config,
                **pipeline_kwargs,
            ),
            total=math.ceil(len(prompts) / get_world_size()),
            disable=not is_main_process(),
        ):
            responses.append(response)

        return responses


GENERATION_ENGINES = {
    "hf": HfGenerationEngine,
    "log_likelihood_pipeline": LogLikelihoodPipelineEngine,
    "loss_pipeline": LossPipelineEngine,
    "text_generation_pipeline": TextGenerationPipelineEngine,
}


def generate(
    prompts: Union[List[Union[str, Dict[str, Any]]], Dataset], generation_engine: str = "hf", **kwargs
) -> List[Dict[str, Any]]:
    """Generate information using the specified generation engine.

    Args:
        prompts: Prompts to generate information from.
        generation_engine: Type of generation engine.

    Returns:
        List of dictionaries containing generated information (and metadata).

    """

    if generation_engine not in GENERATION_ENGINES:
        raise ValueError(
            f"`generation_engine` must be one of {list(GENERATION_ENGINES.keys())}, but got '{generation_engine}'."
        )

    engine = GENERATION_ENGINES[generation_engine]()
    return engine.generate(prompts, **kwargs)
