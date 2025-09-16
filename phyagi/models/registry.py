# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import (
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

logger = get_logger(__name__)

MODELS = {
    "causal_lm": AutoModelForCausalLM,
    "seq_cls": AutoModelForSequenceClassification,
}


def _add_special_tokens(
    tokenizer: PreTrainedTokenizerBase, special_tokens: Optional[Union[List[str], str]] = None
) -> None:
    special_tokens = [special_tokens] if not isinstance(special_tokens, list) else special_tokens
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(sp, normalized=False, rstrip=True, lstrip=False) for sp in special_tokens
            ]
        }
    )


def _apply_hf_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = copy.deepcopy(kwargs)

    # Required for some HF models, such as Replit
    if kwargs.get("trust_remote_code") is None:
        kwargs["trust_remote_code"] = True

    # `torch_dtype` might be supplied as a string, but HF expects a `torch.dtype` object unless it is `auto`
    if isinstance(kwargs.get("torch_dtype"), str):
        if kwargs["torch_dtype"] != "auto":
            kwargs["torch_dtype"] = to_torch_dtype(kwargs["torch_dtype"])

    return kwargs


def _find_latest_checkpoint(checkpoint_dir: Path) -> str:
    latest_checkpoint_path = checkpoint_dir / "latest"
    if latest_checkpoint_path.is_file():
        with open(latest_checkpoint_path, "r") as f:
            latest_checkpoint = f.read().strip()
        return checkpoint_dir / latest_checkpoint

    return str(checkpoint_dir)


def get_model(
    pretrained_model_name_or_path: Optional[Union[str, Path]] = None,
    model_task: str = "causal_lm",
    use_torch_compile: bool = False,
    **kwargs,
) -> Union[AutoModelForCausalLM, AutoModelForSequenceClassification]:
    """Get a model from a local path, Hugging Face hub, or configuration.

    If ``pretrained_model_name_or_path`` is ``None``, a randomly initialized model
    will be created from the configuration in ``kwargs``.

    Args:
        pretrained_model_name_or_path: Pre-trained model name or path.
        model_task: Name of the model task.
        use_torch_compile: Whether to use ``torch.compile()`` for model.

    Returns:
        Model.

    """

    # Apply Hugging Face additional keyword arguments
    kwargs = _apply_hf_kwargs(kwargs)

    if model_task not in MODELS:
        raise ValueError(f"`model_task` must be one of {list(MODELS.keys())}, but got '{model_task}'.")
    model_cls = MODELS[model_task]

    if pretrained_model_name_or_path is None:
        logger.info(f"Creating model with configuration: {kwargs}")

        # If path is not specified then we create the model from scratch
        model_config = AutoConfig.for_model(**kwargs)
        model = model_cls.from_config(model_config)
    else:
        # If checkpoint has been saved with DeepSpeed, there might be a chance that
        # there is a `latest` file with the name of the checkpoint
        pretrained_model_name_or_path = _find_latest_checkpoint(Path(pretrained_model_name_or_path))

        logger.info(f"Loading pre-trained model: {pretrained_model_name_or_path}")
        logger.info(f"Model configuration: {kwargs}")

        model = model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if use_torch_compile:
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            model = torch.compile(model)
        else:
            logger.warning("`use_torch_compile` is not supported with PyTorch < 2.0.0.")

    return model


def get_tokenizer(
    pretrained_tokenizer_name_or_path: Union[str, Path],
    additional_special_tokens: Optional[Union[List[str], str]] = None,
    **kwargs,
) -> PreTrainedTokenizerBase:
    """Get a pre-trained tokenizer from a local path or Hugging Face hub.

    If ``pretrained_tokenizer_name_or_path`` is a local path, the tokenizer will be loaded from
    that path, assuming it is a valid tokenizer. If ``pretrained_tokenizer_name_or_path`` is
    a Hugging Face model name, the tokenizer will be loaded from the Hugging Face hub.

    Args:
        pretrained_tokenizer_name_or_path: Pre-trained tokenizer name or path.
        additional_special_tokens: Special tokens to add to the tokenizer.
        kwargs: Additional keyword arguments to pass to ``AutoTokenizer.from_pretrained()``,
            such as `chat_template` and `trust_remote_code`.

    Returns:
        Pre-trained tokenizer.

    """

    logger.info(f"Loading pre-trained tokenizer: {pretrained_tokenizer_name_or_path}")
    logger.info(f"Tokenizer configuration: {kwargs}")

    # Apply Hugging Face additional keyword arguments
    kwargs = _apply_hf_kwargs(kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path, **kwargs)

    if additional_special_tokens:
        logger.info(f"Using additional special tokens: {additional_special_tokens}")
        _add_special_tokens(tokenizer, additional_special_tokens)

    return tokenizer
