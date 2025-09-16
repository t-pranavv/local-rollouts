# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import shutil
from argparse import Namespace, _SubParsersAction
from pathlib import Path
from typing import Dict, Optional

import torch

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
)
from phyagi.models.model_convert import (
    convert_llama_to_phi3,
    convert_mixformer_sequential_to_llama,
    convert_mixformer_sequential_to_mistral,
    convert_mixformer_sequential_to_phi,
    convert_mixformer_sequential_to_phi3,
    convert_mixformer_sequential_to_phi4,
    convert_phi3_to_mixformer_sequential,
    convert_qwen2_to_mixformer_sequential,
    convert_qwen3_to_mixformer_sequential,
)
from phyagi.models.registry import get_model, get_tokenizer
from phyagi.utils.checkpoint import (
    convert_ds_zero_checkpoint,
    convert_pl_fsdp_checkpoint,
    convert_ray_actor_checkpoint,
)
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

logger = get_logger(__name__)


CONVERSION_FUNCTIONS = {
    "llama": {
        "phi3": convert_llama_to_phi3,
    },
    MIXFORMER_SEQUENTIAL_MODEL_TYPE: {
        "llama": convert_mixformer_sequential_to_llama,
        "mistral": convert_mixformer_sequential_to_mistral,
        "phi": convert_mixformer_sequential_to_phi,
        "phi3": convert_mixformer_sequential_to_phi3,
        "phi4": convert_mixformer_sequential_to_phi4,
    },
    "phi3": {
        MIXFORMER_SEQUENTIAL_MODEL_TYPE: convert_phi3_to_mixformer_sequential,
    },
    "qwen2": {
        MIXFORMER_SEQUENTIAL_MODEL_TYPE: convert_qwen2_to_mixformer_sequential,
    },
    "qwen3": {
        MIXFORMER_SEQUENTIAL_MODEL_TYPE: convert_qwen3_to_mixformer_sequential,
    },
}


class ConvertCommand:
    """CLI-based command for model conversion."""

    def register(parser: _SubParsersAction) -> None:
        """Register the command with the given parser.

        Args:
            parser: Parser to register the command with.

        """

        command_parser = parser.add_parser("convert", help="Convert a model to a different model type.")

        command_parser.add_argument(
            "pretrained_model_name_or_path", type=str, help="Path/name to the pre-trained model."
        )

        command_parser.add_argument("convert_model_type", type=str, help="Target model type to convert to.")

        command_parser.add_argument(
            "-t",
            "--pretrained_tokenizer_name_or_path",
            type=str,
            default=None,
            help="Path/name to the pre-trained tokenizer.",
        )

        command_parser.add_argument(
            "-re",
            "--resize_embeddings",
            action="store_true",
            help="Whether to resize the embeddings to match the tokenizer's vocabulary size (including added tokens).",
        )

        command_parser.add_argument(
            "-d", "--dtype", type=str, default="float32", help="Data type to convert the model to."
        )

        command_parser.add_argument(
            "-dl", "--debug_logits", action="store_true", help="Whether to debug the converted model logits."
        )

        command_parser.add_argument(
            "-dp", "--debug_params", action="store_true", help="Whether to debug the converted model parameters."
        )

        command_parser.add_argument(
            "-fdz",
            "--from_deepspeed_zero",
            action="store_true",
            help="Whether the model was trained with DeepSpeed ZeRO-{2,3}.",
        )

        command_parser.add_argument(
            "-udu",
            "--use_deepspeed_universal",
            action="store_true",
            help="Whether the model should be converted using DeepSpeed universal checkpoint.",
        )

        command_parser.add_argument(
            "-fpl",
            "--from_pl_fsdp",
            action="store_true",
            help="Whether the model was trained with PyTorch Lightning + FSDP.",
        )

        command_parser.add_argument(
            "-fra",
            "--from_ray_actor",
            action="store_true",
            help="Whether the actor model was trained with Ray.",
        )

        command_parser.add_argument(
            "-sic",
            "--save_intermediate_checkpoints",
            action="store_true",
            help="Whether the store all the intermediate checkpoints while using --from_[deep_speed|raw_actor|pl_fsdp] options.",
        )

        command_parser.set_defaults(func=ConvertCommand)

    def __init__(self, args: Namespace, extra_args: Optional[Dict[str, any]] = None) -> None:
        """Initialize the command with the given arguments.

        Args:
            args: Arguments passed to the command.
            extra_args: Extra arguments passed to the command (not captured by the parser).

        """

        self._pretrained_model_name_or_path = Path(args.pretrained_model_name_or_path)
        self._convert_model_type = args.convert_model_type
        self._pretrained_tokenizer_name_or_path = args.pretrained_tokenizer_name_or_path
        self._resize_embeddings = args.resize_embeddings
        self._dtype = args.dtype
        self._debug_logits = args.debug_logits
        self._debug_params = args.debug_params

        self._from_deepspeed_zero = args.from_deepspeed_zero
        self._from_pl_fsdp = args.from_pl_fsdp
        self._from_ray_actor = args.from_ray_actor
        self._save_intermediate_checkpoints = args.save_intermediate_checkpoints
        if self._from_deepspeed_zero and self._from_pl_fsdp:
            raise ValueError("`from_deepspeed_zero` and `from_pl_fsdp` can not be True at the same time.")
        if self._from_deepspeed_zero and self._from_ray_actor:
            raise ValueError("`from_deepspeed_zero` and `from_ray_actor` can not be True at the same time.")
        if self._from_pl_fsdp and self._from_ray_actor:
            raise ValueError("`from_pl_fsdp` and `from_ray_actor` can not be True at the same time.")

        self._use_deepspeed_universal = args.use_deepspeed_universal
        if not self._from_deepspeed_zero and self._use_deepspeed_universal:
            raise ValueError("`use_deepspeed_universal` can only be True if `from_deepspeed_zero` is True.")

    @property
    def _convert_model_name_or_path(self) -> Path:
        if self._pretrained_model_name_or_path.exists():
            return self._pretrained_model_name_or_path / self._convert_model_type
        return Path(self._pretrained_model_name_or_path.name) / self._convert_model_type

    def _compare_logits(self, model: torch.nn.Module, converted_model: torch.nn.Module) -> Dict[str, float]:
        if not torch.cuda.is_available():
            raise RuntimeError("`--debug_logits` requires CUDA to be available.")

        n_positions = getattr(model.config, "n_positions", None) or getattr(
            model.config, "max_position_embeddings", 2048
        )
        input_ids = torch.randint(1, model.config.vocab_size, (1, n_positions)).cuda()

        def _get_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
            model = model.cuda()
            model.eval()

            with torch.amp.autocast("cuda", dtype=model.dtype):
                with torch.no_grad():
                    logits = model(input_ids).logits.cpu()
            model = model.cpu()

            return logits

        logits = _get_logits(model, input_ids)
        c_logits = _get_logits(converted_model, input_ids)
        diff_logits = (logits - c_logits).abs()

        return {
            "max": diff_logits.max().item(),
            "mean": diff_logits.mean().item(),
            "median": diff_logits.median().item(),
            "std": diff_logits.std().item(),
        }

    def _compare_params(self, model: torch.nn.Module, converted_model: torch.nn.Module) -> Dict[str, int]:
        return {
            "n_model_params": sum(p.numel() for p in model.parameters()),
            "n_converted_model_params": sum(p.numel() for p in converted_model.parameters()),
        }

    def run(self) -> None:
        """Run the command."""

        temporary_model_name_or_path = self._pretrained_model_name_or_path
        if self._from_deepspeed_zero or self._from_pl_fsdp or self._from_ray_actor:
            temporary_model_name_or_path = self._pretrained_model_name_or_path / "tmp"

        if self._from_deepspeed_zero:
            logger.info("Converting DeepSpeed ZeRO checkpoint...")
            convert_ds_zero_checkpoint(
                self._pretrained_model_name_or_path,
                temporary_model_name_or_path,
                use_universal_checkpoint=self._use_deepspeed_universal,
            )

        if self._from_pl_fsdp:
            logger.info("Converting PyTorch Lightning FSDP checkpoint...")
            convert_pl_fsdp_checkpoint(self._pretrained_model_name_or_path, temporary_model_name_or_path)

        if self._from_ray_actor:
            logger.info("Converting Ray actor FSDP checkpoint...")
            convert_ray_actor_checkpoint(self._pretrained_model_name_or_path, temporary_model_name_or_path)

        model = get_model(temporary_model_name_or_path)
        tokenizer = (
            get_tokenizer(self._pretrained_tokenizer_name_or_path) if self._pretrained_tokenizer_name_or_path else None
        )

        if model.config.model_type not in CONVERSION_FUNCTIONS:
            raise ValueError(
                f"`model_type` must be one of {list(CONVERSION_FUNCTIONS.keys())}, but got '{model.config.model_type}'."
            )
        if self._convert_model_type not in CONVERSION_FUNCTIONS[model.config.model_type]:
            raise ValueError(
                f"`convert_model_type` must be one of {list(CONVERSION_FUNCTIONS[model.config.model_type].keys())}, but got '{self._convert_model_type}'."
            )

        if model.config.model_type != self._convert_model_type:
            logger.info(f"Converting from {model.config.model_type} to {self._convert_model_type}...")
            convert_fn = CONVERSION_FUNCTIONS[model.config.model_type][self._convert_model_type]
            converted_model = convert_fn(model)
            logger.info("Model converted.")

        model = model.to(to_torch_dtype(self._dtype))
        converted_model = converted_model.to(to_torch_dtype(self._dtype))

        if self._debug_logits:
            logger.info("Debugging logits...")
            metrics = self._compare_logits(model, converted_model)
            logger.info(metrics)

        if self._debug_params:
            logger.info("Debugging parameters...")
            metrics = self._compare_params(model, converted_model)
            logger.info(metrics)

        if tokenizer is not None:
            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(self._convert_model_name_or_path)
            logger.info("Tokenizer saved.")

            if self._resize_embeddings:
                logger.info("Resizing embeddings...")
                converted_model.resize_token_embeddings(len(tokenizer))
                logger.info("Embeddings resized.")

        logger.info(f"Saving converted model: {self._convert_model_name_or_path}")

        if model.config.model_type == self._convert_model_type:
            # If the model type is the same, we just rename the temporary directory to the final one to avoid wasting time
            shutil.move(temporary_model_name_or_path, self._convert_model_name_or_path)
        else:
            converted_model.save_pretrained(self._convert_model_name_or_path)

            shutil.copy(inspect.getfile(converted_model.__class__), self._convert_model_name_or_path)
            shutil.copy(inspect.getfile(converted_model.config.__class__), self._convert_model_name_or_path)

            if not(self._save_intermediate_checkpoints) and (self._from_deepspeed_zero or self._from_pl_fsdp or self._from_ray_actor):
                shutil.rmtree(temporary_model_name_or_path, ignore_errors=True)

        logger.info("Model saved.")
