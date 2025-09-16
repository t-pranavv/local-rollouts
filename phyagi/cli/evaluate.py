# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from argparse import Namespace, _SubParsersAction
from typing import Dict, Optional

from phyagi.eval.registry import run_task
from phyagi.models.registry import get_model, get_tokenizer
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

logger = get_logger(__name__)


class EvaluateCommand:
    """CLI-based command for model evaluation."""

    def register(parser: _SubParsersAction) -> None:
        """Register the command with the given parser.

        Args:
            parser: Parser to register the command with.

        """

        command_parser = parser.add_parser("evaluate", help="Evaluate a pre-trained model.")

        command_parser.add_argument(
            "pretrained_model_name_or_path", type=str, help="Path/name to the pre-trained model."
        )

        command_parser.add_argument("task_name", type=str, help="Name to the evaluation task.")

        command_parser.add_argument(
            "-t",
            "--pretrained_tokenizer_name_or_path",
            type=str,
            default=None,
            help="Path/name to the pre-trained tokenizer.",
        )

        command_parser.add_argument(
            "-dm",
            "--device_map",
            type=str,
            default="cuda",
            choices=["auto", "balanced", "balanced_low_0", "cpu", "cuda", "sequential"],
            help="Device map to use for the model.",
        )

        command_parser.add_argument(
            "-d", "--dtype", type=str, default="float32", help="Data type to use for the model."
        )

        command_parser.add_argument(
            "-bs", "--batch_size", type=int, default=1, help="Batch size to use for evaluation."
        )

        command_parser.add_argument(
            "-amp", "--use_amp", action="store_true", help="Whether to use automatic mixed precision."
        )

        command_parser.set_defaults(func=EvaluateCommand)

    def __init__(self, args: Namespace, extra_args: Optional[Dict[str, any]] = None) -> None:
        """Initialize the command with the given arguments.

        Args:
            args: Arguments passed to the command.
            extra_args: Extra arguments passed to the command (not captured by the parser).

        """

        self._pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self._task_name = args.task_name
        self._pretrained_tokenizer_name_or_path = (
            args.pretrained_tokenizer_name_or_path or args.pretrained_model_name_or_path
        )
        self._device_map = args.device_map
        self._dtype = to_torch_dtype(args.dtype)
        self._batch_size = args.batch_size
        self._use_amp = args.use_amp

        # For the `evaluate` command, `extra_args` are used as `run_task()` keyword arguments
        self._task_kwargs = extra_args

    def run(self) -> None:
        """Run the command."""

        model = get_model(self._pretrained_model_name_or_path, device_map=self._device_map, torch_dtype=self._dtype)
        tokenizer = get_tokenizer(self._pretrained_tokenizer_name_or_path)

        metrics = run_task(
            self._task_name, model, tokenizer, batch_size=self._batch_size, use_amp=self._use_amp, **self._task_kwargs
        )
        logger.info(metrics)
