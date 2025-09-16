# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from argparse import Namespace, _SubParsersAction
from typing import Dict, Optional

import torch
from torch.utils.benchmark import Timer

from phyagi.models.registry import get_model
from phyagi.utils.config import load_config
from phyagi.utils.hf_utils import to_device_map
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

logger = get_logger(__name__)


class SpeedBenchmarkCommand:
    """CLI-based command for speed-benchmarking architectures (timing of forward and backward passes)."""

    def register(parser: _SubParsersAction) -> None:
        """Register the command with the given parser.

        Args:
            parser: Parser to register the command with.

        """

        command_parser = parser.add_parser(
            "speed-benchmark", help="Speed-benchmark forward/backward passes of an architecture."
        )

        command_parser.add_argument("model_config_file_path", type=str, help="Path to the model configuration file.")

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

        command_parser.add_argument("-nt", "--n_trials", type=int, default=10, help="Number of trials to run.")

        command_parser.set_defaults(func=SpeedBenchmarkCommand)

    def __init__(self, args: Namespace, extra_args: Optional[Dict[str, any]] = None) -> None:
        """Initialize the command with the given arguments.

        Args:
            args: Arguments passed to the command.
            extra_args: Extra arguments passed to the command (not captured by the parser).

        """

        self._model_config_file_path = args.model_config_file_path
        self._device_map = args.device_map
        self._dtype = to_torch_dtype(args.dtype)
        self._n_trials = args.n_trials

    def _fwd_pass(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> None:
        with torch.amp.autocast(device_type=model.device.type, dtype=model.dtype):
            model(**inputs)

    def _fwd_and_bwd_pass(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> None:
        with torch.amp.autocast(device_type=model.device.type, dtype=model.dtype):
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

    def run(self) -> None:
        """Run the command."""

        config = load_config(self._model_config_file_path)
        if config.get("model", None) is None:
            raise ValueError(f"`model` must be a key defined in '{self._model_config_file_path}', but got None.")

        model = get_model(**config["model"]).eval()
        model = to_device_map(model, device_map=self._device_map)
        model = model.to(dtype=self._dtype)

        input_ids = torch.randint(1, model.config.vocab_size, (1, model.config.n_positions)).to(model.device)
        inputs = {"input_ids": input_ids, "labels": input_ids}

        for fn in ["fwd_pass", "fwd_and_bwd_pass"]:
            logger.info(f"Benchmarking function: {fn}")

            latency = Timer(
                stmt=f"{fn}(model, inputs)",
                globals={
                    "model": model,
                    "inputs": inputs,
                    "fwd_pass": self._fwd_pass,
                    "fwd_and_bwd_pass": self._fwd_and_bwd_pass,
                },
            ).timeit(self._n_trials)

            logger.info(
                {
                    "fn": fn,
                    "device": model.device.type,
                    "dtype": model.dtype,
                    "seq_len": model.config.n_positions,
                    "median_latency_ms": latency.mean * 1e3,
                }
            )
