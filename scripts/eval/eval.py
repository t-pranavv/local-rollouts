# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import copy
import json
import os
import re
from typing import Any, Dict

import pandas as pd

from phyagi.eval.registry import run_task
from phyagi.models.registry import get_model, get_tokenizer
from phyagi.utils.config import load_config
from phyagi.utils.file_utils import get_checkpoints_info, get_full_path
from phyagi.utils.hf_utils import to_device_map
from phyagi.utils.logging_handlers import MlflowHandler, WandbHandler
from phyagi.utils.logging_utils import get_logger
from phyagi.version import get_package_information

STEP_KEY = "checkpoint/idx"

logger = get_logger(__name__)


def _log_task_results(task_results: Dict[str, Any], step: int) -> None:
    for name, results in task_results.items():
        metrics = {f"{name}/{metric.replace('@', '_')}": value for metric, value in results.items()}
        metrics.update({STEP_KEY: step})

        logger.info(metrics)


def _save_results(results: Dict[str, Any], output_dir: str) -> None:
    current_results = copy.deepcopy(results)

    # Save results to a JSON file
    output_json_path = os.path.join(output_dir, "eval_results.json")
    with open(output_json_path, "w") as f:
        json.dump(current_results, f, indent=4)

    # Save results to a markdown file
    # Also remove unnecessary keys to avoid cluttering the markdown file
    current_results.pop("checkpoint/path")
    current_results.pop("checkpoint/idx")

    output_md_path = os.path.join(output_dir, "eval_results.md")
    with open(output_md_path, "w") as f:
        f.write(pd.DataFrame.from_dict(current_results, orient="index").to_markdown())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a model.")

    parser.add_argument(
        "config_file_path",
        type=str,
        nargs="*",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "pretrained_model_name_or_path",
        type=str,
        help="Path to the pre-trained model folder (not the actual checkpoint folder).",
    )

    parser.add_argument(
        "-n",
        "--n_checkpoints",
        type=int,
        default=None,
        help="Number of checkpoints to evaluate (default evaluates all checkpoints).",
    )

    parser.add_argument(
        "-so",
        "--step_offset",
        type=int,
        default=0,
        help="Add an optional step offset for WandB logging.",
    )

    parser.add_argument(
        "-dm",
        "--device_map",
        type=str,
        default="cuda",
        choices=["auto", "balanced", "balanced_low_0", "cpu", "cuda", "sequential"],
        help="Device map to use for the model.",
    )

    parser.add_argument(
        "-wp",
        "--wandb_project",
        type=str,
        default=None,
        help="Specify a WandB project.",
    )

    parser.add_argument(
        "-wg",
        "--wandb_group",
        type=str,
        default=None,
        help="Specify a WandB group.",
    )

    args, extra_args = parser.parse_known_args()

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = _parse_args()
    config = load_config(*args.config_file_path, extra_args)

    # Print information about the package (including branch and commit hash if available)
    print(get_package_information())

    # Attach the MLflow and WandB handlers
    logger.addHandler(MlflowHandler(tags=config, step_key=STEP_KEY))
    logger.addHandler(
        WandbHandler(
            project=args.wandb_project,
            group=args.wandb_group,
            entity="phyagi",
            config=config,
            step_key=STEP_KEY,
        )
    )

    # Load the tokenizer (optional since some tasks do not require a tokenizer)
    tokenizer_config = config.get("tokenizer", {})
    if tokenizer_config.get("pretrained_tokenizer_name_or_path", None) is None:
        tokenizer_config["pretrained_tokenizer_name_or_path"] = args.pretrained_model_name_or_path
    try:
        tokenizer = get_tokenizer(**tokenizer_config)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        tokenizer = None

    # Find `n_checkpoints` checkpoints in the `pretrained_model_name_or_path` folder
    # according to a regex pattern (in this case, the pattern is last, number or checkpoint-number)
    checkpoints = get_checkpoints_info(
        args.pretrained_model_name_or_path,
        checkpoint_regex=re.compile(r"^((\d+)|last)$|checkpoint-\d+"),
        n_checkpoints=args.n_checkpoints,
        reverse=False,
    )
    for checkpoint in checkpoints:
        assert "output_dir" in config, "`output_dir` must be available in configuration."
        output_dir = str(get_full_path(os.path.join(config["output_dir"], checkpoint["basename"]), create_folder=True))

        # Load model from checkpoint and optionally override the model configuration
        model_config = config.get("model", {})
        model = get_model(checkpoint["path"], **model_config)
        model = to_device_map(model, device_map=args.device_map)

        # Filters tasks according to the `task_names` key in the configuration file
        # If `task_names` is not available, all tasks are evaluated
        eval_tasks = config.get("eval_tasks", {})
        eval_tasks_filter = config.get("task_names", list(eval_tasks.keys()))
        eval_tasks = {k: v for k, v in eval_tasks.items() if k in eval_tasks_filter}

        results = {"checkpoint/path": checkpoint["path"], STEP_KEY: checkpoint["step"]}

        for task in eval_tasks.values():
            logger.info(f"Loading model from checkpoint: {checkpoint['path']}")

            task_name = task.get("task_name", "")
            task_kwargs = task.get("task_kwargs", {})
            if "output_file_path" in task_kwargs.keys():
                task_kwargs["output_file_path"] = os.path.join(output_dir, task_kwargs["output_file_path"])

            # Run the evaluation task
            try:
                task_results = run_task(task_name, model, tokenizer, **task_kwargs)
                if task_name != "lm_eval":
                    task_results = {task_name: task_results}
            except Exception as e:
                logger.warning(f"Failed to run {task_name} task: {e}")
                task_results = {task_name: {}}

            # Log current task results to the console, MLflow and WandB
            step = checkpoint["step"] + args.step_offset if args.step_offset else checkpoint["step"]
            _log_task_results(task_results, step)

            # Save results to output files
            results.update(task_results)
            _save_results(results, output_dir)

    # Finalize the handlers
    for handler in logger.handlers:
        if isinstance(handler, (MlflowHandler, WandbHandler)):
            handler.end()
