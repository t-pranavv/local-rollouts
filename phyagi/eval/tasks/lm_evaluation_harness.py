# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import get_rank, get_world_size, is_main_process


def _clean_results(results: Dict[str, Any]) -> Dict[str, Any]:
    remove_metrics_keys = ["alias"]
    clean_metrics_map = {
        "acc": "accuracy",
        "acc_stderr": "accuracy_stderr",
        "acc_norm": "accuracy_norm",
        "acc_norm_stderr": "accuracy_norm_stderr",
    }
    regex_map = {re.compile(rf"^{k},.*$"): v for k, v in clean_metrics_map.items()}

    clean_results = {}
    for task, task_results in results.items():
        clean_metrics = {}
        for metric_key, metric_value in task_results.items():
            if metric_key in remove_metrics_keys:
                continue
            clean_metric_key = next((v for pattern, v in regex_map.items() if pattern.match(metric_key)), metric_key)
            clean_metrics[clean_metric_key] = metric_value
        clean_results[task] = clean_metrics

    return clean_results


class LMEvaluationHarness:
    """Language model evaluation harness task.

    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness

    """

    @staticmethod
    def run(
        model: Union[str, torch.nn.Module],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_args: Optional[Union[str, Dict[str, Any]]] = "",
        tasks: Optional[Union[str, List[str]]] = None,
        num_fewshot: int = 0,
        batch_size: Union[int, str] = 1,
        max_batch_size: Optional[int] = None,
        device: Optional[Union[int, torch.device]] = None,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = False,
        gen_kwargs: Optional[str] = None,
        task_manager: Optional[TaskManager] = None,
        verbosity: str = "INFO",
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        use_amp: bool = True,
    ) -> Dict[str, Any]:
        tasks = tasks or []
        tasks = [tasks] if isinstance(tasks, str) else tasks

        lm_eval_model = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_batch_size=max_batch_size
        )

        # Since `lm-eval` does not initialize anything if we pass a `torch.nn.Module`,
        # we manually set the required arguments
        lm_eval_model._rank = get_rank()
        lm_eval_model._world_size = get_world_size()
        if lm_eval_model._world_size > 1:
            lm_eval_model.accelerator = Accelerator()

        # `use_amp` ensures that the model is run in mixed precision mode, e.g., flash-attn only supports fp16/bf16
        amp_dtype = torch.bfloat16 if lm_eval_model.model.dtype == torch.bfloat16 else None
        with torch.autocast(lm_eval_model.model.device.type, dtype=amp_dtype, enabled=use_amp):
            results = simple_evaluate(
                lm_eval_model,
                model_args=model_args,
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device,
                use_cache=use_cache,
                cache_requests=cache_requests,
                rewrite_requests_cache=rewrite_requests_cache,
                delete_requests_cache=delete_requests_cache,
                limit=limit,
                bootstrap_iters=bootstrap_iters,
                check_integrity=check_integrity,
                write_out=write_out,
                log_samples=log_samples,
                gen_kwargs=gen_kwargs,
                task_manager=task_manager,
                predict_only=predict_only,
                random_seed=random_seed,
                numpy_random_seed=numpy_random_seed,
                torch_random_seed=torch_random_seed,
            )

            # `lm-eval` returns None for non-main processes, so we ensure that
            # we always have a dictionary
            results = results or {}

        # Since non-main processes will not have a complete dictionary, we only process
        # the results in the main process
        if is_main_process():
            results = _clean_results(results["results"])

        return results
