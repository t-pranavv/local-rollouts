# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.pipelines.base import ChunkPipeline


def _create_target_mask(input_ids: torch.LongTensor, target_ids: torch.LongTensor) -> torch.BoolTensor:
    input_len = len(input_ids)
    target_len = len(target_ids)
    target_mask = torch.zeros(input_len, dtype=torch.bool)

    if target_len == 0:
        return torch.roll(target_mask, -1)

    # Search from right to left since we want the last occurrence of the target
    for i in range(input_len - target_len, -1, -1):
        if torch.equal(input_ids[i : i + target_len], target_ids):
            target_mask[i : i + target_len] = True
            break
    else:
        # Shift and search again if target not found
        return _create_target_mask(input_ids, target_ids[1:])

    return target_mask


class LogLikelihoodPipeline(ChunkPipeline):
    """Log-likelihood pipeline.

    This pipeline is used to compute the log-likelihood of a target given an input sequence. For example,
    given an input sequence 'I love Paris', and a target sequence 'Paris', the pipeline will compute the
    log-likelihood of 'Paris' given 'I love'.

    """

    _EXPECTED_INPUTS_KEYS = {"text", "target", "label"}

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        **kwargs,
    ) -> None:
        # Make a deep copy of the tokenizer to prevent side-effects of changing its attributes
        tokenizer = copy.deepcopy(tokenizer)

        # Ensure that tokenizer has the correct maximum length, truncation and padding settings
        tokenizer.model_max_length = getattr(model.config, "n_positions", tokenizer.model_max_length)
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(model, tokenizer, device=device, **kwargs)

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        preprocess_kwargs = {}
        if "return_inputs" in kwargs:
            preprocess_kwargs["return_inputs"] = kwargs["return_inputs"]

        forward_kwargs = {}
        if "pad_token_id" in kwargs:
            forward_kwargs["pad_token_id"] = kwargs["pad_token_id"]
        if "use_amp" in kwargs:
            forward_kwargs["use_amp"] = kwargs["use_amp"]

        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, inputs: List[Dict[str, Any]], return_inputs: bool = True) -> Iterator[Dict[str, Any]]:
        if not isinstance(inputs, list):
            raise TypeError(f"`inputs` must be a list, but got '{type(inputs)}'.")

        for i, example in enumerate(inputs):
            if not self._EXPECTED_INPUTS_KEYS.issubset(example.keys()):
                raise ValueError(
                    f"`inputs` item should have at least {self._EXPECTED_INPUTS_KEYS} keys, but got {example.keys()}."
                )

            # Tokenize source and target
            tokenized_example = self.tokenizer(
                example["text"], text_target=example["target"], truncation=True, return_attention_mask=False
            )

            input_ids = tokenized_example["input_ids"]
            target_ids = tokenized_example["labels"]
            label = example.pop("label", None)

            input_ids = torch.tensor(input_ids)
            target_ids = torch.tensor(target_ids)

            target_mask = _create_target_mask(input_ids, target_ids)

            yield {
                "inputs": example if return_inputs else None,
                "input_ids": input_ids.unsqueeze(0),
                "target_mask": target_mask.unsqueeze(0),
                "target_length": len(example["target"]),
                "label": label,
                "is_last": i == len(inputs) - 1,
            }

    def _forward(
        self, model_inputs: Dict[str, Any], pad_token_id: Optional[int] = None, use_amp: bool = True
    ) -> Dict[str, Any]:
        amp_dtype = torch.bfloat16 if self.model.dtype == torch.bfloat16 else None
        with torch.autocast(self.model.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = self.model(input_ids=model_inputs["input_ids"]).logits

        return {
            "logits": logits,
            **model_inputs,
        }

    def forward(self, model_inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        with self.device_placement():
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs = self._forward(model_inputs, **forward_params)

        return model_outputs

    def postprocess(self, model_outputs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        inputs = [model_output["inputs"] for model_output in model_outputs]
        labels = [model_output["label"] for model_output in model_outputs]
        target_lengths = [model_output["target_length"] for model_output in model_outputs]

        exact_matches, log_likelihoods = [], []
        for model_output in model_outputs:
            # Since things can be batched during `_forward`, we need to convert `target_mask` to a boolean tensor
            # and retrieve the original `target_ids` from `input_ids`
            target_mask = model_output["target_mask"].to(torch.bool)
            target_ids = torch.masked_select(model_output["input_ids"], target_mask).unsqueeze(0)

            # Since we are predicting the next token, we need to left-shift the `target_mask` by 1
            # as the log-probability of ith token is predicted using the (i-1)th token
            probs = F.log_softmax(model_output["logits"], dim=-1)
            probs = probs[:, target_mask.roll(-1).squeeze(), :]

            # If there is an available target, check if predictions have been an exact match
            if target_ids.numel() == 0:
                exact_match = False
            else:
                exact_match = torch.equal(probs.argmax(dim=-1), target_ids)

            # Gather log-probabilities of the predicted tokens and calculate the log-likelihood
            log_likelihood = torch.gather(probs, 2, target_ids.unsqueeze(-1))
            log_likelihood = log_likelihood.sum().cpu().item()

            exact_matches.append(exact_match)
            log_likelihoods.append(log_likelihood)

        return {
            "inputs": inputs,
            "labels": labels,
            "target_lengths": target_lengths,
            "exact_matches": exact_matches,
            "log_likelihoods": log_likelihoods,
        }
