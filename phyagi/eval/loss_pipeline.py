# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.pipelines import Pipeline


class LossPipeline(Pipeline):
    """Loss-based pipeline.

    This pipeline is used to compute the the loss and perplexity of predicting
    ``t + 1`` labels from ``t`` inputs.

    """

    _EXPECTED_INPUTS_KEYS = {"text"}

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: Optional[Union[int, torch.device]] = None,
        **kwargs,
    ) -> None:
        # If `tokenizer` is not available, we expect that inputs are already encoded
        # and thus we mock a tokenizer to allow batched inference
        tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        tokenizer = copy.deepcopy(tokenizer)

        # Ensure that tokenizer has the correct maximum length, padding and truncation settings
        tokenizer.model_max_length = getattr(model.config, "n_positions", tokenizer.model_max_length)
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(model, tokenizer, device=device, **kwargs)

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        forward_kwargs = {}
        if "pad_token_id" in kwargs:
            forward_kwargs["pad_token_id"] = kwargs["pad_token_id"]
        if "use_amp" in kwargs:
            forward_kwargs["use_amp"] = kwargs["use_amp"]
        if "shift_labels" in kwargs:
            forward_kwargs["shift_labels"] = kwargs["shift_labels"]

        return {}, forward_kwargs, {}

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise TypeError(f"`inputs` must be a dict, but got '{type(inputs)}'.")

        if "input_ids" in inputs.keys():
            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        else:
            if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"`tokenizer` must be an instance of PreTrainedTokenizerBase when `input_ids` is not available, but got '{type(self.tokenizer)}'."
                )
            if not self._EXPECTED_INPUTS_KEYS.issubset(inputs.keys()):
                raise ValueError(
                    f"`inputs` should have at least {self._EXPECTED_INPUTS_KEYS} keys, but got {inputs.keys()}."
                )

            input_ids = self.tokenizer(inputs["text"], truncation=True, return_tensors="pt")["input_ids"]

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        return {"input_ids": input_ids}

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        pad_token_id: Optional[int] = None,
        use_amp: bool = True,
        shift_labels: bool = True,
    ) -> Dict[str, Any]:
        amp_dtype = torch.bfloat16 if self.model.dtype == torch.bfloat16 else None
        with torch.autocast(self.model.device.type, dtype=amp_dtype, enabled=use_amp):
            logits = self.model(input_ids=model_inputs["input_ids"]).logits
        labels = model_inputs["input_ids"].clone()

        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        return {
            "logits": logits,
            "labels": labels,
        }

    def forward(self, model_inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        with self.device_placement():
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs = self._forward(model_inputs, **forward_params)

        return model_outputs

    def postprocess(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        logits = model_outputs["logits"]
        labels = model_outputs["labels"]

        return {
            "loss": F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=self.tokenizer.pad_token_id
            )
            .cpu()
            .item()
        }
