# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.pipelines.base import ChunkPipeline


class _MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self._stop_tokens = stop_tokens
        self._max_stop_tokens = stop_tokens.shape[-1]

        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self._max_stop_tokens :].unsqueeze(1), self._stop_tokens)
        is_stop_token = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        stop_token_idx = torch.any(is_stop_token, dim=1)
        stop_token_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[stop_token_idx & stop_token_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


class TextGenerationPipeline(ChunkPipeline):
    """Text generation pipeline.

    This pipeline is used to generate text given an input sequence. For example, given an input sequence
    'I love Paris', the pipeline will generate a sequence of text that continues the input sequence.

    """

    _EXPECTED_INPUTS_KEYS = {"text"}

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        **kwargs,
    ) -> None:
        # Make a deep copy of the tokenizer to prevent side-effects of changing its attributes
        tokenizer = copy.deepcopy(tokenizer)
        if tokenizer.eos_token is None:
            raise ValueError("`tokenizer.eos_token` must be provided, but got None.")

        # Ensure that tokenizer has the correct maximum length, truncation and padding settings
        tokenizer.model_max_length = getattr(model.config, "n_positions", tokenizer.model_max_length)
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        super().__init__(model, tokenizer, device=device, **kwargs)

    def _sanitize_parameters(
        self,
        n_samples: int = 1,
        return_inputs: bool = True,
        use_attention_mask: bool = True,
        pad_token_id: Optional[int] = None,
        use_amp: bool = True,
        **generate_kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        preprocess_kwargs = {}
        preprocess_kwargs["n_samples"] = n_samples
        preprocess_kwargs["return_inputs"] = return_inputs
        preprocess_kwargs["use_attention_mask"] = use_attention_mask

        forward_kwargs = {}
        forward_kwargs["pad_token_id"] = pad_token_id
        forward_kwargs["use_amp"] = use_amp
        forward_kwargs.update(generate_kwargs)

        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(
        self,
        inputs: Dict[str, Any],
        n_samples: int = 1,
        return_inputs: bool = True,
        use_attention_mask: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        if not isinstance(inputs, dict):
            raise TypeError(f"`inputs` must be a dict, but got '{type(inputs)}'.")
        if not self._EXPECTED_INPUTS_KEYS.issubset(inputs.keys()):
            raise ValueError(
                f"`inputs` should have at least {self._EXPECTED_INPUTS_KEYS} keys, but got {inputs.keys()}."
            )

        model_inputs = self.tokenizer(inputs["text"], truncation=True, return_tensors="pt")
        label = inputs.pop("label", None)

        for i in range(n_samples):
            yield {
                "inputs": inputs if return_inputs else None,
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"] if use_attention_mask else None,
                "label": label,
                "is_last": i == n_samples - 1,
            }

    def _forward(
        self, model_inputs: Dict[str, Any], pad_token_id: Optional[int] = None, use_amp: bool = True, **generate_kwargs
    ) -> Dict[str, Any]:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        # If `use_attention_mask = False` and `batch_size > 1`, the `attention_mask` is
        # passed as a list of None, which is not supported by models
        if isinstance(attention_mask, list):
            attention_mask = None if attention_mask[0] is None else attention_mask

        # If `attention_mask` exists and is needed, it will be passed to `generate()`
        # Note that some models do not have an attention mask argument
        if attention_mask is not None and batch_size > 1:
            generate_kwargs["attention_mask"] = attention_mask

        # If `stop_tokens` are supplied, we add the `eos_token` to them and
        # tokenize to create the stopping criteria
        stop_tokens = generate_kwargs.pop("stop_tokens", None)
        stop_tokens = stop_tokens + [self.tokenizer.eos_token] if stop_tokens is not None else None

        stop_tokens_ids = (
            self.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
            if stop_tokens is not None
            else None
        )

        # Llama-based tokenizers need to remove the initial space for stop tokens encoding
        if isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            bos_token_id = self.tokenizer.bos_token_id
            space_token_id = self.tokenizer.convert_tokens_to_ids("▁")

            if stop_tokens_ids is not None:
                while stop_tokens_ids[0, 0] in [bos_token_id, space_token_id]:
                    # Tokens in the batch must be equal or padding tokens
                    cond = stop_tokens_ids[:, 0] == stop_tokens_ids[0, 0]
                    cond |= stop_tokens_ids[:, 0] == self.tokenizer.pad_token_id
                    if not torch.all(cond):
                        raise ValueError(
                            f"{stop_tokens} contain a token that is not equal to the first token in the batch."
                        )

                    stop_tokens_ids = stop_tokens_ids[:, 1:]

        amp_dtype = torch.bfloat16 if self.model.dtype == torch.bfloat16 else None
        with torch.autocast(self.model.device.type, dtype=amp_dtype, enabled=use_amp):
            if stop_tokens_ids is not None:
                stop_tokens_ids = stop_tokens_ids.to(self.model.device)
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=batch_size)]
                )

            generated_sequence = self.model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        if stop_tokens_ids is not None:
            # Since we use an external `generate()`, we need to access the stopping criteria class
            # to gather the sequence index of the stop token
            stop_tokens_idx = generate_kwargs["stopping_criteria"][0].stop_tokens_idx

            # If a stop token was produced, we need to remove its length from the found index,
            # however there might be a chance that the stop token was not produced and the index
            # returned is the length of the generated sequence
            stop_tokens_idx = torch.where(
                stop_tokens_idx > 0, stop_tokens_idx - stop_tokens_ids.shape[-1], generated_sequence.shape[-1]
            )
        else:
            stop_tokens_idx = torch.full(
                (batch_size,), generated_sequence.shape[-1], dtype=torch.long, device=self.model.device
            )

        return {
            "inputs": model_inputs["inputs"],
            "generated_sequence": generated_sequence,
            "stop_tokens_idx": stop_tokens_idx,
            "label": model_inputs["label"],
            "is_last": model_inputs["is_last"],
        }

    def forward(self, model_inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        with self.device_placement():
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                model_outputs = self._forward(model_inputs, **forward_params)

        return model_outputs

    def postprocess(self, model_outputs: List[Dict[str, Any]]) -> Dict[str, Union[Any, List[Any]]]:
        # `_input` and `label` are the same regardless of the number of samples
        _input = model_outputs[0]["inputs"]
        label = model_outputs[0]["label"]

        generated_texts = []
        for model_output in model_outputs:
            generated_sequence = model_output["generated_sequence"].squeeze(0)
            stop_tokens_idx = model_output["stop_tokens_idx"]

            # Decode the generated sequence and cut off any text after the stop token (including the stop token)
            generated_texts.append(
                self.tokenizer.decode(generated_sequence[:stop_tokens_idx], skip_special_tokens=True)
            )

        return {"input": _input, "label": label, "responses": generated_texts}
