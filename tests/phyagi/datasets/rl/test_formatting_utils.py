# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import AutoTokenizer

from phyagi.datasets.rl.formatting_utils import (
    GEN_TEMPLATE_PATCHES,
    apply_chat_template,
    patch_tokenizer_generation_tag,
)
from phyagi.datasets.rl.special_tokens import get_mask_token


def test_apply_chat_template():
    example = {
        "prompt": [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]
    }

    formatted_example = apply_chat_template(example, special_token_format="phi", shuffle=False, add_mask_tokens=False)
    assert isinstance(formatted_example, dict)
    assert "prompt" in formatted_example
    assert isinstance(formatted_example["prompt"], str)

    # Ensure mask tokens are applied
    formatted_with_masks = apply_chat_template(example, special_token_format="phi", shuffle=False, add_mask_tokens=True)
    assert get_mask_token("start") in formatted_with_masks["prompt"]
    assert get_mask_token("end") in formatted_with_masks["prompt"]


def test_patch_tokenizer_generation_tag():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
    patched = patch_tokenizer_generation_tag(tokenizer)
    assert "{% generation %}" in patched.chat_template
    assert patched.chat_template == GEN_TEMPLATE_PATCHES["microsoft/phi-3-mini-4k-instruct"]
