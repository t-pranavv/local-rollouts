# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from itertools import product
from random import Random
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from phyagi.datasets.rl.special_tokens import get_mask_token, get_special_token

CHATML_CHAT_TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}"
PHI_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"
GEN_TEMPLATE_PATCHES = {
    "microsoft/phi-3-mini-128k-instruct": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n'}}{% generation %}{{message['content'] + '<|end|>'}}{% endgeneration %}{{'\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
    "microsoft/phi-3-mini-4k-instruct": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n'}}{% generation %}{{message['content'] + '<|end|>'}}{% endgeneration %}{{'\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
    "microsoft/phi-4": "{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}{% generation %}{{message['content'] + '<|im_end|>'}}{% endgeneration %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}",
}


@functools.cache
def _generate_system_messages() -> Tuple[str, ...]:
    messages = (
        "You are an AI assistant that helps people find information.",
        "You're Phi, a large language model trained by Microsoft to help users",
        "You are a kind and helpful assistant. Respond only with helpful, ethical responses, and avoid harmful or inappropriate content.",
        "You are a kind, smart, capable, and helpful assistant. Give answers that aid the user, while being extremely technically adept",
        "you are a good assistant do your best to answer questions",
        "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
        "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
        "You follow user instruction extremely well",
        "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    )
    return messages


@functools.cache
def _generate_chat_templates(special_token_format: str) -> List[Tuple[str, str, str, str, str]]:
    user_token = get_special_token(special_token_format, "user")
    assistant_token = get_special_token(special_token_format, "assistant")
    end_token = get_special_token(special_token_format, "end")

    # Additional blank roles are added to increase the variety of templates
    base_roles = [
        ("Q", "A", "\n"),
        ("Question", "Answer", "\n"),
        ("Message", "Response", "\n"),
        ("Query", "Response", "\n"),
        ("Query", "Reply", "\n"),
        ("Problem", "Solution", "\n"),
        ("Ask", "Answer", "\n"),
        ("Input", "Output", "\n"),
        ("Message", "Response", "\n"),
        ("Instruction", "Response", "\n"),
        ("Customer", "Support", "\n"),
        ("", "", "\n"),
        ("", "", "\n"),
        ("", "", "\n"),
        ("", "", "\n"),
        ("", "", "\n"),
    ]

    # Extend the base roles with additional variations
    roles = base_roles + [
        (f"{'#' * n} {role[0]}", f"{'#' * m} {role[1]}", "\n")
        for n in range(1, 5)
        for m in range(1, 5)
        for role in base_roles
    ]

    templates = []
    options = [True, False]
    combinations = list(product(options, repeat=8))
    random_seed = Random(42)

    for role in roles:
        for combination in combinations:
            (
                add_user_newline,
                add_assistant_newline,
                add_user_colon,
                add_assistant_colon,
                add_newline_after_user_message,
                add_newline_after_assistant_message,
                add_newline_after_user_token,
                add_space_after_user_token,
            ) = combination

            if random_seed.randint(0, 1) == 0:
                newline_after_user_token = "\n" * random_seed.randint(0, 10) if add_newline_after_user_token else ""
                newline_after_user_token = (
                    newline_after_user_token + " " if add_space_after_user_token else newline_after_user_token
                )
            else:
                newline_after_user_token = (
                    " " * random_seed.randint(0, 10) if add_space_after_user_token else newline_after_user_token
                )
                newline_after_user_token = (
                    newline_after_user_token + "\n" * random_seed.randint(0, 10)
                    if add_newline_after_user_token
                    else newline_after_user_token
                )

            user_string = (
                user_token
                + newline_after_user_token
                + role[0]
                + (":" if add_user_colon else " ")
                + ("\n" if add_user_newline else "")
            )

            assistant_string = "\n" if add_newline_after_user_message else " "
            assistant_string = (
                assistant_string
                + role[1]
                + (":" if add_assistant_colon else " ")
                + ("\n" if add_assistant_newline else "")
            )
            assistant_string += "\n" * random_seed.randint(0, 10) if add_newline_after_assistant_message else " "

            templates.append((user_string, assistant_string, assistant_token, "", end_token))

    return templates


def _format_message(
    message: str,
    role: str,
    special_token_format: str,
    random_seed: Optional[Random],
    shuffle: bool = True,
    add_mask_tokens: bool = False,
) -> str:
    SHUFFLE_PROB = 0.1

    if message is None:
        message = ""

    user_token = get_special_token(special_token_format, "user")
    assistant_token = get_special_token(special_token_format, "assistant")
    end_token = get_special_token(special_token_format, "end")
    system_token = get_special_token(special_token_format, "system")

    mask_start_token = get_mask_token("start")
    mask_end_token = get_mask_token("end")

    template = (user_token, "", assistant_token, "", end_token)
    end_str = template[-1]

    if role == "user":
        if shuffle and random_seed.uniform(0, 1) < SHUFFLE_PROB:
            template = random_seed.sample(_generate_chat_templates(format), 1)[0]
        if add_mask_tokens:
            return mask_start_token + template[0] + message + template[1] + end_str + mask_end_token
        return template[0] + message + template[1] + end_str

    if role == "assistant":
        return template[2] + message + end_str

    if role == "system":
        if add_mask_tokens:
            return mask_start_token + system_token + message + end_str + mask_end_token
        return system_token + message + end_str

    raise ValueError(f"`role` must be 'user', 'assistant', or 'system', but got '{role}'.")


def apply_chat_template(
    example: Dict[str, List[Dict[str, str]]],
    special_token_format: str,
    shuffle: bool = True,
    add_mask_tokens: bool = False,
    system_message_format: str = "random",
    system_message_prob: float = 0.25,
) -> Dict[str, str]:
    """Apply chat template to a given example.

    An example is a dictionary with task-related keys, e.g., ``prompt``, ``completion`` ``chosen``,
    ``rejected``, etc. each containing a list of messages, where each message is a dictionary with
    ``role`` and ``content``. For ``prompt`` key, an optional system message is additionally added
    ``system_message_format`` and ``system_message_prob`` arguments.

    Args:
        example: Example to apply chat template.
        special_token_format: Format of the special tokens, e.g., ``phi`` or ``chatml``.
        shuffle: Whether to shuffle templates.
        add_mask_tokens: Whether to add mask tokens.

    Returns:
        Example with chat template applied.

    """

    VALID_EXAMPLE_KEYS_FOR_SYSTEM = ["prompt"]

    formatted_example = {}

    for key in example.keys():
        formatted_messages = []
        random_seed = Random(len(example[key][0]["content"]))

        if key in VALID_EXAMPLE_KEYS_FOR_SYSTEM and example[key][0]["role"] != "system":
            if random_seed.uniform(0, 1) >= (1 - system_message_prob):  # 25% chance by default
                if system_message_format == "random":
                    system_message = random_seed.sample(_generate_system_messages(), 1)[0]
                elif system_message_format == "cot_final":
                    system_message = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
                elif system_message_format == "cot_tools":
                    raise NotImplementedError("`system_message_format='cot_tools'` is not implemented yet.")
                else:
                    raise ValueError(
                        f"`system_message_format` must be one of ['random', 'cot_final', 'cot_tools'], but got '{system_message_format}'."
                    )

                example[key].insert(0, {"role": "system", "content": system_message})

        for message in example[key]:
            formatted_messages.append(
                _format_message(
                    message["content"],
                    message["role"],
                    special_token_format,
                    random_seed=random_seed,
                    shuffle=shuffle,
                    add_mask_tokens=add_mask_tokens,
                )
            )

        formatted_example[key] = "".join(formatted_messages)

    return formatted_example


def patch_tokenizer_generation_tag(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Patch a tokenizer if it does not have support for generating assistant masks (by using % generation % Jinja tags).

    Args:
        tokenizer: Tokenizer to patch.

    Returns:
        Tokenizer with patched generation tags.

    """

    chat_template = GEN_TEMPLATE_PATCHES.get(tokenizer.name_or_path.lower()) or tokenizer.chat_template
    if chat_template is None:
        raise ValueError(f"'{tokenizer.name_or_path.lower()}' must have a `chat_template`, but got None.")
    if r"{% generation %}" not in chat_template:
        raise ValueError(
            f"'{tokenizer.name_or_path.lower()}' does not have '% generation %' Jinja tags, must be one of {list(GEN_TEMPLATE_PATCHES.keys())}."
        )

    tokenizer.chat_template = chat_template

    return tokenizer
