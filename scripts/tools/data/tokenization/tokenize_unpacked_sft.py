# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import re
from pathlib import Path
from random import Random
from typing import Any, Dict, List

import numpy as np
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

def linearize_messages(row: Dict[str, Any], 
                        mask_start_token: str = "<|dummy_31|>",
                        mask_end_token: str = "<|dummy_32|>"
):   
    # used to linearize tool use scenarios
    messages = row["raw_messages"]
    assert messages[0]["role"]=="user"
    processed_messages = [messages[0]]
    content = ""
    for message in messages[1:]:
        if "cot" in message and message["cot"] is not None:
            content += f"<think> {message['cot']} </think>"
            content += message['content']
        else:
            assert message["role"]=="user"
            text = re.sub(r"(<tool_result>)([\s\S]*?)(</tool_result>)",
                        f"{mask_end_token}\\1\\2\\3{mask_start_token}",
                        message['content'],)
            content += f"{text}"

    processed_messages.append(dict(role="assistant", content=content)) 
    row["messages"] = processed_messages
    return row 

def _tokenize_and_mask_response(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    mask_start_token: str = "<|dummy_31|>",
    mask_end_token: str = "<|dummy_32|>",
    add_eos: bool = True,
    eos_token: str = "<|endoftext|>",
    replace_think_token: bool = False,
    drop_cot: bool = False,
) -> Dict[str, np.ndarray]:
    if not isinstance(messages, list):
        raise ValueError(f"`messages` must be a list, but got {type(messages)}.")
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("No assistant message found in the input messages.")

    messages = messages[0]
    ans_start_id, ans_end_id = tokenizer.encode(mask_start_token + mask_end_token)

    # message is everything upto last assistant message
    if messages[-1]["role"] != "assistant":
        for i, m in enumerate(messages[::-1]):
            if m["role"] == "assistant":
                break
        messages = messages[: len(messages) - i]

    if any(
        t in msg["content"]
        for t in ["<|im_start|>", "<|im_end|>"] #, mask_start_token, mask_end_token, 
        for msg in messages
    ):
        print("Skipping sample due to special tokens or template tokens found inside the input text...")
        return {"input_ids": [], "labels": [], "seqlen": []}

    input_text = tokenizer.apply_chat_template([messages], tokenize=False)[0]
    if replace_think_token:
        input_text = input_text.replace("<|dummy_86|>", "<think>").replace("<|dummy_87|>", "</think>")
    if drop_cot:
        input_text = re.sub(r"<\|dummy_86\|>[\s\S]*?<\|dummy_87\|>", "", input_text)
        input_text = re.sub(r"<think>[\s\S]*?</think>", "", input_text)

    if add_eos:
        input_text += eos_token
    input_text = re.sub(
        r"(<\|im_start\|>assistant<\|im_sep\|>)([\s\S]*?)(<\|im_end\|>)", #"(<\\|assistant\\|>)([\\s\\S]*?)(<\\|end\\|>)",
        f"{mask_start_token}\\1\\2\\3{mask_end_token}",
        input_text,
    )

    input_ids = tokenizer([input_text], add_special_tokens=False, return_tensors="np",)[
        "input_ids"
    ][0]

    mask_start = (input_ids == ans_start_id).cumsum()
    mask_end = (input_ids == ans_end_id).cumsum()

    answer_mask = mask_start - mask_end
    assert answer_mask.max() == 1, f"Masking error!, #{input_text}, #{mask_start}, #{mask_end}, #{answer_mask.max()}"

    added_tokens = np.where((input_ids == ans_start_id) | (input_ids == ans_end_id))[0]
    input_ids = np.delete(input_ids, added_tokens)
    answer_mask = np.delete(answer_mask, added_tokens)

    labels = input_ids.copy()
    labels[answer_mask == 0] = -100

    return {"input_ids": [input_ids], "labels": [labels], "seqlen": [input_ids.shape[0]]}


def _greedy_pack_sequences(
    rows: Dict[str, Any],
    max_seqlen: int,
    pad_token_id: int,
    truncate_longer_texts: bool = False,
) -> Dict[str, Any]:
    input_ids, labels, seqlens = [rows[k] for k in ["input_ids", "labels", "seqlen"]]

    p_inputs, p_labels = [], []
    pack_tokens, pack_labels = [], []

    total, dropped = 0, 0
    for i, seqlen in enumerate(seqlens):
        total += 1

        if seqlen >= max_seqlen:
            if not truncate_longer_texts:
                dropped += 1
                continue

            if not all(label == -100 for label in labels[i][:max_seqlen]):
                p_inputs.append(input_ids[i][:max_seqlen])
                p_labels.append(labels[i][:max_seqlen])
        else:
            if len(pack_tokens) + seqlen <= max_seqlen:
                pack_tokens.extend(input_ids[i])
                pack_labels.extend(labels[i])
            else:
                padding_needed = max_seqlen - len(pack_tokens)

                if padding_needed > 0:
                    pack_tokens.extend([pad_token_id] * padding_needed)
                    pack_labels.extend([-100] * padding_needed)

                p_inputs.append(pack_tokens)
                p_labels.append(pack_labels)

                pack_tokens = input_ids[i].copy()
                pack_labels = labels[i].copy()

    print(f'dropped {dropped} / {total} samples.')

    # Handle the last pack
    if pack_tokens:
        padding_needed = max_seqlen - len(pack_tokens)

        if padding_needed > 0:
            pack_tokens.extend([pad_token_id] * padding_needed)
            pack_labels.extend([-100] * padding_needed)

        p_inputs.append(pack_tokens)
        p_labels.append(pack_labels)

    p_inputs = [np.array(inp, dtype=np.int32) for inp in p_inputs]
    p_labels = [np.array(l, dtype=np.int32) for l in p_labels]

    return {"input_ids": p_inputs, "labels": p_labels}

def get_tool_system_prompt(system_message_format):
    if system_message_format.endswith('_ci'):
        tool_details = '''Execute Python code. Available packages: numpy, scipy, sympy, pandas, matplotlib, requests'''
        response_format = '''Python code should be in markdown format. Format: <tool_call> 
    ```python
    {code here}
    ``` 
    </tool_call>'''
    else:
        raise ValueError(f'system message for {system_message_format} not defined')
    

    prompt_template = '''\
You are a reasoning language model that can reach precise answers through careful reasoning and tool use when needed. 

Structure Rules:
1. All reasoning goes between <think> and </think> (thinking block). 
2. Within the thinking block, whenever a tool would improve your answer, invoke it using <tool_call>...</tool_call> instead of relying solely on memory.
3. Issue one valid <tool_call>...</tool_call> at a time; further tool calls can be sequentially interleaved throughout the reasoning process. 
4. After each tool call, the result of the tool call will be provided in the <tool_result>...</tool_result> tags.
5. Provide the final answer for the user inside the <answer> </answer> tags.
6. Stop the generation only after reaching the final answer.

You can utilize the tools as many times as required. For example, <think> reasoning here  </think> <tool_call> tool call here </tool_call> <tool_result> output of tool call </tool_result> <think> reasoning process here </think> <answer> final answer here </answer>.
# RESPONSE FORMAT FOR TOOL CALLS

{response_format}

# AVAILABLE TOOLS

{tool_details}
'''
    
    return prompt_template.replace('{tool_details}', tool_details).replace('{response_format}', response_format)

def _add_system_messages(
    row: Dict[str, Any],
    rng: Random,
    add_prob: float = 0.25,
    system_message_format: str = "random",
) -> Dict[str, Any]:
    system_prompts = (
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
    reasoning_system_prompt_dummy = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
    reasoning_system_prompt = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
    reasoning_system_prompt_nocot = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. At the end, systematically present the final solution that you deem correct. Now, try to solve the following question through the above guidelines:"

    messages = row["messages"]

    if messages[0]["role"] != "system" and rng.random() <= add_prob:
        if system_message_format == "random":
            system_message = rng.sample(system_prompts, 1)[0]
        elif system_message_format == 'cot':
            system_message = reasoning_system_prompt_dummy
        elif system_message_format == 'cot_mix':
            if rng.random() <= 0.1:
                system_message = rng.sample(system_prompts, 1)[0]
            else:
                system_message = reasoning_system_prompt_dummy
        elif system_message_format == 'cot_final':
            system_message = reasoning_system_prompt
        elif system_message_format == 'reason_nocot':
            system_message = reasoning_system_prompt_nocot
        elif 'tool_use' in system_message_format:
            system_message = get_tool_system_prompt(system_message_format)
        else:
            raise ValueError(f"Invalid system message format: `{system_message_format}`.")

        messages.insert(0, {"role": "system", "content": system_message})

    return {"messages": messages}


def _save_to_file(arrays: List[np.ndarray], output_dir: Path, output_fname: str, seq_len: int) -> int:
    total_size = sum(len(arr) for arr in arrays)

    shape = (len(arrays) * seq_len,)
    memmap_array = np.empty(shape, dtype=np.int32)

    for i in tqdm.tqdm(range(len(arrays))):
        st = i * seq_len
        et = st + seq_len
        memmap_array[st:et] = arrays[i]

    np.save(str(output_dir / output_fname), memmap_array)

    return total_size, arrays[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a .jsonl dataset into NumPy arrays.")

    parser.add_argument(
        "-i", "--input_files", type=Path, nargs="+", required=True, help="Path to the input .jsonl file(s)."
    )

    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Directory to save the .npy files.")

    parser.add_argument(
        "-t",
        "--pretrained_tokenizer_name_or_path",
        type=Path,
        required=True,
        help="Pre-trained tokenizer name or path.",
    )

    parser.add_argument("-n", "--num_proc", type=int, default=16, help="Number of processes to use for dataset.map().")

    parser.add_argument("-sl", "--seq_len", type=int, default=16384, help="Maximum sequence length for packing.")

    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed for shuffling and sampling.")

    parser.add_argument(
        "-ns", "--no_shuffle", action="store_true", help="Do not shuffle the dataset before processing."
    )

    parser.add_argument(
        "-asm", "--add_system_messages", action="store_true", help="Add system messages to the dataset."
    )

    parser.add_argument(
        "-smf",
        "--system_message_format",
        type=str,
        default="cot_mix",
        choices=["random", "cot", "cot_final", "cot_mix", "reason_nocot", "tool_use_ci"],
        help="Format of the system messages to add.",
    )

    parser.add_argument(
        "-smp",
        "--system_message_prob",
        type=float,
        default=0.25,
        help="Probability of adding a system message to each sample.",
    )
    parser.add_argument(
        "-tlt",
        "--truncate_longer_texts",
        action="store_true",
        help="Truncate longer texts to the maximum sequence length instead of skipping them.",
    )
    parser.add_argument("--drop_cot", action="store_true", help="drop the thinking block from the data before tokenization")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rng = Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer_name_or_path, trust_remote_code=True)
    dataset = load_dataset("json", data_files=[str(p) for p in args.input_files], split="train")

    if not args.no_shuffle:
        print("Shuffling dataset...")
        dataset = dataset.shuffle(seed=args.seed)

    if args.drop_cot:
        args.system_message_format = 'reason_nocot'

    mask_start_token = "<|dummy_31|>"
    mask_end_token = "<|dummy_32|>"
    
    if 'tool_use' in args.system_message_format:
        print('Linearizing raw messages to processed format')
        dataset = dataset.map(
            linearize_messages,
            num_proc=args.num_proc,
            fn_kwargs={"mask_start_token": mask_start_token, "mask_end_token": mask_end_token},
            desc="Processing the raw data",
        )

    if args.add_system_messages:
        dataset = dataset.map(
            _add_system_messages,
            num_proc=args.num_proc,
            fn_kwargs={
                "rng": rng,
                "add_prob": args.system_message_prob,
                "system_message_format": args.system_message_format,
            },
            desc="Adding system messages...",
        )

    # Tokenize and calculate mask
    dataset = dataset.map(
        lambda row: _tokenize_and_mask_response(row["messages"], tokenizer=tokenizer,
                                                mask_start_token=mask_start_token, mask_end_token=mask_end_token,
                                                replace_think_token=(args.system_message_format=='cot_final'),
                                                drop_cot=args.drop_cot),
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        batch_size=1,
        batched=True,
        desc="Tokenizing and masking responses...",
    )

    dataset = dataset.map(
        _greedy_pack_sequences,
        batched=True,
        batch_size=256,
        num_proc=args.num_proc,
        fn_kwargs={
            "max_seqlen": args.seq_len,
            "pad_token_id": tokenizer.pad_token_id,
            "truncate_longer_texts": args.truncate_longer_texts,
        },
        remove_columns=dataset.column_names,
        desc="Packing sequences...",
    )
    dataset_dict = dataset.to_dict()

    total_tokens, input_ids_subset = _save_to_file(
        dataset_dict["input_ids"], args.output_dir, "train.npy", args.seq_len
    )
    _, labels_subset = _save_to_file(dataset_dict["labels"], args.output_dir, "train_labels.npy", args.seq_len)
    print("Total tokens:", total_tokens)

    # Preview
    n_preview = 2000
    print(f"Data preview (first {n_preview} tokens):")
    words = tokenizer.batch_decode(input_ids_subset[:n_preview])
    masks = [True if l == -100 else False for l in labels_subset[:n_preview]]

    for word, mask in zip(words, masks):
        if mask:
            print(f"\033[91m{word}\033[0m", end=' ')
        else:
            print(word, end=' ')
    
    # preview = list(
    #     zip(
    #         tokenizer.batch_decode(input_ids_subset[:n_preview]),
    #         tokenizer.batch_decode(tokenizer.encode(" ") if l == -100 else l for l in labels_subset[:n_preview]),
    #     )
    # )
    # print(f"(Tokens, Labels [empty means masked]): {preview}")
    
    print(f"Processing complete. Files saved to {args.output_dir}")
