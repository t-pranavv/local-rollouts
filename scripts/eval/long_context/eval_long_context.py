# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
from pathlib import Path

import datasets
import huggingface_hub
import numpy as np
import pandas as pd
import torch
from source.babilong.prompts import (
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LONGROPE_PARAMS = {
    "long_factor": [
        1.0700000524520874,
        1.1200000047683716,
        1.149999976158142,
        1.4199999570846558,
        1.5699999332427979,
        1.7999999523162842,
        2.129999876022339,
        2.129999876022339,
        3.009999990463257,
        5.910000324249268,
        6.950000286102295,
        9.070000648498535,
        9.930000305175781,
        10.710000038146973,
        11.130000114440918,
        14.609999656677246,
        15.409998893737793,
        19.809999465942383,
        37.279998779296875,
        38.279998779296875,
        38.599998474121094,
        40.12000274658203,
        46.20000457763672,
        50.940006256103516,
        53.66000747680664,
        54.9373893737793,
        56.89738845825195,
        57.28738784790039,
        59.98738479614258,
        60.86738586425781,
        60.887386322021484,
        61.71739196777344,
        62.91739273071289,
        62.957393646240234,
        63.41739273071289,
        63.8173942565918,
        63.83739471435547,
        63.897396087646484,
        63.93739700317383,
        64.06739807128906,
        64.11434936523438,
        64.12435150146484,
        64.15435028076172,
        64.19435119628906,
        64.24435424804688,
        64.57435607910156,
        64.69000244140625,
        64.76000213623047,
    ],
    "short_factor": [
        1.1,
        1.1,
        1.1,
        1.3000000000000003,
        1.3500000000000003,
        1.3500000000000003,
        1.4000000000000004,
        1.5500000000000005,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.000000000000001,
        2.0500000000000007,
        2.0500000000000007,
        2.0500000000000007,
        2.0500000000000007,
        2.0500000000000007,
        2.0500000000000007,
        2.1000000000000005,
        2.1000000000000005,
        2.1500000000000004,
        2.25,
        2.25,
        2.25,
        2.25,
        2.25,
        2.3999999999999995,
        2.4499999999999993,
        2.499999999999999,
        2.6999999999999984,
        2.6999999999999984,
        2.7499999999999982,
        2.799999999999998,
        2.8999999999999977,
        3.049999999999997,
    ],
    "type": "longrope",
}


def _interpolate_array(arr, new_length):
    n = len(arr)

    original_indices = np.arange(n)
    new_indices = np.linspace(0, n - 1, new_length)

    return np.interp(new_indices, original_indices, arr)


LONGROPE_PARAMS["long_factor"] = list(_interpolate_array(np.array(LONGROPE_PARAMS["long_factor"]), 64))
LONGROPE_PARAMS["short_factor"] = list(_interpolate_array(np.array(LONGROPE_PARAMS["short_factor"]), 64))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a model on Babilong evaluations.")

    parser.add_argument("pretrained_model_name_or_path", type=str, help="Pre-trained model name or path.")

    parser.add_argument(
        "-rst",
        "--rope_scaling_type",
        type=str,
        default="none",
        choices=["none", "yarn", "yarn_qwen", "linear", "llama3", "longrope"],
        help="Type of rope scaling to apply.",
    )

    parser.add_argument("-rt", "--rope_theta", type=int, default=None, help="Theta value for rope scaling.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    rope_scaling_type = args.rope_scaling_type
    rope_theta = args.rope_theta
    if rope_theta is not None:
        if rope_scaling_type is not None:
            raise ValueError("`rope_theta` and `rope_scaling_type` cannot be set at the same time.")

    # Define the rope scaling parameters based on the specified type
    rope_scaling = None
    if rope_scaling_type == "yarn":
        rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 16384,
            "rope_type": "yarn",
        }
    elif rope_scaling_type == "yarn_qwen":
        rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_type": "yarn",
        }
    elif rope_scaling_type == "linear":
        rope_scaling = {"factor": 4.0, "rope_type": "linear"}
    elif rope_scaling_type == "llama3":
        rope_scaling = {
            "factor": 4.0,
            "high_freq_factor": 2.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 16384,
            "rope_type": "llama3",
        }
    elif rope_scaling_type == "longrope":
        rope_scaling = LONGROPE_PARAMS

    # Load the model with specified parameters
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }

    if rope_scaling:
        model_kwargs["rope_scaling"] = rope_scaling
    elif rope_theta is not None:
        model_kwargs["rope_theta"] = rope_theta

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)

    # Define generation and prompt configuration
    gen_config = {
        "max_new_tokens": 20,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "num_beams": 1,
        "do_sample": False,
    }
    prompt_flags = {"chat_template": True, "instruction": True, "examples": True, "post_prompt": True}

    tasks = ["qa1"]
    lengths = ["0k", "8k", "16k", "32k", "64k"]
    results_dir = Path("babilong_evals")

    # Iterate through tasks and lengths, and evaluate the model on each
    for task in tqdm(tasks):
        prompt_cfg = {
            "template": DEFAULT_TEMPLATE,
            **{k: DEFAULT_PROMPTS[task][k] if prompt_flags[k] else "" for k in prompt_flags if k != "chat_template"},
            "chat_template": prompt_flags["chat_template"],
        }
        prompt_name = "_".join([f"{k}_yes" if prompt_flags[k] else f"{k}_no" for k in prompt_flags])

        for split in tqdm(lengths):
            data = datasets.load_dataset("RMT-team/babilong", split)[task]

            results_path = (
                results_dir
                / args.pretrained_model_name_or_path
                / f"{task}_{split}_{args.rope_scaling_type}_{args.rope_theta}_{prompt_name}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)

            config_path = results_path.with_suffix(".json")
            with config_path.open("w") as f:
                json.dump({"prompt": prompt_cfg, "generate_kwargs": gen_config}, f, indent=4)

            df = pd.DataFrame(columns=["target", "output", "question"])

            for sample in tqdm(data, desc=f"{task}-{split}"):
                input_text = get_formatted_input(
                    sample["input"],
                    sample["question"],
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )

                if prompt_flags["chat_template"]:
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_text}], add_generation_prompt=True, return_tensors="pt"
                    ).to(model.device)
                    inputs = {"input_ids": tokens}
                else:
                    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_config)

                output_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

                df.loc[len(df)] = [sample["target"], output_text, sample["question"]]
                df.to_csv(results_path, index=False)
