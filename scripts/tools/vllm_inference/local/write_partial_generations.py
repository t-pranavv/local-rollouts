from __future__ import annotations

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phigen.datagen.local_dataset import local_dataset
from phigen.logging import set_log_level
from run_inference_on_vllm_server import VLLMLocalInferenceFlow, VLLMLocalResponseAgent, VLLMLocalToolResponseAgent

__all__ = ["VLLMLocalResponseAgent", "VLLMLocalToolResponseAgent"]

set_log_level("INFO")


def eval_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("--local_db_dir", type=str, default=None, help="Path to local dataset directory.")
    parser.add_argument("--local_db_name", type=str, default=None, help="Name of the local dataset.")
    parser.add_argument("-i", "--prompts_file", type=str, help="Path to the prompts jsonl.")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="Field within the prompts jsonl to use as prompt. (default: prompt)",
    )
    parser.add_argument("-o", "--output_dir", type=str, help="Path to save results.")
    parser.add_argument(
        "--agent_cls",
        type=str,
        default="VLLMLocalResponseAgent",
        help="Name of the agent class to use for inference.",
    )
    parser.add_argument(
        "--api_type",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="API type to use for inference.",
    )
    parser.add_argument(
        "--model_utils_name",
        type=str,
        default="phi-think",
        help="Utility class to use for model-specific operations.",
    )
    parser.add_argument("--system_message", type=str, default=None, help="System message to add to the prompt.")
    parser.add_argument("--thinking_model", action="store_true", help="Add thinking generation prompt.")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Options")
    parser = eval_args(parser)
    args = parser.parse_args()

    args.agent_cls = globals()[args.agent_cls]
    local_db = local_dataset(args.local_db_name, dir=args.local_db_dir)
    with local_db:
        VLLMLocalInferenceFlow.write(args, remove_cols=["idx"])
