from __future__ import annotations

import os
import uuid
import json
import asyncio
import argparse
import random
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Any
from transformers import AutoTokenizer

from model_utils import ModelUtilsFactory
from prompts.prompt_templates import (
    DEFAULT_SYSTEM_PROMPTS,
    ci_tool_details,
    ci_response_format,
    bfcl_response_format,
)
from tools import ToolExecutor

from phigen.client.model_client import complete_text_full_async
from phigen.client.models.model_types import ModelKind, ModelMetadata
from phigen.datagen.record import phield
from phigen.datagen.agent import AssistantAgent, Flow
from phigen.datagen.shared_data import load_file_shared
from phigen.datagen.local_dataset import local_dataset, temp_dataset
from phigen.local_inference.vllm_local_client import _setup_vllm_local_client
from phigen.logging import logger, set_log_level

set_log_level("INFO")


def vllm_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("--base_ip", type=str, default="localhost", help="Base IP address for the VLLM server.")
    parser.add_argument("--base_port", type=int, default=9000, help="Base port number.")
    parser.add_argument("--api_key", type=str, default="key", help="API key for the VLLM server.")
    parser.add_argument("--num_servers", type=int, default=1, help="Number of VLLM servers running")
    parser.add_argument("--served_model_name", type=str, default="phi-4", help="Name of the served model.")
    parser.add_argument("--server_heartbeat", type=int, default=10, help="Heartbeat interval for the VLLM server.")
    parser.add_argument(
        "--max_wait_time_for_vllm_heartbeat",
        type=int,
        default=3600,
        help="Max wait time for VLLM server heartbeat.",
    )
    return parser


def eval_args(parser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--phigen_dataset", type=str, default="local", choices=["local", "temp"], help="Type of dataset to use."
    )
    parser.add_argument("-i", "--prompts_file", type=str, help="Path to the prompts jsonl.")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="Field within the prompts jsonl to use as prompt. (default: prompt)",
    )
    parser.add_argument("-o", "--output_dir", type=str, help="Path to save results.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer.")
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
    parser.add_argument(
        "--max_tool_call_steps",
        type=int,
        default=5,
        help="Maximum number of tool calls per round.",
    )
    parser.add_argument(
        "--tool_call_timeouts",
        type=str,
        default="""{"code_interpreter": {"python": 60}}""",
        help="Timeouts per tool in seconds.",
    )
    parser.add_argument(
        "--generate_multi_each_step",
        action="store_true",
        help="Generate multiple responses for each step based on `num_samples_to_generate`.",
    )
    parser.add_argument("--max_model_seq_len", type=int, default=32768, help="Maximum sequence length for the model.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to generate.")
    parser.add_argument("--system_message", type=str, default=None, help="System message to add to the prompt.")
    parser.add_argument("--thinking_model", action="store_true", help="Add thinking generation prompt.")
    parser.add_argument("--num_worker_procs", type=int, default=512, help="Number of worker processes to use.")
    parser.add_argument("--num_samples_to_generate", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--generation_config", type=str, default="{}", help="JSON of generation configuration.")
    return parser


class VLLMLocalResponseAgent(AssistantAgent):
    _agent_name: str = "VLLMLocalResponseAgent"
    next_user_messages: List[List[Dict | str]] = phield(default_factory=list)
    response: Dict[str, Any]
    steps: int = 0
    turns: int = 0

    def __init__(self, *args, next_user_messages=[], steps=0, turns=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_user_messages = next_user_messages
        self.steps = steps
        self.turns = turns

    async def run(self):
        ignore_client_kwargs = [
            "model_utils_name",
            "tokenizer",
            "api_type",
            "max_tokens",
            "max_model_seq_len",
            "thinking_model",
            "generate_multi_each_step",
        ]
        model = self.client_kwargs.get("model")
        max_tokens = self.client_kwargs.get("max_tokens", None)
        max_model_seq_len = self.client_kwargs.get("max_model_seq_len", None)
        api_type = self.client_kwargs.get("api_type")
        tokenizer = self.client_kwargs.get("tokenizer", None)

        _setup_vllm_local_client(
            str(model),
            getattr(ModelKind, api_type.upper()),
            ModelMetadata(
                user_token="user",
                asst_token="assistant",
                organization="openai",
            ),
        )

        client_kwargs = {}
        if api_type == "completion" and max_tokens is None:
            assert tokenizer is not None, "Tokenizer must be provided for completion API when max_tokens is not set"
            assert (
                max_model_seq_len is not None
            ), "Max model sequence length must be set when max_tokens is not provided"
            client_kwargs["max_tokens"] = max(
                max_model_seq_len - len(tokenizer.encode(self.messages, add_special_tokens=True)), 1
            )
        elif max_tokens is not None:
            client_kwargs["max_tokens"] = max_tokens
        client_kwargs.update({k: v for k, v in self.client_kwargs.items() if k not in ignore_client_kwargs})

        try:
            tries = 0
            max_retires = int(os.environ.get("PHYAGI_API_MAX_RETRY", "2"))
            while tries < max_retires:
                try:
                    response = await asyncio.wait_for(
                        complete_text_full_async(self.messages, **client_kwargs),
                        timeout=int(os.environ.get("PHYAGI_API_TIMEOUT", "1800")) + 10,
                    )
                    break
                except asyncio.TimeoutError:
                    tries += 1
                    if tries >= max_retires:
                        raise
                    logger().warning(f"VLLM API TimeoutError encountered for instance {self.idx}. Retrying {tries}/{max_retires}...")
        except Exception as e:
            self.response = {
                "idx": [self.idx],
                "completions": [""],
                "finish_reasons": [str(e)],
                "stop_reasons": [str(e)],
                "steps": [self.steps],
                "turns": [self.turns],
                "do_filter": [False],
            }
            self.submit_next_user_messages()
            return

        assert response.all_contents is not None, "Response contents are None"
        self.response = {
            "idx": [self.idx for _ in response.all_contents],
            "completions": response.all_contents,
            "finish_reasons": response.all_finish_reasons,
            "stop_reasons": response.all_stop_reasons,
            "steps": [self.steps for _ in response.all_contents],
            "turns": [self.turns for _ in response.all_contents],
            "do_filter": [False for _ in response.all_contents],
        }
        self.submit_next_user_messages()

    def submit_next_user_messages(self):
        """Submit next user messages to the agent."""
        if len(self.next_user_messages) > 0:
            api_type = self.client_kwargs.get("api_type")
            tokenizer = self.client_kwargs.get("tokenizer", None)
            thinking_model = self.client_kwargs.get("thinking_model", False)
            generate_multi_each_step = self.client_kwargs.get("generate_multi_each_step", False)
            model_utils_name = self.client_kwargs.get("model_utils_name", "phi-think")
            mutils = ModelUtilsFactory.get_utils(model_utils_name)

            next_user_message = self.next_user_messages[0]
            next_inputs = []
            for i, completion in enumerate(self.response["completions"]):
                # If we are already filtering this response, skip generating next turn
                if self.response["do_filter"][i]:
                    continue

                next_input = {}
                next_input["idx"] = self.idx
                if api_type == "completion":
                    next_input["messages"] = self.messages + completion + tokenizer.decode(tokenizer.eos_token_id)
                    if isinstance(next_user_message, str):
                        next_user_message = [{"role": "user", "content": next_user_message}]
                    next_user_message = mutils.apply_chat_template(
                        next_user_message, add_generation_prompt=True, thinking=thinking_model
                    )
                    next_input["messages"] += mutils.get_message_join_token() + next_user_message
                else:
                    next_input["messages"] = deepcopy(self.messages)
                    next_input["messages"].append({"role": "assistant", "content": completion})
                    if isinstance(next_user_message, str):
                        next_user_message = [{"role": "user", "content": next_user_message}]
                    next_input["messages"].extend(next_user_message)
                next_input["next_user_messages"] = self.next_user_messages[1:]
                next_inputs.append(next_input)

                # # Uncomment to only keep the final response with all turn completions and filter out the rest
                # self.response["do_filter"][i] = True

            next_client_kwargs = {k: v for k, v in self.client_kwargs.items()}
            if not generate_multi_each_step:
                next_client_kwargs["n"] = 1

            self.add_agent_jobs(
                inputdata=next_inputs,
                client_kwargs=next_client_kwargs,
                preprocess_kwargs={"steps": 0, "turns": self.turns + 1},
            )

    @classmethod
    def preprocess_wrapper(cls, inputdata, user_prompt_path=None, client_kwargs={}, steps=0, turns=0, **kwargs):
        def wrap():
            for d in inputdata:
                yield {
                    "user_prompt_path": user_prompt_path,
                    "client_kwargs": client_kwargs,
                    "steps": steps,
                    "turns": turns,
                    **d,
                }

        return wrap()

    def get_training_data(self) -> Dict[str, Any]:
        return self.response


class VLLMLocalToolResponseAgent(VLLMLocalResponseAgent):
    _agent_name: str = "VLLMLocalToolResponseAgent"
    session_id: str | None = None
    involved_classes: list[str] | None
    initial_config: dict | None

    def __init__(self, *args, session_id=None, involved_classes=None, initial_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id
        self.involved_classes = involved_classes
        self.initial_config = initial_config

    async def run(self):
        ignore_client_kwargs = [
            "model_utils_name",
            "tokenizer",
            "api_type",
            "max_tokens",
            "max_model_seq_len",
            "thinking_model",
            "max_tool_call_steps",
            "generate_multi_each_step",
            "tool_call_timeouts",
        ]
        model = self.client_kwargs.get("model")
        max_tokens = self.client_kwargs.get("max_tokens", None)
        max_model_seq_len = self.client_kwargs.get("max_model_seq_len", None)
        api_type = self.client_kwargs.get("api_type")
        tokenizer = self.client_kwargs.get("tokenizer", None)
        model_utils_name = self.client_kwargs.get("model_utils_name", "phi-think")
        max_tool_call_steps = self.client_kwargs.get("max_tool_call_steps", 5)
        generate_multi_each_step = self.client_kwargs.get("generate_multi_each_step", False)
        tool_call_timeouts = self.client_kwargs.get("tool_call_timeouts", {})

        assert api_type == "completion", "VLLMLocalToolResponseAgent only supports completion API type"
        mutils = ModelUtilsFactory.get_utils(model_utils_name)

        _setup_vllm_local_client(
            str(model),
            getattr(ModelKind, api_type.upper()),
            ModelMetadata(
                user_token="user",
                asst_token="assistant",
                organization="openai",
            ),
        )

        client_kwargs = {}
        if max_tokens is None:
            assert tokenizer is not None, "Tokenizer must be provided for completion API when max_tokens is not set"
            assert (
                max_model_seq_len is not None
            ), "Max model sequence length must be set when max_tokens is not provided"
            client_kwargs["max_tokens"] = max(
                max_model_seq_len - len(tokenizer.encode(self.messages, add_special_tokens=True)), 1
            )
        else:
            client_kwargs["max_tokens"] = max_tokens
        client_kwargs.update({k: v for k, v in self.client_kwargs.items() if k not in ignore_client_kwargs})

        full_completion = str(self.messages).split(mutils.get_generation_prompt(thinking=False), 1)[1]
        try:
            tries = 0
            max_retires = int(os.environ.get("PHYAGI_API_MAX_RETRY", "3"))
            while tries < max_retires:
                try:
                    response = await asyncio.wait_for(
                        complete_text_full_async(self.messages, **client_kwargs),
                        timeout=int(os.environ.get("PHYAGI_API_TIMEOUT", "2700")),
                    )
                    break
                except asyncio.TimeoutError:
                    tries += 1
                    if tries >= max_retires:
                        logger().warning(f"Max retries reached for instance {self.idx}. Raising TimeoutError.")
                        raise
                    logger().warning(f"VLLM API TimeoutError encountered for instance {self.idx}. Retrying {tries}/{max_retires}...")
        except Exception as e:
            self.response = {
                "idx": [self.idx],
                "completions": [full_completion],
                "finish_reasons": [str(e)],
                "stop_reasons": [str(e)],
                "agent_finish_reason": ["CompletionErrorStop"],
                "steps": [self.steps],
                "turns": [self.turns],
                "do_filter": [False],
            }
            self.submit_next_user_messages()
            return

        assert (
            response.all_contents is not None and response.all_finish_reasons and response.all_stop_reasons is not None
        ), "Response contents, finish reasons, or stop reasons are None"

        self.response = defaultdict(list)
        model_response = (response.all_contents, response.all_finish_reasons, response.all_stop_reasons)
        think_end_token = mutils.get_think_tokens()[1]
        tool_call_end_token = mutils.get_tool_call_tokens()[1]

        for content, finish_reason, stop_reason in zip(*model_response):
            self.response["idx"].append(self.idx)
            self.response["finish_reasons"].append(finish_reason)
            self.response["stop_reasons"].append(stop_reason)
            self.response["steps"].append(self.steps)
            self.response["turns"].append(self.turns)
            completion, agent_finish_reason, do_filter = None, None, False
            cur_full_completion = full_completion + content
            if finish_reason == "stop":
                next_client_kwargs = {k: v for k, v in self.client_kwargs.items()}
                if not generate_multi_each_step:
                    next_client_kwargs["n"] = 1

                if stop_reason == think_end_token:
                    completion = content + think_end_token
                    do_filter = True

                    cur_full_generation = self.messages + completion
                    next_client_kwargs["stop"] = [tool_call_end_token] + next_client_kwargs["stop"][1:]
                    next_input = {
                        "idx": self.idx,
                        "messages": cur_full_generation,
                        "next_user_messages": self.next_user_messages,
                        "session_id": self.session_id,
                        "involved_classes": self.involved_classes,
                        "initial_config": self.initial_config,
                    }

                    self.add_agent_jobs(
                        inputdata=[next_input],
                        client_kwargs=next_client_kwargs,
                        preprocess_kwargs={"steps": self.steps, "turns": self.turns},
                    )
                elif stop_reason == tool_call_end_token:
                    if self.steps < max_tool_call_steps:
                        completion = content + tool_call_end_token
                        do_filter = True

                        tool_call = mutils.parse_tool_call(completion)
                        tool_response = await ToolExecutor.execute_tool_call_async(
                            tool_call,
                            session_id=self.session_id,
                            initial_config=self.initial_config,
                            involved_classes=self.involved_classes,
                            tool_call_timeouts=tool_call_timeouts,
                        )
                        cur_full_generation = self.messages + completion
                        next_client_kwargs["stop"] = [think_end_token] + next_client_kwargs["stop"][1:]
                        next_prompt = mutils.add_tool_response(cur_full_generation, tool_response)
                        next_input = {
                            "idx": self.idx,
                            "messages": next_prompt,
                            "next_user_messages": self.next_user_messages,
                            "session_id": self.session_id,
                            "involved_classes": self.involved_classes,
                            "initial_config": self.initial_config,
                        }

                        self.add_agent_jobs(
                            inputdata=[next_input],
                            client_kwargs=next_client_kwargs,
                            preprocess_kwargs={"steps": self.steps + 1, "turns": self.turns},
                        )
                    elif self.steps >= max_tool_call_steps:
                        completion = cur_full_completion + tool_call_end_token
                        agent_finish_reason = "MaxToolCallStepsStop"
                else:
                    completion = cur_full_completion
                    agent_finish_reason = "CompletedStop"
            else:
                completion = full_completion
                agent_finish_reason = "IncompleteStop"
            self.response["completions"].append(completion)
            self.response["agent_finish_reason"].append(agent_finish_reason)
            self.response["do_filter"].append(do_filter)
        self.submit_next_user_messages()

    def submit_next_user_messages(self):
        """Submit next user messages to the agent."""
        if len(self.next_user_messages) > 0:
            tokenizer = self.client_kwargs.get("tokenizer", None)
            thinking_model = self.client_kwargs.get("thinking_model", False)
            generate_multi_each_step = self.client_kwargs.get("generate_multi_each_step", False)
            model_utils_name = self.client_kwargs.get("model_utils_name", "phi-think")
            mutils = ModelUtilsFactory.get_utils(model_utils_name)

            next_user_message = self.next_user_messages[0]
            next_inputs = []
            for i, completion in enumerate(self.response["completions"]):
                # If we are already filtering this response, skip generating next turn
                if self.response["do_filter"][i]:
                    continue

                next_input = {}
                next_input["idx"] = self.idx
                prompt = str(self.messages).split(mutils.get_generation_prompt(thinking=False), 1)[0]
                prompt += mutils.get_generation_prompt(thinking=False)
                next_input["messages"] = prompt + completion + tokenizer.decode(tokenizer.eos_token_id)
                if isinstance(next_user_message, str):
                    next_user_message = [{"role": "user", "content": next_user_message}]
                next_user_message = mutils.apply_chat_template(
                    next_user_message, add_generation_prompt=True, thinking=thinking_model
                )
                next_input["messages"] += mutils.get_message_join_token() + next_user_message
                next_input["next_user_messages"] = self.next_user_messages[1:]
                next_input["session_id"] = self.session_id
                next_input["involved_classes"] = self.involved_classes
                next_input["initial_config"] = self.initial_config
                next_inputs.append(next_input)

                # # Uncomment to only keep the final response with all turn completions and filter out the rest
                # self.response["do_filter"][i] = True

            next_client_kwargs = {k: v for k, v in self.client_kwargs.items()}
            if not generate_multi_each_step:
                next_client_kwargs["n"] = 1

            self.add_agent_jobs(
                inputdata=next_inputs,
                client_kwargs=next_client_kwargs,
                preprocess_kwargs={"steps": 0, "turns": self.turns + 1},
            )


def add_sm_and_format(prompt, sm):
    """Adds system message to the prompt and converts it into messages format."""
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    if sm:
        first_msg = prompt[0]
        if first_msg["role"] != "system":
            prompt = [{"role": "system", "content": sm}] + prompt
        elif first_msg["content"] != sm:
            prompt[0] = {"role": "system", "content": sm}
    return prompt


class VLLMLocalInferenceFlow(Flow):
    _flow_name: str = "VLLMLocalInferenceFlow"
    args: argparse.Namespace = phield(default_factory=lambda: argparse.Namespace())

    def __init__(self, args: argparse.Namespace, **kwargs):
        self.args = args

    def run(self):
        mutils = ModelUtilsFactory.get_utils(self.args.model_utils_name)
        client_kwargs = {
            "model": self.args.served_model_name,
            "api_type": self.args.api_type,
            "n": self.args.num_samples_to_generate,
            "model_utils_name": self.args.model_utils_name,
            "thinking_model": self.args.thinking_model,
            "generate_multi_each_step": self.args.generate_multi_each_step,
        }

        for cfg_key in ["temperature", "top_p", "top_k", "skip_special_tokens"]:
            if cfg_key in self.args.generation_config:
                client_kwargs[cfg_key] = self.args.generation_config[cfg_key]

        if self.args.api_type == "completion":
            if self.args.max_tokens is not None:
                client_kwargs["max_tokens"] = int(self.args.max_tokens)
            else:
                client_kwargs["max_model_seq_len"] = int(self.args.max_model_seq_len)
                assert self.args.tokenizer_path is not None, "Tokenizer path must be provided for completion API type"
                client_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

            if self.args.agent_cls.__name__ == "VLLMLocalToolResponseAgent":
                if "tokenizer" not in client_kwargs:
                    assert (
                        self.args.tokenizer_path is not None
                    ), "Tokenizer path must be provided for tool response agent"
                    client_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

                client_kwargs["stop"] = [
                    mutils.get_think_tokens()[1],
                    client_kwargs["tokenizer"].decode(client_kwargs["tokenizer"].eos_token_id),
                ]
                client_kwargs["max_tool_call_steps"] = self.args.max_tool_call_steps
                client_kwargs["tool_call_timeouts"] = json.loads(self.args.tool_call_timeouts)

        data = load_file_shared(self.args.prompts_file, show_progress=False)
        self.args.agent_cls.add_agent_jobs(
            inputdata=self.get_data(self.args, data, mutils),
            client_kwargs=client_kwargs,
            preprocess_kwargs={"steps": 0, "turns": 0},
        )

    @classmethod
    def get_data(cls, args, data, mutils):
        system_message = args.system_message
        prompt_field = args.prompt_field

        for idx, item in enumerate(data):
            prompt = item.get(prompt_field, None)
            assert prompt is not None, f"Prompt field '{prompt_field}' not found in item {idx}"

            next_user_messages = []
            if isinstance(prompt, list) and isinstance(prompt[0], list):
                next_user_messages = prompt[1:]
                prompt = prompt[0]

            if system_message:
                assert isinstance(system_message, str), "System message must be a string"
                if system_message in DEFAULT_SYSTEM_PROMPTS:
                    sm = DEFAULT_SYSTEM_PROMPTS[system_message]
                    if system_message == "random":
                        sm = random.choice(sm)
                else:
                    sm = system_message

                if system_message in ["re_tool_qwen_template_sys"]:
                    # If system_message is a tool template, replace placeholders with actual values
                    _tool_details = "{tool_details}"
                    _response_format = "{response_format}"
                    if "involved_classes" in item:
                        involved_classes = item["involved_classes"]
                        tool_details = (
                            ToolExecutor.construct_tools_from_involved_classes(involved_classes)
                            + f"\n\n[Classes Involved: {involved_classes}]"
                        )
                        response_format = bfcl_response_format
                    else:
                        tool_details = ci_tool_details
                        response_format = ci_response_format
                    sm = sm.replace(_tool_details, tool_details).replace(_response_format, response_format)
                elif system_message == "auto":
                    sm = item.get("system_message", sm)

                prompt = add_sm_and_format(prompt, sm)
            else:
                prompt = add_sm_and_format(prompt, None)

            input_item = {"idx": str(idx), "messages": prompt, "next_user_messages": next_user_messages}
            if args.api_type == "completion":
                input_item["messages"] = mutils.apply_chat_template(
                    input_item["messages"], add_generation_prompt=True, thinking=args.thinking_model
                )
                if args.agent_cls.__name__ == "VLLMLocalToolResponseAgent":
                    input_item["involved_classes"] = item.get("involved_classes", [])
                    input_item["session_id"] = item.get("session_id", str(uuid.uuid4()))
                    input_item["initial_config"] = item.get("initial_config", {})
            yield input_item

    @classmethod
    def write(cls, args=None, remove_cols=[], **kwargs):
        assert args is not None, "Arguments must be provided"
        mutils = ModelUtilsFactory.get_utils(args.model_utils_name)

        def get_flow_output(input_data):
            for idx, inp in enumerate(input_data):
                prompt = cls.get_data(args, [inp], mutils).__next__()["messages"]
                if isinstance(prompt, str):
                    inp["generation_prompt"] = prompt
                else:
                    inp["generation_prompt"] = json.dumps(prompt, ensure_ascii=False)
                for out in (item.get_training_data() for item in args.agent_cls.load_successful(idx=str(idx))):
                    for c in remove_cols:
                        out.pop(c, None)
                    filtered_out = {
                        key: [v for v, do_filter in zip(values, out["do_filter"]) if not do_filter]
                        for key, values in out.items()
                        if key != "do_filter"
                    }
                    for k, v in filtered_out.items():
                        inp.setdefault(k, []).extend(v)
                yield inp

        data = load_file_shared(args.prompts_file, show_progress=False)
        filename, _ = os.path.splitext(os.path.basename(args.prompts_file))
        cls.write_all_data(
            generations=get_flow_output(data),
            writer_name_or_cls="jsonl",
            path=args.output_dir,
            name=f"{filename}_output",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Options")
    parser = vllm_args(parser)
    parser = eval_args(parser)
    args = parser.parse_args()

    args.generation_config = json.loads(args.generation_config)
    args.agent_cls = globals()[args.agent_cls]
    args.num_servers = 1  # Pin to one shared server port as we are using vllm internal ddp

    logger().info(f"Running VLLM inference with args: {args}")
    os.environ["VLLM_LOCAL_BASE_URL"] = "http://{base_ip}:{{port}}/v1".format(base_ip=args.base_ip)
    os.environ["VLLM_LOCAL_BASE_PORT"] = str(args.base_port)
    os.environ["VLLM_LOCAL_API_KEY"] = str(args.api_key)
    os.environ["VLLM_LOCAL_NUM_INSTANCES"] = str(args.num_servers)
    os.environ["VLLM_LOCAL_SERVER_HEARTBEAT"] = str(args.server_heartbeat)
    os.environ["VLLM_LOCAL_MAX_WAIT_TIME_FOR_HEARTBEAT"] = str(args.max_wait_time_for_vllm_heartbeat)

    _setup_vllm_local_client(
        str(args.served_model_name),
        getattr(ModelKind, args.api_type.upper()),
        ModelMetadata(
            user_token="user",
            asst_token="assistant",
            organization="openai",
        ),
    )

    filename, _ = os.path.splitext(os.path.basename(args.prompts_file))
    if args.phigen_dataset == "local":
        dataset = local_dataset(f"{filename}_output_dataset", dir=args.output_dir)
    elif args.phigen_dataset == "temp":
        dataset = temp_dataset()
    else:
        raise ValueError(f"Invalid phigen dataset type: {args.phigen_dataset}")

    VLLMLocalInferenceFlow.run_flow(
        dataset=dataset,
        data_kwargs={"args": args, "remove_cols": ["idx"]},
        run_kwargs={"multiprocess": True, "ordered": False, "num_worker_procs": args.num_worker_procs},
    )
