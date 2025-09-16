# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import time
import sys
import json
import argparse
import datetime
import asyncio
import random
import subprocess
import traceback
from tqdm.asyncio import tqdm

from logging import StreamHandler, getLogger
from openai import AsyncOpenAI

import torch
import torch.distributed as dist

# Logger
logger = getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(StreamHandler(sys.stdout))

# Constants
API_KEY = "key"
SERVER_HEARTBEAT = 60


def init_distributed_backend(backend="gloo"):
    # Initialize the distributed environment
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=2.0))
    # Get the rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def heartbeat(api: str) -> subprocess.CompletedProcess:
    return subprocess.run(f"curl -s {api}/ping", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


async def vllm_server_heartbeat(base_port: int, rank: int) -> None:
    first_heartbeat = True
    while True:
        response = heartbeat(f"http://localhost:{base_port + rank}")
        if response.returncode == 0:
            logger.debug(f"Rank [{rank}] [VLLM]: Server is up and running ...")
            first_heartbeat = False
        elif first_heartbeat:
            logger.debug(
                f"Rank [{rank}] [VLLM]: Server starting...Waiting another {SERVER_HEARTBEAT}s for server to start ..."
            )
        else:
            logger.error(f"Rank [{rank}] [VLLM]: Rank [{rank}]: Server has crashed. Shutting down ...")
            break
        await asyncio.sleep(SERVER_HEARTBEAT)


def vllm_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("--base_port", type=int, default=8000, help="Base port number.")
    parser.add_argument("--served_model_name", type=str, default="phi-4", help="Name of the served model.")
    parser.add_argument(
        "--max_wait_time_for_vllm_heartbeat", type=int, default=3600, help="Max wait time for VLLM server heartbeat."
    )
    return parser


def eval_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("-i", "--prompts_file", type=str, help="Path to the prompts jsonl.")
    parser.add_argument(
        "--prompt_field", type=str, default="prompt", help="Field within the prompts jsonl to use as prompt."
    )
    parser.add_argument("-o", "--output_dir", type=str, help="Path to save results.")
    parser.add_argument("--add_system_message", action="store_true", help="Add system message to the prompt.")
    parser.add_argument("--system_message", type=str, default="cot", help="System message to add to the prompt.")
    parser.add_argument("--max_parallel_requests_per_gpu", type=int, default=1, help="Maximum parallel requests.")
    parser.add_argument("--num_samples_to_generate", type=int, default=1, help="Number of samples to generate.")
    return parser


async def get_model_response(
    input_data_queue: asyncio.Queue,
    output_data_queue: asyncio.Queue,
    model_client: AsyncOpenAI,
    served_model_name: str,
    progress: tqdm,
    num_samples_to_generate: int,
) -> None:
    while True:
        try:
            input_data = await input_data_queue.get()
            if input_data is None:
                input_data_queue.task_done()
                break
            idx, prompt = input_data
            completions = await model_client.chat.completions.create(
                model=served_model_name, messages=prompt, n=num_samples_to_generate, timeout=9999999
            )
            logger.debug(f"IDX [{idx}] [Eval]: \n {completions}\n ================================================")
            finish_reasons = [completions.choices[i].finish_reason for i in range(num_samples_to_generate)]
            completions = [completions.choices[i].message.content for i in range(num_samples_to_generate)]
            await output_data_queue.put((idx, completions, finish_reasons))
            input_data_queue.task_done()
            progress.update(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            break
    return


async def write_output(
    rank: int, output_data_queue: asyncio.Queue, all_data: list, output_dir: str, filename: str
) -> None:
    with open(os.path.join(output_dir, f"results_rank{rank}_{filename}"), "w", encoding="utf-8") as f:
        logger.info(f"Rank [{rank}] [Eval]: Writing results to {output_dir}/results_rank{rank}_{filename}")
        processed = 0
        while True:
            output_data = await output_data_queue.get()
            idx, completions, finish_reasons = output_data
            data = all_data[idx]
            data["index"] = idx
            data["completions"] = completions
            data["finish_reasons"] = finish_reasons
            f.write(json.dumps(data) + "\n")
            f.flush()
            output_data_queue.task_done()
            processed += 1
            if processed == len(all_data):
                break
    return


def add_system_prompt(message, system_message, add_system_message):
    if add_system_message:
        message = [{"role": "system", "content": system_message}] + message
    return message


def pack_message(system_message, prompt, add_system_message):
    message = []
    message = add_system_prompt(message, system_message, add_system_message)
    message += [{"role": "user", "content": f"{prompt}"}]
    return message


DEFAULT_SYSTEM_PROMPTS = {
    "cot": "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
    "cot2": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
    "cot_final": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
    "cot_final_rl": "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:",
    "boring": "You are a helpful AI agent.",
    "random": [
        "You are an AI assistant that helps people find information.",
        "You're Phi, a large language model trained by Microsoft to help users",
        "You are a kind and helpful assistant. Respond only with helpful, ethical responses, and avoid harmful or inappropriate content.",
        "You are a kind, smart, capable, and helpful assistant. Give answers that aid the user, while being extremely technically adept",
        "you are a good assistant do your best to answer questions",
        "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
        "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
        "You follow user instruction extremely well",
        "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    ],
}


async def run_eval(
    base_port: int,
    rank: int,
    world_size: int,
    max_wait_time_for_vllm_heartbeat: int,
    prompts_file: str,
    prompt_field: str,
    output_dir: str,
    add_system_message: bool,
    system_message: str,
    max_parallel_requests: int,
    served_model_name: str,
    num_samples_to_generate: int,
) -> None:
    max_wait_time_for_vllm_heartbeat = float(max_wait_time_for_vllm_heartbeat)
    check_starttime = time.time()
    while True:
        response = heartbeat(f"http://localhost:{base_port + rank}")
        if response.returncode == 0:
            logger.info(f"Rank [{rank}] [Eval]: Server is up and running ...")
            break
        else:
            logger.info(f"Rank [{rank}] [Eval]: Server not up yet. Waiting 10s ...")
        await asyncio.sleep(10)
        if time.time() - check_starttime > max_wait_time_for_vllm_heartbeat:
            raise RuntimeError("Server max wait time exceeded. Shutting down ...")

    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    input_data_queue = asyncio.Queue()
    with open(prompts_file, "r", encoding="utf-8") as f:
        index = 0
        for idx, line in enumerate(f.readlines()):
            if idx % world_size != rank:
                continue
            data = json.loads(line)
            all_data.append(data)

            sm = None
            if system_message.startswith("cot"):
                sm = DEFAULT_SYSTEM_PROMPTS[system_message]
            elif system_message == "auto" or system_message == "boring":
                sm = data.get("system_message", DEFAULT_SYSTEM_PROMPTS["boring"])
            elif system_message == "random":
                sm = random.choice(DEFAULT_SYSTEM_PROMPTS["random"])
            elif system_message:
                sm = system_message

            if "messages" in data:
                prompt = data["messages"]
                prompt = add_system_prompt(prompt, sm, add_system_message)
            else:
                prompt = data[prompt_field]
                prompt = pack_message(sm, prompt, add_system_message)
            if idx < 9:  # print one prompt from each rank
                logger.info(f"Rank [{rank}] [Eval]: {'-' * 5} Prompt: {prompt}")
            else:
                logger.debug(f"Rank [{rank}] [Eval]: {'-' * 5} Prompt: {prompt}")
            await input_data_queue.put((index, prompt))
            index += 1

    logger.info(f"Rank [{rank}] [Eval]: Found {len(all_data)} total prompts")

    progress = tqdm(total=len(all_data), desc=f"Rank [{rank}] [Eval]: Processing prompts")
    model_client = AsyncOpenAI(
        base_url=f"http://localhost:{base_port + rank}/v1",
        api_key=API_KEY,
    )

    output_data_queue = asyncio.Queue()
    workers = []
    for _ in range(max_parallel_requests):
        await input_data_queue.put(None)
        workers.append(
            asyncio.create_task(
                get_model_response(
                    input_data_queue=input_data_queue,
                    output_data_queue=output_data_queue,
                    model_client=model_client,
                    num_samples_to_generate=num_samples_to_generate,
                    served_model_name=served_model_name,
                    progress=progress,
                )
            )
        )

    writer = asyncio.create_task(
        write_output(rank, output_data_queue, all_data, output_dir, os.path.basename(prompts_file))
    )

    await input_data_queue.join()
    all_workers_done = False
    while not all_workers_done:
        for worker in workers:
            if worker.done():
                worker.cancel()
                workers.remove(worker)
                all_workers_done = True
            else:
                all_workers_done = False
                break
        await asyncio.sleep(1)

    while not writer.done():
        await asyncio.sleep(1)
    logger.info(f"Rank [{rank}] [Eval]: Completed processing prompts")
    writer.cancel()
    return


async def main():
    parser = argparse.ArgumentParser(description="Inference Options")
    parser = vllm_args(parser)
    parser = eval_args(parser)
    args = parser.parse_args()

    base_port = args.base_port
    rank, world_size = init_distributed_backend()
    logger.info(f"Rank [{rank}]: World Size={world_size}, Base Port={base_port + rank}")

    server_heartbeat = asyncio.create_task(vllm_server_heartbeat(base_port, rank))
    await run_eval(
        base_port,
        rank,
        world_size,
        args.max_wait_time_for_vllm_heartbeat,
        args.prompts_file,
        args.prompt_field,
        args.output_dir,
        args.add_system_message,
        args.system_message,
        args.max_parallel_requests_per_gpu,
        args.served_model_name,
        args.num_samples_to_generate,
    )
    server_heartbeat.cancel()
    dist.barrier()


if __name__ == "__main__":
    asyncio.run(main())
