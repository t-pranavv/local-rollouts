# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import time
import sys
import json
import argparse
import asyncio
import random
import traceback
from tqdm.asyncio import tqdm

from logging import StreamHandler, getLogger
from phigen import get_default_client
from phigen.model_client import ModelClient
from phigen.model_registry import get_model_from_registry

# Logger
logger = getLogger(__name__)
logger.setLevel("INFO")
logger.addHandler(StreamHandler(sys.stdout))


def eval_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("--model_name", type=str, default="phi-4", help="Name of the gateway model.")
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
    model_client: ModelClient,
    model_name: str,
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

            model = get_model_from_registry(model_name)
            completions = await model_client.text_async(prompt, model=model, n=num_samples_to_generate)

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


async def write_output(output_data_queue: asyncio.Queue, all_data: list, output_dir: str, filename: str) -> None:
    with open(os.path.join(output_dir, f"results_{filename}"), "w", encoding="utf-8") as f:
        logger.info(f"[Eval]: Writing results to {output_dir}/results_{filename}")
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
    prompts_file: str,
    prompt_field: str,
    output_dir: str,
    add_system_message: bool,
    system_message: str,
    max_parallel_requests: int,
    served_model_name: str,
    num_samples_to_generate: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    input_data_queue = asyncio.Queue()
    with open(prompts_file, "r", encoding="utf-8") as f:
        index = 0
        for idx, line in enumerate(f.readlines()):
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
                logger.info(f"[Eval]: {'-' * 5} Prompt: {prompt}")
            else:
                logger.debug(f"[Eval]: {'-' * 5} Prompt: {prompt}")
            await input_data_queue.put((index, prompt))
            index += 1

    logger.info(f"[Eval]: Found {len(all_data)} total prompts")

    progress = tqdm(total=len(all_data), desc=f"[Eval]: Processing prompts")
    model_client = get_default_client()

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

    writer = asyncio.create_task(write_output(output_data_queue, all_data, output_dir, os.path.basename(prompts_file)))

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
    logger.info(f"[Eval]: Completed processing prompts")
    writer.cancel()
    return


async def main():
    parser = argparse.ArgumentParser(description="Inference Options")
    parser = eval_args(parser)
    args = parser.parse_args()

    await run_eval(
        args.prompts_file,
        args.prompt_field,
        args.output_dir,
        args.add_system_message,
        args.system_message,
        args.max_parallel_requests_per_gpu,
        args.model_name,
        args.num_samples_to_generate,
    )


if __name__ == "__main__":
    asyncio.run(main())
