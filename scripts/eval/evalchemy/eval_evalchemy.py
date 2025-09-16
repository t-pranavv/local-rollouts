# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import shlex
import subprocess
import sys
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates a model with vLLM and evalchemy.")

    parser.add_argument("pretrained_model_name_or_path", type=str, help="Pre-trained model name or path.")

    parser.add_argument("-t", "--tasks", type=str, default="AIME24", help="Evaluation tasks (e.g., AIME24).")

    parser.add_argument(
        "-w", "--wait", type=int, default=100, help="Time (in seconds) to wait for the vLLM server to start."
    )

    parser.add_argument("-op", "--output_path", type=str, default="logs", help="Path to store evaluation logs.")

    parser.add_argument("-n", "--num_concurrent", type=int, default=10, help="Number of concurrent evaluations to run.")

    parser.add_argument("-ml", "--max_length", type=int, default=16384, help="Maximum model length for vLLM.")

    parser.add_argument(
        "-mgt", "--max_gen_toks", type=int, default=15384, help="Maximum number of generated tokens for evaluation."
    )

    parser.add_argument(
        "-tps", "--tensor_parallel_size", type=int, default=2, help="Tensor parallel size for the vLLM server."
    )

    parser.add_argument("-ne", "--num_evals", type=int, default=1, help="Number of evaluations to run.")

    parser.add_argument(
        "-wd",
        "--work_dir",
        type=str,
        default="/evalchemy",
        help="Working directory where the evaluation command will be executed.",
    )

    parser.add_argument(
        "-oai", "--openai_api_key", type=str, default="YOUR_API_KEY_HERE", help="OpenAI API key for authentication."
    )

    parser.add_argument(
        "-spt",
        "--system_prompt_type",
        type=str,
        required=True,
        help="Type of system prompt to use ('cot' or 'simple').",
    )

    parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        required=True,
        help="Temperature for model generation (controls randomness).",
    )

    parser.add_argument(
        "-p", "--top_p", type=float, required=True, help="Top-p for model generation (controls diversity)."
    )

    parser.add_argument("-s", "--vllm_seed", type=int, default=42, help="vLLM seed for generation.")

    return parser.parse_args()


def _run_vllm_server(args: argparse.Namespace, eval_index: int) -> subprocess.Popen:
    vllm_command = (
        f"vllm serve {args.pretrained_model_name_or_path} "
        f"--tensor-parallel-size {args.tensor_parallel_size} "
        f"--gpu_memory_utilization 0.8 "
        f"--enable-chunked-prefill "
        f"--max-model-len {args.max_length} "
        f"--port 8000 "
        f"--seed {args.vllm_seed + eval_index} "
        f"--trust-remote-code"
    )

    print(f"Starting vLLM server with command: {vllm_command}")
    return subprocess.Popen(shlex.split(vllm_command))


def _run_evaluation(args: argparse.Namespace, eval_command: str, eval_env: dict, eval_index: int) -> None:
    print(f"\nRunning evaluation {eval_index + 1} of {args.num_evals} with command: {eval_command}")

    eval_result = subprocess.run(shlex.split(eval_command), cwd=args.work_dir, env=eval_env)
    if eval_result.returncode != 0:
        print(f"Evaluation {eval_index + 1} failed with an error.", file=sys.stderr)
    else:
        print(f"Evaluation {eval_index + 1} completed successfully.")


if __name__ == "__main__":
    args = _parse_args()

    eval_command = (
        f"python -m eval.eval "
        "--model local-completions "
        f"--tasks {args.tasks} "
        f"--model_args model={args.pretrained_model_name_or_path},"
        "base_url=http://0.0.0.0:8000/v1/completions,"
        f"num_concurrent={args.num_concurrent},"
        "max_retries=3,"
        "tokenized_requests=True,"
        f"max_length={args.max_length},"
        f"max_gen_toks={args.max_gen_toks} "
        "--batch_size 1 "
        f"--output_path {args.output_path} "
        f"--temperature {args.temperature} "
        f"--top_p {args.top_p} "
        f"--system_prompt_type {args.system_prompt_type}"
    )

    eval_env = os.environ.copy()
    eval_env["OPENAI_API_KEY"] = args.openai_api_key

    for eval_index in range(args.num_evals):
        server_proc = _run_vllm_server(args, eval_index)
        try:
            print(f"\nWaiting {args.wait} seconds for the vLLM server to initialize...")
            time.sleep(args.wait)
            _run_evaluation(args, eval_command, eval_env, eval_index)

        finally:
            print("\nShutting down the vLLM server...")
            server_proc.terminate()

            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server did not shut down gracefully. Forcing termination...")
                server_proc.kill()
