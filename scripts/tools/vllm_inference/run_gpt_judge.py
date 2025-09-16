from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from model_utils import ModelUtilsFactory
from tools import ToolExecutor

from phigen.utils.prompt_render import JinjaPromptTemplate
from phigen.client.model_client import complete_text_full, complete_json_full
from phigen.datagen.record import phield
from phigen.datagen.agent import PromptAgent, Flow
from phigen.datagen.shared_data import load_file_shared
from phigen.datagen.local_dataset import local_dataset
from phigen.logging import logger, set_log_level

set_log_level("INFO")

current_directory = Path(__file__).resolve().parent


SUPPORTED_EVALS = ["AIME", "AIME_SIMPLE", "MATH", "GPQA", "HUMANEVAL"]

JUDGE_PROMPT_TEMPLATES = {
    "AIME": str(current_directory / "prompts/compare_answers_math.jinja"),
    "MATH": str(current_directory / "prompts/compare_answers_math.jinja"),
    "AIME_SIMPLE": str(current_directory / "prompts/compare_answers_math_simple.jinja"),
    "GPQA": str(current_directory / "prompts/compare_answers_gpqa.jinja"),
    "HUMANEVAL": None,
}
CHECK_EXECUTION_PROMPT_MATH = JinjaPromptTemplate(
    template_path=str(current_directory / "prompts/check_execution_prompt_math.jinja")
)


def eval_args(parser) -> argparse.ArgumentParser:
    parser.add_argument("-i", "--prompts_file", type=str, help="Path to the prompts jsonl.")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="Field within the prompts jsonl to use as prompt.",
    )
    parser.add_argument(
        "--answer_field",
        type=str,
        default="answer",
        help="Field referring to ground_truth answer in the results jsonl file.",
    )
    parser.add_argument(
        "--completions_field",
        type=str,
        default="completions",
        help="Field referring to completions in the results jsonl file.",
    )
    parser.add_argument("--completion_type", type=str, default="response", help="Which type of completion to use.")
    parser.add_argument(
        "--model_utils_name",
        type=str,
        default="phi-think",
        help="Utility class to use for model-specific operations.",
    )
    parser.add_argument(
        "--tool_call_timeouts",
        type=str,
        default="""{"code_interpreter": {"python": 60}}""",
        help="Timeout for tool calls in seconds.",
    )
    parser.add_argument("--n_samples", default=1, type=int, help="number of eval samples.")
    parser.add_argument("--repeat", default=1, type=int, help="number of times to repeat the evaluation.")
    parser.add_argument("--output_dir", type=str, help="Path to save results.")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-impact", help="Name of the judge model to use.")
    parser.add_argument(
        "--eval_type", type=str, default="MATH", help=f"Type of eval, currently supported: {SUPPORTED_EVALS}"
    )
    parser.add_argument("--num_worker_procs", type=int, default=512, help="Number of worker processes to use.")
    parser.add_argument("--generation_config", type=str, default="{}", help="Path to the generation config file.")
    return parser


class JudgeAgent(PromptAgent):
    _agent_name: str = "JudgeAgent"
    response: List[str] | None = None
    judge_result: List[bool] | None
    completion_type: str = "response"

    def __init__(self, idx: str, prompt_path: str, prompt_field: str, data: dict, completion_type: str, **kwargs):
        super().__init__(idx, prompt_path, prompt_field, data, **kwargs)
        self.data = data
        self.completion_type = completion_type

    async def judge_function(self):
        self.judge_result = []

    async def execute_code(self, code: str) -> dict:
        tool_response = await ToolExecutor.execute_tool_call_async(
            [{"tool_category": "code_interpreter", "tool_subcategory": "python", "codeblock": code}],
            tool_call_timeouts=self.client_kwargs.get("tool_call_timeouts", {}),
        )
        return tool_response[0]

    def parse_cot(self, response: str) -> Dict[str, str]:
        cot = ""
        answer = ""
        combined = response

        # find the farthest special end token
        mutils = ModelUtilsFactory.get_utils(self.client_kwargs.get("model_utils_name"))
        end_token = mutils.get_think_tokens()[1]
        if end_token is None:
            return {"response": combined, "cot": "", "combined": combined}

        last_end_token = combined.rfind(end_token)
        # split the combined response at the last end token into cot and answer
        if last_end_token != -1:
            cot = combined[: last_end_token + len(end_token)].strip()
            answer = combined[last_end_token + len(end_token) :].strip()
        else:
            # if no end token found, assume the whole response is the cot
            cot = combined.strip()
            answer = ""

        return {"response": answer, "cot": cot, "combined": combined}

    async def run(self):
        ignore_client_kwargs = ["model_utils_name", "skip_special_tokens", "tool_call_timeouts"]
        client_kwargs = {k: v for k, v in self.client_kwargs.items() if k not in ignore_client_kwargs}
        prompt = self.parse_cot(self.prompt)
        self.prompt = prompt[self.completion_type]
        self.response = complete_text_full(self.prompt, **client_kwargs).all_contents
        await self.judge_function()

    @classmethod
    def preprocess_wrapper(
        cls,
        inputdata,
        prompt_path=None,
        completion_type="response",
        client_kwargs={},
        **kwargs,
    ):
        def wrap():
            for d in inputdata:
                yield {
                    "idx": d.pop("idx"),
                    "prompt_path": prompt_path,
                    "prompt_field": None,
                    "data": d,
                    "client_kwargs": client_kwargs,
                    "completion_type": completion_type,
                }

        return wrap()

    def get_training_data(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "gen_idx": self.data.get("gen_idx"),
            "prompt": self.prompt,
            "completions": self.response,
            "judge_results": self.judge_result,
        }


class MathJudgeAgent(JudgeAgent):
    _agent_name: str = "MathJudgeAgent"
    check_pattern = re.compile(r"check\((?:.*?)\)", re.DOTALL)

    async def judge_function(self):
        self.judge_result = []
        for response in self.response:
            response = response.lower()
            if "unknown" in response:
                self.judge_result.append(None)
                continue

            equivalent = "[yes]" in response and "[no]" not in response
            if "[check]" in response:
                equivalent = None
                matches = self.check_pattern.findall(response)
                if len(matches) > 0:
                    check = matches[0]
                    code = CHECK_EXECUTION_PROMPT_MATH.create({"check": check})
                    ci_response = await self.execute_code(code)
                    if ci_response["success"]:
                        try:
                            equivalent = eval(ci_response["stdout"])
                        except Exception as e:
                            logger().warning(f"Error evaluating check: {check}. Error: {e}")
            self.judge_result.append(equivalent)


class MathSimpleJudgeAgent(JudgeAgent):
    _agent_name: str = "MathSimpleJudgeAgent"

    async def run(self):
        ignore_client_kwargs = ["model_utils_name", "skip_special_tokens", "tool_call_timeouts"]
        client_kwargs = {k: v for k, v in self.client_kwargs.items() if k not in ignore_client_kwargs}
        prompt = self.parse_cot(self.prompt)
        self.prompt = prompt[self.completion_type]
        try:
            response = complete_json_full(self.prompt, **client_kwargs).all_contents
        except Exception as e:
            logger().warning(f"Error in client JSON completion: {e}")
            response = ["{}" for _ in range(self.client_kwargs.get("n", 1))]
        self.response = [json.loads(r) for r in response]
        await self.judge_function()

    async def judge_function(self):
        self.judge_result = []
        for response in self.response:
            gt_solution = self.data["correct_solution"]
            if "extracted_answer" not in response:
                logger().warning(f"Missing extracted_answer in response: {response}")
                self.judge_result.append(None)
                continue
            extracted_answer = response["extracted_answer"]

            if "correct_answer" not in response:
                logger().warning(f"Missing correct_answer in response: {response}")
                self.judge_result.append(None)
                continue
            correct_answer = response["correct_answer"]

            try:
                if correct_answer != type(correct_answer)(gt_solution):
                    logger().warning(
                        f"Correct answer mismatch; {correct_answer} != {gt_solution} in response: {response}"
                    )
                    self.judge_result.append(None)
                    continue
            except Exception as e:
                logger().warning(f"Error in parsing judge response: {json.dumps(response)} Error: {e}")
                self.judge_result.append(None)
                continue

            if "equivalence" not in response:
                logger().warning(f"Missing equivalence in response: {response}")
                self.judge_result.append(None)
                continue

            if "comparison_discussion" not in response:
                logger().warning(f"Missing summary in response: {response}")
                self.judge_result.append(None)
                continue

            if extracted_answer == "unknown":
                self.judge_result.append(None)
                continue

            self.judge_result.append(bool(int(response["equivalence"])))


class GPQAJudgeAgent(JudgeAgent):
    _agent_name: str = "GPQAJudgeAgent"

    async def judge_function(self):
        self.judge_result = []
        for response in self.response:
            response = response.lower()
            if "unknown" in response:
                self.judge_result.append(None)
                continue
            equivalent = "[yes]" in response and "[no]" not in response
            self.judge_result.append(equivalent)


class HumanevalJudgeAgent(JudgeAgent):
    _agent_name: str = "HumanevalJudgeAgent"
    markdown_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)

    async def judge_function(self):
        self.judge_result = []
        for response in self.response:
            # Taken from https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
            def find_code():
                matches = self.markdown_pattern.findall(response)
                extracted_answer = matches[0] if len(matches) >= 1 else response
                extracted_answer = extracted_answer[extracted_answer.find(":\n    ") + 2 :]  # remove signature
                return extracted_answer

            code = find_code()
            snippet = code + self.data["test"] + f"\ncheck({self.data['entry_point']})"
            ci_response = await self.execute_code(snippet)
            self.judge_result.append(ci_response["success"])

    @classmethod
    def preprocess_wrapper(cls, *args, **kwargs):
        def wrap():
            for d in super().preprocess_wrapper(*args, **kwargs):
                d.pop("prompt_field", None)
                yield {"promp_field": "student_solution", **d}

        return wrap()


class JudgeFlow(Flow):
    _flow_name: str = "JudgeFlow"
    args: argparse.Namespace = phield(default_factory=lambda: argparse.Namespace())

    def __init__(self, args: argparse.Namespace, **kwargs):
        self.args = args

    def run(self):
        client_kwargs = {
            "model": self.args.judge_model_name,
            "n": self.args.n_samples,
            "model_utils_name": self.args.model_utils_name,
            "tool_call_timeouts": json.loads(self.args.tool_call_timeouts),
        }
        for cfg_key in ["temperature", "top_p", "top_k", "skip_special_tokens"]:
            if cfg_key in self.args.generation_config:
                client_kwargs[cfg_key] = self.args.generation_config[cfg_key]

        data = load_file_shared(self.args.prompts_file, show_progress=False)
        self.args.agent_cls.add_agent_jobs(
            inputdata=self.get_data(data),
            client_kwargs=client_kwargs,
            preprocess_kwargs={
                "prompt_path": self.args.judge_prompt_template_path,
                "completion_type": self.args.completion_type,
            },
        )

    def get_data(self, data):
        for idx, item in enumerate(data):
            completions = item.get(self.args.completions_field)
            for gen_idx, completion in enumerate(completions):
                input_item = {
                    "idx": str(idx),
                    "gen_idx": str(gen_idx),
                    "question": item.get(self.args.prompt_field),
                    "student_solution": completion,
                    "correct_solution": item.get(self.args.answer_field),
                }
                yield input_item

    @classmethod
    def write(cls, args=None, remove_cols=[], **kwargs):
        assert args is not None, "Arguments must be provided"

        def judge_output_data(input_data):
            for idx, inp in enumerate(input_data):
                output_items = [item.get_training_data() for item in args.agent_cls.load_successful(idx=str(idx))]
                output_items = sorted(output_items, key=lambda x: x["gen_idx"])
                for out in output_items:
                    for c in remove_cols:
                        out.pop(c, None)
                    for k, v in out.items():
                        if k in ["prompt", "completions", "judge_results"]:
                            if k == "prompt":
                                inp["judge_prompt"] = v
                            elif k == "completions":
                                inp.setdefault("judge_responses", []).append(v)
                            else:
                                inp.setdefault("judge_results", []).append(v)
                                scores = [v_i for v_i in v if v_i is not None]
                                if len(scores) > 0:
                                    inp.setdefault("aggregated_score", []).append(sum(scores) / len(scores))
                                else:
                                    inp.setdefault("aggregated_score", []).append(None)
                yield inp

        data = load_file_shared(args.prompts_file, show_progress=False)
        cls.write_all_data(
            generations=judge_output_data(data),
            writer_name_or_cls="jsonl",
            path=args.output_dir,
            name=f"{filename}_{args.eval_type.lower()}_judge",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Options")
    parser = eval_args(parser)
    args = parser.parse_args()

    args.prompt_field = None
    if args.eval_type in ["AIME", "MATH"]:
        args.agent_cls = MathJudgeAgent
    elif args.eval_type == "AIME_SIMPLE":
        args.agent_cls = MathSimpleJudgeAgent
    elif args.eval_type == "GPQA":
        args.agent_cls = GPQAJudgeAgent
    elif args.eval_type == "HUMANEVAL":
        args.agent_cls = HumanevalJudgeAgent
    else:
        raise ValueError(f"Unsupported eval type: {args.eval_type}")
    args.judge_prompt_template_path = JUDGE_PROMPT_TEMPLATES.get(args.eval_type)

    filename, _ = os.path.splitext(os.path.basename(args.prompts_file))
    JudgeFlow.run_flow(
        dataset=local_dataset(f"{filename}_{args.eval_type.lower()}_judge", dir=args.output_dir),
        data_kwargs={"args": args, "remove_cols": ["idx"]},
        run_kwargs={"multiprocess": True, "ordered": False, "num_worker_procs": args.num_worker_procs},
        repeat=args.repeat,
    )
