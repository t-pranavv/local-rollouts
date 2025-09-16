import os
import random
import math
import argparse
import re
import threading
import requests
import json
import base64
import json

from utils import load_jsonl, load_txt, get_messages_training_data
import shutil

SEED = 42

import phigen
from phigen import (
    Generation,
    SharedSingleton,
    local_dataset,
    complete_text,
    complete_json,
    set_log_level,
    interactive,
)

# INTERACTIVE = False

# client = new_client_by_name("phyagi")
# if INTERACTIVE:
#    client = interactive(client=client)
# set_default_client(client)

set_log_level("INFO")
logger = phigen.logger()

SUPPORTED_EVALS = ["AIME", "MATH", "GPQA", "HUMANEVAL"]
SUPPORTED_EVALS.append("AIME_SIMPLE")  # simplely

## Config
JUDGEMODEL = "gpt-4o-impact"

QUESTION_FIELD = "prompt"
GT_SOLUTION_FIELD = "answer"
COMPLETION_SOLUTION_FIELD = "response"  # cot #combined


class GPTJudgeData(SharedSingleton):
    def __init__(self, data_path: str):
        self.prompt_dir = "./prompts"
        self.compare_answers_math_template = load_txt(f"{self.prompt_dir}/compare_answers_math.jinja")
        self.compare_answers_gpqa_template = load_txt(f"{self.prompt_dir}/compare_answers_gpqa.jinja")
        self.compare_answers_math_simple_template = load_txt(
            f"{self.prompt_dir}/compare_answers_math_simple.jinja"
        )  # simpleonly

        self.all_problems = load_jsonl(data_path)
        logger.info(f"Loaded {len(self.all_problems)} completed problems.")


# Replace the current global auth function
class AuthProvider:
    def __init__(self):
        self._bearer_token_provider = None

    def get_ces_headers(self):
        if self._bearer_token_provider is None:
            from azure.identity import AzureCliCredential, get_bearer_token_provider

            credential = AzureCliCredential()
            self._bearer_token_provider = get_bearer_token_provider(
                credential, "api://17b0ad65-ed36-4194-bb27-059c567bc41f/.default"
            )
        return {"Content-Type": "application/json", "Authorization": "Bearer " + self._bearer_token_provider()}


CES_URL = "https://ces-dev1.azurewebsites.net/api/ces/executeCode"


def execute_code(code, language="python3"):
    data = {"code": base64.b64encode(code.encode("utf-8")).decode("utf-8"), "language": language}
    auth_provider = AuthProvider()
    response = requests.post(CES_URL, headers=auth_provider.get_ces_headers(), data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 408:
        return {
            "ExitCode": 408,
            "Output": "",
        }
    else:
        raise RuntimeError(f"Request failed with status code {response.status_code}: {response.text}")


class GPTJudge(Generation):
    index: int
    data_path: str
    prompt_field: str
    answer_field: str
    completions_field: str
    n_samples: int
    full_seed: dict
    question_seed: str
    gt_solution: str
    ckpt_cots: list[str]
    ckpt_completions: list[str]

    # full_judgments: list[dict]
    final_scores: list[list[bool]]  # k compeltions * n judgements
    final_prompts: list[list[bool]]
    final_responses: list[list[bool]]

    def __init__(self, problem_index: int, data_config: dict[str, str], n_samples: int = 1):
        self.index = problem_index
        self.data_path = data_config["path"]
        self.prompt_field = data_config.get("prompt_field", QUESTION_FIELD)
        self.answer_field = data_config.get("answer_field", GT_SOLUTION_FIELD)
        self.completions_field = data_config.get("completions_field", "completions")
        self.n_samples = n_samples

    def initialize(self):
        self.full_seed = self.data.all_problems[self.index]
        self.question_seed = self.full_seed[self.prompt_field]
        self.gt_solution = self.full_seed[self.answer_field]
        self.ckpt_completions = get_messages_training_data(self.full_seed, completions_field=self.completions_field)
        if not self.ckpt_completions:
            return

        # self.full_judgments = [[None] * self.n_samples] * len(self.ckpt_completions)

        self.final_scores = [[None for _ in range(self.n_samples)] for _ in range(len(self.ckpt_completions))]
        self.final_responses = [[None for _ in range(self.n_samples)] for _ in range(len(self.ckpt_completions))]
        self.final_prompts = [[None for _ in range(self.n_samples)] for _ in range(len(self.ckpt_completions))]

    def run(self):
        try:
            self.data = GPTJudgeData(self.data_path)
            self.initialize()
            if self.ckpt_completions is None:
                logger.debug(f"KeyError or IndexError ISSUE with completion seed {self.index}")
                return

            for k in range(len(self.ckpt_completions)):
                for i in range(self.n_samples):
                    self.complete_compare_full(completion_k=k, judge_i=i)

        except Exception as e:
            logger.exception(f"Error in log index {self.index}:\n{e}")
            raise e

    def score_math_eval_response(self, response):
        if "unknown" in response.lower():
            return None

        equivalent = "[yes]" in response.lower() and "[no]" not in response.lower()

        def check(a, b, epsilon=1e-8):
            # a: extracted model solution, b: ground truth answer, epsilon: tolerance
            if type(a) == str and "math" in a:
                try:
                    a = eval(a, {"math": math})
                except Exception as e:
                    print(f"math eval error in {a}:", e)
            if type(b) == str and "math" in b:
                try:
                    b = eval(b, {"math": math})
                except Exception as e:
                    print(f"math eval error in {b}:", e)
            if type(b) == int:
                return a == b
            if type(b) == str:
                if type(a) == int and "%" in b:
                    a = str(a) + "%"
                return a.replace(" ", "").lower() == b.replace(" ", "").lower()
            if type(b) == set:
                return a == b
            if type(b) == list:
                return a == b
            if type(b) == tuple:
                return a == b
            return abs(a - b) < epsilon

        if "[check]" in response.lower():
            code = response.lower().split("[check]")[-1].strip().split("#")[0].strip()
            code = code.replace("```python", "").replace("```", "").strip()
            # assert('\n' not in code)
            logger.debug(f"evaluating:\n{code}")
            equivalent = None
            count = 0
            while count < 10:
                try:
                    equivalent = eval(code)
                    break
                except BaseException as e:
                    print("Error evaluating code:", e, f"```{code}```")
                    count += 1
                    continue

        return equivalent

    def score_gpqa_eval_response(self, response):
        if "unknown" in response.lower():
            return None
        equivalent = "[yes]" in response.lower() and "[no]" not in response.lower()

        return equivalent

    def score_math_simple_eval_response(self, response):  # sipmleonly
        assert "extracted_answer" in response, f"Missing extracted_answer; Response: {response}"
        extracted_answer = response["extracted_answer"]
        assert "correct_answer" in response, f"Missing correct_answer; Response: {response}"
        correct_answer = response["correct_answer"]
        try:
            assert correct_answer == type(correct_answer)(
                self.gt_solution
            ), f"Correct answer mismatch; {correct_answer} != {self.gt_solution}"
        except Exception as e:
            with open("errored_prompt.txt", "a") as f:
                import json

                f.write(json.dumps(response))
                f.write("\n===========================\n")

        if "equivalence" not in response:
            print(f"Missing equivalence; Response: {response}")
            return None
        # assert "equivalence" in response, f"Missing equivalence; Response: {response}"
        assert "comparison_discussion" in response, f"Missing summary; Response: {response}"

        if extracted_answer == "unknown":
            return None
        equivalent = bool(int(response["equivalence"]))
        return equivalent

    def test_code(self, completion, **judge_kwargs):
        # Taken from https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
        def find_code(completion):
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(completion)
            extracted_answer = matches[0] if len(matches) >= 1 else completion
            extracted_answer = extracted_answer[extracted_answer.find(":\n    ") + 2 :]  # remove signature
            return extracted_answer

        code = find_code(completion)
        snippet = (
            code + self.full_seed["test"] + f"\ncheck({self.full_seed['entry_point']})"
        )  # self.full_seed["signature"] +
        response = execute_code(snippet)
        return response

    def score_humaneval_eval_response(self, response):
        return response["ExitCode"] == 0

    def complete_compare_full(self, completion_k: int, judge_i: int):
        if EVAL_TYPE in ["AIME", "MATH"]:
            prompt = self.data.compare_answers_math_template.format(
                correct_solution=self.gt_solution,
                student_solution=self.ckpt_completions[completion_k][COMPLETION_SOLUTION_FIELD],
            )
            complete_function = complete_text  # sipmleonly
            score_function = self.score_math_eval_response
        elif EVAL_TYPE == "GPQA":
            prompt = self.data.compare_answers_gpqa_template.format(
                correct_solution=self.gt_solution,
                student_solution=self.ckpt_completions[completion_k][COMPLETION_SOLUTION_FIELD],
            )
            complete_function = complete_text  # sipmleonly
            score_function = self.score_gpqa_eval_response
        elif EVAL_TYPE == "AIME_SIMPLE":  # sipmleonly
            prompt = self.data.compare_answers_math_simple_template.format(
                correct_solution=self.gt_solution,
                student_solution=self.ckpt_completions[completion_k][COMPLETION_SOLUTION_FIELD],
            )
            complete_function = complete_json
            score_function = self.score_math_simple_eval_response
        elif EVAL_TYPE == "HUMANEVAL":
            prompt = self.ckpt_completions[completion_k][COMPLETION_SOLUTION_FIELD]
            complete_function = self.test_code
            score_function = self.score_humaneval_eval_response
        else:
            logger.error(f"EVAL_TYPE: {EVAL_TYPE} not supported.")
        with interactive():
            response = complete_function(prompt, model=JUDGEMODEL, temperature=0.01, tier="impact")

        self.final_prompts[completion_k][judge_i] = prompt
        # self.full_judgments[i] = response
        self.final_scores[completion_k][judge_i] = score_function(response)
        self.final_responses[completion_k][judge_i] = response
        return

    def get_training_data(self):
        # self.full_seed["n_full_judgment"] = self.full_judgments
        self.full_seed["scores"] = self.final_scores
        scores = (
            [[s for s in completion_scores if s is not None] for completion_scores in self.final_scores]
            if self.final_scores
            else []
        )
        self.full_seed["aggregated_score"] = [
            sum(scores_k) / len(scores_k) if len(scores_k) > 0 else None for scores_k in scores
        ]
        self.full_seed["judge_responses"] = self.final_responses
        self.full_seed["judge_prompts"] = self.final_prompts
        return self.full_seed
        # try:
        #     return {
        #         "score": sum(self.full_seed["aggregated_score"])/len(self.full_seed["aggregated_score"]),
        #         **self.full_seed
        #     }
        # except Exception as e:
        #     print("*****")
        #     print(self.full_seed.keys())
        #     print("*****")
        #     raise e


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run judgement eval.")
    parser.add_argument("-i", "--results_file", type=str, help="Path to the file with results and prompts jsonl.")
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="prompt",
        help="Field referring to prompt/problem in the results jsonl file.",
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
    parser.add_argument(
        "-n", "--n_samples", default=1, type=int, help="number of eval samples."
    )  # take all samples in completions field
    parser.add_argument("-o", "--output_dir", type=str, help="Path to save results.")
    parser.add_argument(
        "--eval_type", type=str, default="MATH", help=f"Type of eval, currently supported: {SUPPORTED_EVALS}"
    )
    args = parser.parse_args()

    file_path = args.results_file
    n_samples = args.n_samples
    EVAL_TYPE = args.eval_type

    if EVAL_TYPE not in SUPPORTED_EVALS:
        logger.error(f"EVAL_TYPE: {EVAL_TYPE} not supported, use: {SUPPORTED_EVALS}.")
        raise
    id = random.randint(0, 100000)
    logger.info(f"Starting judgement eval for {file_path} with model {JUDGEMODEL}.")
    if args.output_dir is None:
        output_dir = f"./judged_outputs/{file_path.split('/')[-1].replace('.jsonl', '')}_scored_x{n_samples}_judgements_{EVAL_TYPE}_type/"
    else:
        output_dir = args.output_dir

    data_config = {
        "path": file_path,
        "prompt_field": args.prompt_field,
        "answer_field": args.answer_field,
        "completions_field": args.completions_field,
    }

    with open("errored_prompt.txt", "w") as f:
        f.write("")
    # Delete the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    full_size = len(load_jsonl(file_path))
    with local_dataset(f"gpt_judge_{id}"), GPTJudge.open_training_data_writer(
        writer_name_or_cls="jsonl", path=output_dir
    ) as writer:
        GPTJudge.create(
            params=[dict(problem_index=i, data_config=data_config, n_samples=n_samples) for i in range(0, full_size)]
        )
        logger.info(f"Starting judegement eval for {full_size} problems.")
        GPTJudge.run_unfinished(training_data_writer=writer, max_concurrent_jobs=50)

        # Report average score
        # scores = []
        # for instance in GPTJudge.load_successful():
        #     output = instance.get_training_data()
        #     scores.append(sum(output["aggregated_score"])/len(output["aggregated_score"]))
        # logger.info(f"Average score: {sum(scores)/len(scores)} (over {len(scores)} instances)")
