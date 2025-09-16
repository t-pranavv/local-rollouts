import json
import os
import re
import argparse
import numpy as np
import pandas as pd

BENCHMARKS = {
    "aime_2025": 30,
    "aime_2024": 30,
    "aime_2023": 30,
    "aime_2022": 30,
    "gpqa_diamond": 198,
    "math_500_test": 500,
}

BENCHMARK_SUFFIX = {
    "aime_2025": "_2025",
    "aime_2024": "_2024",
    "aime_2023": "_2023",
    "aime_2022": "_2022",
    "gpqa_diamond": "",
    "math_500_test": "",
}

EVAL_TYPES = {
    "aime_2025": ["AIME", "AIME_SIMPLE"],
    "aime_2024": ["AIME", "AIME_SIMPLE"],
    "aime_2023": ["AIME", "AIME_SIMPLE"],
    "aime_2022": ["AIME", "AIME_SIMPLE"],
    "gpqa_diamond": ["GPQA"],
    "math_500_test": ["MATH"],
}


def process_jsonl_results(result_folder, num_expected_problems):
    """Process JSONL result files and extract evaluation metrics."""
    if not os.path.exists(result_folder):
        return None, f"Folder does not exist: {result_folder}"

    # List all *.jsonl files in this folder (excluding annotation files)
    jsonl_files = [f for f in os.listdir(result_folder) if f.endswith(".jsonl") and not f.endswith("_annot.jsonl")]

    if not jsonl_files:
        return None, "No result files found"

    # Load all results
    results = []
    for jsonl_file in jsonl_files:
        with open(os.path.join(result_folder, jsonl_file), "r") as f:
            results.extend([json.loads(line) for line in f.readlines()])

    if not results:
        return None, "Empty result files"

    if len(results) != num_expected_problems:
        error = (f"Expected {num_expected_problems} results, got {len(results)}", len(results))
        if not len(results):
            return None, error
    else:
        error = None

    # Process metrics for each problem
    metrics = {
        "pass_at_1": [],
        "pass_at_5": [],
        "all_scores": [],
        # "std_pass_at_1": [],
        "unknowns": [],
    }

    for result in results:
        # Process all scores
        # scores are from GPTEvals and are #completions-for-prompt x #n-in-gpteval
        # all_scores = [0 if s is None else int(s) for s1 in result["scores"] for s in s1]
        # avg_all_scores = sum(all_scores) / len(all_scores) if all_scores else 0

        # Process aggregated scores
        # aggregated_scores are aggregated within GPTEvals per completion (for each prompt len(result["aggregated_score"])=#completions-per-prompt; This is average of scores across n-samples in GPTEval per completion. We typically use n=1, so effetively no aggregation.
        agg_scores = result.get("aggregated_score", [None])

        # assert len(agg_scores) == 10, f"Expected 5 aggregated scores, got {len(agg_scores)}"

        # Replace None with 0
        agg_scores_clean = [0 if s is None else s for s in agg_scores]
        unknowns = [1 if s is None else 0 for s in agg_scores]

        # Calculate metrics
        pass_at_1 = sum(agg_scores_clean) / len(agg_scores_clean)
        pass_at_5 = any([s > 0.5 for s in agg_scores_clean])
        # std_pass_at_1 = math.sqrt(sum([(s - pass_at_1)**2 for s in agg_scores_clean]) / len(agg_scores_clean))
        avg_unknowns = sum(unknowns) / len(unknowns)

        # Add to results
        metrics["pass_at_1"].append(pass_at_1)
        metrics["pass_at_5"].append(pass_at_5)
        metrics["all_scores"].append(agg_scores_clean)
        # metrics["std_pass_at_1"].append(std_pass_at_1)
        metrics["unknowns"].append(avg_unknowns)

    # Calculate averages across all problems
    if error is None:
        assert all(
            len(metrics[key]) == num_expected_problems for key in metrics
        ), f"metrics does not have right number of problems: {[len(metrics[key])==num_expected_problems for key in metrics]}, {len(results)}"

    accuracy_per_sample = []
    n_samples = len(metrics["all_scores"][0])
    if n_samples == 0:
        print("[WARNING]: No samples found in all_scores, returning empty results.")
        accuracy_per_sample = [0.0]
    else:
        for i in range(n_samples):
            acc_per_gen = []
            for s in metrics["all_scores"]:
                if len(s) > i:
                    acc_per_gen.append(s[i])
            if len(acc_per_gen) < n_samples:
                print(
                    f"[WARNING]: n_samples={n_samples} but only found {len(acc_per_gen)} accuracy scores in the generation"
                )
                if len(acc_per_gen) == 0:
                    acc_per_gen = [0.0]
            accuracy_per_sample.append(np.mean(acc_per_gen))

    # assert len(accuracy_per_sample)==5
    results_summary = {
        "pass_at_1": sum(metrics["pass_at_1"]) / len(metrics["pass_at_1"]),
        "pass_at_5": sum(metrics["pass_at_5"]) / len(metrics["pass_at_5"]),
        "std": np.std(accuracy_per_sample),
        # "mean_std": sum(metrics["std_pass_at_1"]) / len(metrics["std_pass_at_1"]),
        # "std_pass_at_1_per_prompt": metrics["std_pass_at_1"],
        "unknowns": sum(metrics["unknowns"]) / len(metrics["unknowns"]),
    }

    return results_summary, error


def evaluate_models(output_dir, temperature):
    """Main function to evaluate models across benchmarks."""
    # Initialize result dictionaries
    metrics = {
        "pass_at_1": {},
        "pass_at_5": {},
        "std": {},
        # "mean_std": {},
        # "std_pass_at_1_per_prompt": {},
        "unknowns": {},
    }

    # Track issues
    incomplete_folders = None
    no_results_folders = {}

    # Process each benchmark
    for benchmark in BENCHMARKS:
        num_expected_problems = BENCHMARKS[benchmark]

        # Process each evaluation type
        for eval_type in EVAL_TYPES[benchmark]:
            experiment_name = f"{benchmark}"

            # Process each temperature
            result_folder = os.path.join(output_dir, benchmark, eval_type)

            print(f"Checking folder: {result_folder}")

            # Process results
            results_summary, error = process_jsonl_results(result_folder, num_expected_problems)

            # Handle errors
            if error:
                if isinstance(error, tuple):
                    incomplete_folders = (result_folder, error[1])
                    if not results_summary:
                        continue
                elif "Folder does not exist" in error or "No result files found" in error:
                    no_results_folders[result_folder] = no_results_folders.get(result_folder, []) + [eval_type]
                    continue

            # Record metrics
            metric_key = f"{eval_type}{BENCHMARK_SUFFIX[benchmark]}_{temperature}"

            if results_summary:
                metrics["pass_at_1"][metric_key] = results_summary["pass_at_1"]
                metrics["pass_at_5"][metric_key] = results_summary["pass_at_5"]
                metrics["std"][metric_key] = results_summary["std"]
                # metrics["std_pass_at_1_per_prompt"][metric_key] = results_summary["std_pass_at_1_per_prompt"]
                metrics["unknowns"][metric_key] = results_summary["unknowns"]

                print(
                    f"Results for {metric_key}: Pass@1={results_summary['pass_at_1']:.4f}, pass_at_5={results_summary['pass_at_5']:.4f}, Unknowns={results_summary['unknowns']:.4f}"
                )

    # Report issues
    print("\n=== Incomplete Folders ===")
    print(f"{incomplete_folders}")
    print("\n===============================")

    print("\n=== Missing Results Folders ===")
    for folder, eval_types in no_results_folders.items():
        print(f"{folder}: {eval_types}")
    print("\n===============================")

    return metrics, incomplete_folders


def main():
    parser = argparse.ArgumentParser(description="Summarize results from JSONL files.")
    # parser.add_argument("--temperature", type=str, help="temperature.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    try:
        temperature = re.search(r"temperature_(\d+(?:\.\d+)?)", args.output_dir).group(0)
    except:
        print("WARNING: Could not find temperature in output_dir. Using default temperature of 0.8 for printing.")
        temperature = "0.8"
    # Run evaluation
    metrics, incomplete_folders = evaluate_models(args.output_dir, temperature)

    columns = ["pass_at_1", "std", "pass_at_5", "unknowns"]
    filtered_metrics = {col_name: metrics[col_name] for col_name in columns}
    df = pd.DataFrame(filtered_metrics)
    print(df)
    df.to_csv(os.path.join(args.output_dir, "summatized_scores.csv"), index=True)


if __name__ == "__main__":
    main()
