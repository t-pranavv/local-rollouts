import os
from glob import glob
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merges output files from multiple runs.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))

    print(f"Rank [{rank}]: Merging output files from: {args.output_dir}")
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        files = list(glob(f"{args.output_dir}/results_rank*.jsonl"))
        print(f"Rank [{rank}]: Found these files: {files}")
        for file in files:
            with open(file, "r", encoding="utf-8") as in_file:
                for line in in_file:
                    out_file.write(line)
