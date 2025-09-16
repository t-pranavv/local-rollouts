from datasets import load_dataset, get_dataset_config_names
import json

BENCHMARKS = {
    "aime_2025": 30,
    "aime_2024": 30,
    "aime_2023": 30,
    "aime_2022": 30,
    "gpqa_diamond": 198,
    "math_500_test": 500,
}

HF_DATASET_MAP = {
    "aime_2025": "opencompass/AIME2025",
    "aime_2024": "opencompass/AIME2024",
    "aime_2023": "opencompass/AIME2023",
    "aime_2022": "opencompass/AIME2022",
    "gpqa_diamond": "Idavidrein/gpqa",
    "math_500_test": "HuggingFaceH4/MATH-500",
}

def dataset_to_jsonl(dataset, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def download_and_save(dataset_key: str):
    dataset_name = HF_DATASET_MAP[dataset_key]

    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = []

    if not configs:  # no configs
        configs = [None]

    for cfg in configs:
        if cfg is None:
            ds = load_dataset(dataset_name, split="test")
            suffix = ""
        else:
            ds = load_dataset(dataset_name, cfg, split="test")
            suffix = f"-{cfg}"

        output_path = f"/home/t-pranavv/phyagi-sdk/prompts/{dataset_key}{suffix}.jsonl"
        dataset_to_jsonl(ds, output_path)
        print(f"✅ Saved {dataset_key}{' ' + (cfg or '')} test split to {output_path}")

def main():
    for key in BENCHMARKS:
        download_and_save(key)

if __name__ == "__main__":
    main()
