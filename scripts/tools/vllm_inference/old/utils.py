import json
from datasets import load_dataset
import os
import itertools


def load_txt(filename):
    with open(filename, "r") as f:
        print(f"loading {filename}")
        return f.read()


# load json
def load_json(filename):
    print(f"loading {filename}")
    with open(filename, "r") as f:
        return json.load(f)


def load_jsonl(filename):
    print(f"loading {filename}")

    if filename.startswith("hf/"):
        filename = filename[3:]
        return load_dataset(filename)["train"].to_list()

    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as reader:
        if limit:
            data = [json.loads(obj) for obj in itertools.islice(reader.readlines(), limit)]
        else:
            data = [json.loads(obj) for obj in reader.readlines()]
    return data


def write_jsonl(file_path, data):
    # find parent dir
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(file_path, "w", encoding="utf-8") as writer:
        for obj in data:
            writer.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_messages_training_data(a, completions_field="completions"):
    messages = []
    if completions_field not in a:
        return None
    for i in range(len(a[completions_field])):
        cot = ""
        response = ""
        combined = a[completions_field][i]
        try:
            comp = a[completions_field][i]
            # check for cot block - earlier we were using <|dummy_86|> and <|dummy_87|> but now we use <think> and </think>
            if "<|dummy_86|>" in comp:
                cot = comp.split("<|dummy_86|>")[1].split("<|dummy_87|>")[0].strip()
                response = comp.split("<|dummy_86|>")[1].split("<|dummy_87|>")[1].strip()
            elif "<|dummy_87|>" in comp:  # model didnt generate think token but still doing cot
                cot = comp.split("<|dummy_87|>")[0]
                response = comp.split("<|dummy_87|>")[1]
            elif "<think>" in comp:
                cot = comp.split("<think>")[1].split("</think>")[0].strip()
                response = comp.split("<think>")[1].split("</think>")[1].strip()
            elif "</think>" in comp:
                cot = comp.split("</think>")[0]
                response = comp.split("</think>")[1]
            else:
                cot = ""
                response = comp
            msg = {
                "response": response,
                "cot": cot,
                "combined": f"{cot}\n{response}",
            }
            messages.append(msg)
        except IndexError as e:
            msg = {
                "response": combined,  # use combined for judge eval to support non-cot models
                "cot": cot,
                "combined": combined,
            }
            messages.append(msg)
            continue
    if len(messages) == 0:
        return None
    return messages


def count_tokens_estimate(text):
    return len(text) // 3.5
