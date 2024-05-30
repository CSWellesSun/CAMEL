import json
from transformers import AutoTokenizer
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--base-result-path", type=str, default=None)
    parser.add_argument("--result-path", type=str, default=None)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    jsonl_file = args.base_result_path
    jsonl_file_base = args.result_path
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]["turns"]
        tokens = sum(datapoint["choices"][0]["new_tokens"])
        times = sum(datapoint["choices"][0]["wall_time"])
        speeds.append(tokens / times)

    data = []
    with open(jsonl_file_base, "r", encoding="utf-8") as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time = 0
    total_token = 0
    speeds0 = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]["turns"]
        tokens = 0
        for i in answer:
            tokens += len(tokenizer(i).input_ids) - 1
        times = sum(datapoint["choices"][0]["wall_time"])
        speeds0.append(tokens / times)
        total_time += times
        total_token += tokens

    print("speed", np.array(speeds).mean())
    print("speed0", np.array(speeds0).mean())
    print("ratio", np.array(speeds).mean() / np.array(speeds0).mean())
