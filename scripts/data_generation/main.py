import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model.model_adapter import get_conversation_template

system_prompt = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe.  "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="Aeala/ShareGPT_Vicuna_unfiltered"
    )
    parser.add_argument(
        "--data-files", type=str, default="ShareGPT_V4.3_unfiltered_cleaned_split.json"
    )
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--outdir", type=str, default="data/")
    parser.add_argument("--model-type", type=str, default="llama-2-chat")
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.float16
    )

    def preprocess_tokenize(data):
        new_data = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(data["id"])):
            conv = get_conversation_template(args.model_type)
            conv.system_message = system_prompt
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source = data["conversations"][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert (
                    role == conv.roles[j % 2]
                ), f"current role should be {conv.roles[j % 2]}"
                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                conv.append_message(role, sentence["value"])
            conversation = conv.get_prompt()
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            sep = conv.sep + conv.roles[1] + " "
            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            for k, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # Ignore the user instructions
                loss_mask[cur_len : cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len += 2

                if k != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0

            new_data["conversation"].append(conversation)
            new_data["input_ids"].append(input_ids[None, :])
            new_data["loss_mask"].append(loss_mask[None, :])
        return new_data

    @torch.no_grad()
    def preprocess_forward(data):
        device = model.device
        input_ids = data["input_ids"]
        outputs = model(input_ids.to(device), output_hidden_states=True)
        hidden_state_last = outputs.hidden_states[-1]
        return {
            "input_ids": input_ids.cpu()[0],
            "hidden_state": hidden_state_last.cpu()[0],
            "loss_mask": data["loss_mask"].cpu()[0],
        }

    dataset = load_dataset(
        args.data_path, data_files=args.data_files, split="train"
    ).shuffle(seed=42)
    dataset = dataset.map(
        preprocess_tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        num_proc=args.num_proc,
    )
    dataset.set_format("torch")

    for index, data in enumerate(tqdm(dataset)):
        data = preprocess_forward(data)
        torch.save(data, f"{args.outdir}/data_{index}.ckpt")
