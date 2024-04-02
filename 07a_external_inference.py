from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from utils import get_level_x_pairs
import argparse
import os
import torch
from tqdm import tqdm

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    return tokenizer.apply_chat_template(
        messages, tokenize=False
    )

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "06_answer.json")
    output_file = os.path.join(args.dir, args.output)
    pairs = get_level_x_pairs(input_file, 5)
    print(pairs)
    
    model_name = args.model
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    # load infer data (task 5 only)
    prediction = []
    for sample in tqdm(pairs):
        instruction = sample['instruction']
        if args.chat_template is not None:
            from fasttext import get_conversation_template
            chat_template = get_conversation_template(args.chat_template)
            chat_template.append_message(chat_template.roles[0], instruction)
            chat_template.append_message(chat_template.roles[1], None)
            text = chat_template.get_prompt()
        else:
            text = apply_chat_template({"messages": [{"role": "user", "content": instruction}]}, tokenizer)

        input_ids = tokenizer([text]).input_ids
        output_ids = model.generate(torch.as_tensor(input_ids).cuda(), max_new_tokens=2048, temperature=0.8, top_p=0.6, do_sample=True)
        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        prediction.append({
            'prompt': instruction,
            'predict': output
        })

    print(prediction[:5])
    print(f"Saving to {output_file}")
    with open(output_file, "w") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--dir", type=str, help="output directory", default="conifer_data")
    parser.add_argument("--output", type=str, help="output file path", default="07_reference.json")
    parser.add_argument("--chat-template", type=str, default=None)
    args = parser.parse_args()
    
    main(args)