import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, pred, ref, prompt):
    for _ in range(MAX_API_RETRY):
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                max_tokens=1024,
                top_p=0.6,
                temperature=0.8,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
            )
            content = response.choices[0].message.content
            print(content)
        except Exception as e:
            print(f"failed... {e}")
            continue
        content = content.lower().strip()
        if "yes" in content or "does follow" in content:
            return query, [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': ref}
            ]
        else:
            return query, [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': pred},
                {'role': 'user', 'content': content + ' Please give a revised version.'},
                {'role': 'assistant', 'content': ref}
            ]


def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "06_answer.json")
    reference_file = os.path.join(args.dir, args.reference_file)
    output_file = os.path.join(args.dir, "06_answer_external.json")

    gt_answer = json.load(open(input_file, "r"))
    pred = json.load(open(reference_file, "r"))

    external = {}

    # load prompt
    prompt = get_prompt("get answer external")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            external = json.load(f)
            dedup_set = set(external.keys())
    except:
        pass

    passed_args = []
    for sample in pred:
        pred_question = sample['prompt']
        pred_answer = sample['predict']
        assert pred_question in gt_answer
        if pred_question in dedup_set:
            continue
        passed_args.append((pred_question, pred_answer, gt_answer[pred_question]['answer'], prompt % (pred_question, pred_answer, gt_answer[pred_question]['answer'])))

    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, messages = future
                external[query] = messages
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(external, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")
    
    with open(output_file, "w") as f:
        json.dump(external, f, ensure_ascii=False, indent=4)
    print(f"File Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--reference_file", type=str, help="input file path for the generated response", default="07_reference.json")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)
