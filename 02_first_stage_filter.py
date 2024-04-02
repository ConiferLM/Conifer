import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, prompt, item):
    tmp_saver = {}
    client = OpenAI(api_key=API_KEY)
    for k, v in item.items():
        for _ in range(MAX_API_RETRY):
            try:
                prompt_ = prompt % v
                response = client.chat.completions.create(
                    model='gpt-4-turbo-preview',
                    max_tokens=1,
                    temperature=0.0,
                    messages=[{
                        'role': 'user',
                        'content': prompt_,
                    }],
                )
                content = response.choices[0].message.content
            except Exception as e:
                print(f"failed... {e}")
                continue
            content = content.lower().strip()
            if "yes" in content:
                tmp_saver[k] = v
            break
    return query, tmp_saver

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "01_reframed_questions.json")
    output_file = os.path.join(args.dir, "02_reframed_questions_filtered.json")

    reframed_questions = json.load(open(input_file, "r"))
    reframed_questions_filtered = {}

    # load prompt
    prompt = get_prompt("first stage filter")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            reframed_questions_filtered = json.load(f)
            dedup_set = set(reframed_questions_filtered.keys())
    except:
        pass

    passed_args = []
    for query, item in reframed_questions.items():
        if query in dedup_set:
            continue
        passed_args.append((query, prompt, item))
    
    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, ans = future
                reframed_questions_filtered[query] = ans
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(reframed_questions_filtered, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")
    
    with open(output_file, "w") as f:
        json.dump(reframed_questions_filtered, f, ensure_ascii=False, indent=4)
    print(f"File Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)
