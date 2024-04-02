import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, prompt):
    for _ in range(MAX_API_RETRY):
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                max_tokens=2048,
                top_p=0.8,
                temperature=1.0,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"failed...{e}")
            continue
        try:
            if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
                content = content[7:-3]
            answer = json.loads(content)
        except Exception as e:
            print(f"json failed to parse: {e}")
            print(f"content: {content}")
            return None
        return query, answer

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "02_reframed_questions_filtered.json")
    output_file = os.path.join(args.dir, "03_constraint_list.json")

    reframed_questions = json.load(open(input_file, "r"))
    constraints = {}

    # load prompt
    prompt = get_prompt("constraints generation")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            constraints = json.load(f)
            dedup_set = set(constraints.keys())
    except:
        pass

    passed_args = []
    for _, item in reframed_questions.items():
        for _, query in item.items():
            if query in dedup_set:
                continue
            passed_args.append((query, prompt % query))

    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, ans = future
                constraints[query] = ans
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(constraints, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")

    # save
    with open(output_file, "w") as f:
        json.dump(constraints, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)
