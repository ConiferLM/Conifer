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
                max_tokens=1024,
                top_p=0.8,
                temperature=0.7,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"failed... {e}")
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
    # load input queries and output file
    input_file = os.path.join(args.dir, "selected_query.txt")
    input_queries = []
    with open(input_file, "r") as f:
        for line in f:
            input_queries.append(line.strip())
    
    output_file = os.path.join(args.dir, "01_reframed_questions.json")
    reframed_questions = {}
    
    # load prompt
    prompt = get_prompt("question reframing")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            reframed_questions = json.load(f)
            dedup_set = set(reframed_questions.keys())
    except:
        pass

    passed_args = []
    for sample in input_queries:
        if sample in dedup_set:
            continue
        passed_args.append((sample, prompt % sample))
    
    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, ans = future
                reframed_questions[query] = ans
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(reframed_questions, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")
    
    with open(output_file, "w") as f:
        json.dump(reframed_questions, f, ensure_ascii=False, indent=4)
    print(f"File Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)