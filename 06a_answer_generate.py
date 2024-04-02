import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, diff, question, prompt):
    tmp_save = {}
    for _ in range(MAX_API_RETRY):
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                max_tokens=2048,
                top_p=0.6,
                temperature=0.8,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"failed...{e}")
            continue
        tmp_save = {
            'query': query,
            'diff': diff,
            'question': question,
            'answer': content
        }
    return question, tmp_save

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "05_constrainted_question_filtered.json")
    output_file = os.path.join(args.dir, "06_answer.json")

    constrainted_question = json.load(open(input_file, "r"))
    answer = {}

    # load prompt
    prompt = get_prompt("get answer")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            answer = json.load(f)
            dedup_set = set(answer.keys())
    except:
        pass

    passed_args = []
    for query, item in constrainted_question.items():
        for diff, question in item['constrainted_questions'].items():
            if question in dedup_set:
                continue
            passed_args.append((query, diff, question, prompt % question))

    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                question, tmp_save = future
                answer[question] = tmp_save
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(answer, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")
    
    with open(output_file, "w") as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)
    print(f"File Saved")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)
