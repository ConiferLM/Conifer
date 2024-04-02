import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, item, prompt):
    tmp_save = {
        'constrainted_questions': {}
    }
    for k, v in item.items():
        if k != 'constrainted_questions':
            tmp_save[k] = v

    client = OpenAI(api_key=API_KEY)
    for diff, question in item['constrainted_questions'].items():
        for _ in range(MAX_API_RETRY):
            try:
                prompt_ = prompt % question
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
                print(f"failed...{e}")
                continue
            content = content.lower().strip()
            if "yes" in content:
                tmp_save['constrainted_questions'][diff] = question
            break
    return query, tmp_save
    

def main(args):
    # load input queries
    for suffix in ['', '_format_number']:
        input_file = os.path.join(args.dir, f"04_constrainted_question{suffix}.json")
        output_file = os.path.join(args.dir, f"05_constrainted_question{suffix}_filtered.json")

        constrainted_question = json.load(open(input_file, "r"))
        constrainted_question_filtered = {}

        # load prompt
        prompt = get_prompt("second stage filter")

        # load saved samples
        dedup_set = set()
        try:
            with open(output_file, "r") as f:
                constrainted_question_filtered = json.load(f)
                dedup_set = set(constrainted_question_filtered.keys())
        except:
            pass

        passed_args = []
        for query, item in constrainted_question.items():
            if query in dedup_set:
                continue
            passed_args.append((query, item, prompt))
        
        with ThreadPoolExecutor(max_workers=args.worker) as executor:
            for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
                if future is not None:
                    query, tmp_save = future
                    constrainted_question_filtered[query] = tmp_save
                    if save_count % args.save_iterval == 0:
                        with open(output_file, "w") as f:
                            json.dump(constrainted_question_filtered, f, ensure_ascii=False, indent=4)
                        print(f"File Saved")
        
        with open(output_file, "w") as f:
            json.dump(constrainted_question_filtered, f, ensure_ascii=False, indent=4)
        print(f"File Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=1)
    args = parser.parse_args()
    main(args)