import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]


def get_response(query, transformed_constraint_list, prompt):
    tmp_save = {}
    for _ in range(MAX_API_RETRY):
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                max_tokens=1024,
                top_p=0.3,
                temperature=0.4,
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
        tmp_save = {
            'constraints': transformed_constraint_list,
            'constrainted_questions': answer
        }
        return query, tmp_save

def transform_constraints(constraints):
    constraints_str = ""
    for k, v in constraints.items():
        if type(v) is str:
            v = v.split('，')
            if len(v) == 1:
                v = v[0].split('、')
            v = str(v)
        elif type(v) is dict:
            tmp_str = ""
            for k_, v_ in v.items():
                if type(v_) is str:
                    v_ = str(v_.split('、'))
                else:
                    v_ = str(v_)
                tmp_str += f"{k_}：{v_}\n"
            v = tmp_str
        elif type(v) is list:
            v = str(v)
        if type(v) is not str:
            return {}
        constraints_str += f"{k}: {v}\n"
    return constraints_str

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "03_constraint_list.json")
    output_file = os.path.join(args.dir, "04_constrainted_question.json")

    constraints = json.load(open(input_file, "r"))
    constrainted_question = {}

    # load prompt
    prompt = get_prompt("general recombination")

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            constrainted_question = json.load(f)
            dedup_set = set(constrainted_question.keys())
    except:
        pass

    passed_args = []
    for query, constraint_list in constraints.items():
        if query in dedup_set:
            continue
        transformed_constraint_list = transform_constraints(constraint_list)
        passed_args.append((query, transformed_constraint_list, prompt % (transformed_constraint_list, query)))

    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, tmp_save = future
                constrainted_question[query] = tmp_save
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(constrainted_question, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")

    # save
    with open(output_file, "w") as f:
        json.dump(constrainted_question, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=4)
    args = parser.parse_args()
    main(args)