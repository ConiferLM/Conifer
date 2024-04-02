import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from utils import get_prompt
from openai import OpenAI

MAX_API_RETRY = 5
API_KEY = os.environ["OPENAI_API_KEY"]

def get_response(query, level, constrainted_question, prompt):
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
            answer = json.loads(content)['rewritten question']
        except Exception as e:
            print(f"json failed to parse: {e}")
            print(f"content: {content}")
            return None
        return query, level, {'constrainted_original_questions': {level: constrainted_question}, 'constrainted_questions': {level: answer}}

def main(args):
    # load input queries
    input_file = os.path.join(args.dir, "04_constrainted_question.json")
    output_file = os.path.join(args.dir, "04_constrainted_question_format_number.json")

    constrainted_question = json.load(open(input_file, "r"))
    constrainted_question_fn = {}

    # randomly select 1000 samples from constrainted questions
    seed_list = []
    for q, item in constrainted_question.items():
        for level, rq in item['constrainted_questions'].items():
            seed_list.append((q, level, rq))
    import random
    random.seed(42)
    random.shuffle(seed_list)
    seed_list = seed_list[:1000]

    # load prompt
    prompt = get_prompt("fn recombination")

    # load constraints list
    format_constraint_list = [
        "To create a blockquote, begin a paragraph with >> followed by a space.\n",
        "For strikethrough text, wrap the text with ~~ on each side.\n",
        "Indicate a block of code by wrapping the text with backticks ` on each side for inline code or triple backticks ``` for a code block.\n",
        "Enclose a spoiler or hidden text with || on each side to allow readers to choose whether to view it.\n",
        "For a highlighted annotation, use %% around the text.\n",
        "To create a drop cap at the beginning of a paragraph, use << followed by the letter and >>.\n",
        "Use <<< and >>> on separate lines above and below a paragraph to center it.\n",
        "Indicate an action by prefacing the sentence with an asterisk *.\n",
        "To create a separator or thematic break, use a line of asterisks ****** on a separate line.\n",
        "For a title or heading, start the line with three or more equals signs === at both the beginning and the end of the line.\n",
        "Surround text that should be in a different font with <<>> at the beginning and <<<\font name>>> at the end.\n",
        "Start each list item with > followed by a space.\n",
        "Enclose the word or phrase to be emphasized with ** on each side.\n",
        "Highlight the significant word or phrase by wrapping it with __ on each side for underlining.\n",
        "Mark the focus word or phrase with // on either side to indicate italics.\n",
        "Frame actions or sound effects with brackets [ ] on each side.\n",
        "Surround actions or sound effects with parentheses ( ) on each side.\n",
        "Start a sentence with ##.\n",
        "Use -- before and after reponse.\n",
        "Surround a list header with double underscores __ __ for strong emphasis.\n",
        "Use ** ** around a list header to indicate bold text formatting.\n"
    ]
    numerical_constraint_list = [
        "Answer with [5, 10, 15] sentences, using ##Sen as the beginning of each sentence.\n",
        "Answer with [1, 3, 5] paragraphs.\n",
        "Answer within [100, 150, 200, 250, 300] words.\n",
        "Answer with at leat [100, 150, 200, 300] words.\n",
        "Limit each answer to no more than [10, 20, 30] characters per word.\n",
        "Provide an answer consisting of exactly [2, 3, 4] sentences per paragraph, with a minimum of [1, 2] paragraphs.\n",
        "Begin each answer with a rhetorical question, followed by [2, 3, 4] explanatory sentences.\n",
        "Answer using only simple sentences, with each sentence containing [5, 7, 10] words.\n",
        "Construct the answer as a dialogue between two characters with [4, 6, 8] exchanges (back and forth).\n",
        "Answer with an introductory sentence, a [3, 4, 5]-item list, and a concluding sentence.\n",
        "Provide an answer where the first word of each sentence starts with sequential letters of the alphabet (A, B, C, etc.), with [3, 5, 7] sentences total.\n",
        "Respond with a poem of [4, 6, 8] lines, where each line is [4, 6, 8] words long.\n",
    ]

    ratio = args.ratio
    sample_k = args.sample_k

    # load saved samples
    dedup_set = set()
    try:
        with open(output_file, "r") as f:
            constrainted_question_fn = json.load(f)
            for k, v in constrainted_question_fn.items():
                for level, _ in v['constrainted_original_questions'].items():
                    dedup_set.add(k + ' ' + level)
    except Exception as e:
        pass

    passed_args = []
    for query, level, constrainted_question in seed_list:
        if query + ' ' + level in dedup_set:
            continue
        random.shuffle(format_constraint_list)
        random.shuffle(numerical_constraint_list)
        constraint_list = format_constraint_list[:int(sample_k * ratio)] + numerical_constraint_list[:int(sample_k * (1 - ratio))]
        
        transformed_constraint_list = ''.join(constraint_list[:8])
        passed_args.append((query, level, constrainted_question, prompt % (constrainted_question, transformed_constraint_list)))

    with ThreadPoolExecutor(max_workers=args.worker) as executor:
        for save_count, future in enumerate(tqdm(executor.map(get_response, *zip(*passed_args)), total=len(passed_args))):
            if future is not None:
                query, level, tmp_save = future
                if query in constrainted_question_fn:
                    constrainted_question_fn[query]['constrainted_original_questions'][level] = tmp_save['constrainted_original_questions'][level]
                    constrainted_question_fn[query]['constrainted_questions'][level] = tmp_save['constrainted_questions'][level]
                else:
                    constrainted_question_fn[query] = tmp_save
                if save_count % args.save_iterval == 0:
                    with open(output_file, "w") as f:
                        json.dump(constrainted_question_fn, f, ensure_ascii=False, indent=4)
                    print(f"File Saved")

    # save
    with open(output_file, "w") as f:
        json.dump(constrainted_question_fn, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="input file path for sentences", default="conifer_data")
    parser.add_argument("--save-iterval", type=int, help="save to file after generating K samples", default=2)
    parser.add_argument("--worker", type=int, help="number of concurrent workers", default=1)
    parser.add_argument("--ratio", type=float, help="ratio of format constraint, larger ratio means more format constraints and less numerical constriants", default=0.6)
    parser.add_argument("--sample-k", type=int, help="total number of format and numerical constraints in one sample", default=6)
    args = parser.parse_args()
    main(args)