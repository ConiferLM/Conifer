import json
import fasttext
import argparse
import os

def main(args):
    # load fasttext model
    model = fasttext.load_model(args.fasttext_model)

    # load seed queries
    seed_file = args.seed_path
    queries = []
    samples = json.loads(open(seed_file, "r").read())
    for sample in samples:
        q = sample["conversations"][0]
        if len(q) > 200:
            continue
        if len(q.split(' ')) < 5:
            continue
        predict = model.predict(q.split('\n')[0], k=1)[0][0]
        lang = predict.split('__label__')[1]
        if lang != 'en':
            continue 

        queries.append(q)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"./{args.output_dir}/selected_query.txt", 'w') as f:
        for q in queries:
            f.write(q + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-path", type=str, help="input file path for seed data")
    parser.add_argument("--fasttext-model", type=str, help="input file path for fasttext model", default="./lid.176.bin")
    parser.add_argument("--dir", type=str, help="output directory", default="conifer_data")
    args = parser.parse_args()

    main(args)