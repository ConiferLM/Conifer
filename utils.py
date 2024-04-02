import json

def get_prompt(stage):
    stage = stage.replace(' ', '_')
    with open(f"prompts/{stage}.txt", 'r') as f:
        return f.read()

def get_level_x_pairs(input, level, output=None):
    samples = json.load(open(input, "r"))
    pairs = []
    for _, item in samples.items():
        diff = int(item['diff'].split(' ')[-1])
        question = item['question']
        answer = item['answer']
        if diff == level:
            pairs.append({'instruction': question, 'response': answer, 'level': diff})
    if output is not None:
        with open(output, "w") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=4)
    return pairs

def answer_to_pairs(input, output):
    samples = json.load(open(input, "r"))
    pairs = []
    for _, item in samples.items():
        diff = item['diff']
        question = item['question']
        answer = item['answer']
        pairs.append({'instruction': question, 'response': answer, 'level': diff})
    with open(output, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)

# get easy-to-hard progression data
def get_multi_turn(input, output):
    samples = json.load(open(input, "r"))
    messages = []
    temp = {}
    for _, sample in samples.items():
        query = sample['query']
        diff = int(sample['diff'].split(' ')[-1])
        question = sample['question']
        answer = sample['answer']
        if query not in temp:
            temp[query] = {diff: (question, answer)}
        else:
            temp[query][diff] = (question, answer)
    
    for _, item in temp.items():
        message = []
        item = dict(sorted(item.items()))
        for diff, (question, answer) in item.items():
            message.append({'role': 'user', 'content': question})
            message.append({'role': 'assistant', 'content': answer})
    
        messages.append({'messages':message})
    
    with open(output, "w") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    return messages

# get internal process feedback data
def get_internal(input, output):
    samples = json.load(open(input, "r"))
    messages = []
    for _, sample in samples.items():
        question = sample['question']
        answer = sample['answer']
        messages.append({'messages':[
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ]})
    with open(output, "w") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    return messages

# get external process feedback data
def get_external(input, output):
    samples = json.load(open(input, "r"))
    messages = []
    for _, sample in samples.items():
        query = sample['query']
        messages.append({'messages':[
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': sample['answer']}
        ]})
    with open(output, "w") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    return messages