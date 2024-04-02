#!/bin/bash
set -eo pipefail

export OPENAI_API_KEY="your openai api key"

# the default output dir is ./conifer_data
# if you want to use a different output dir, change the --dir argument

# you can change the --save-interval to control how often the results is saved, default is 2
# also, the --worker controls the number of concurrent requests to openai, default is 4
python 01_question_reframing.py
python 02_first_stage_filter.py
python 03_constraints_generation.py
python 04a_recombination.py
python 04b_recombination_format_number.py
python 05_second_stage_filter.py
python 06a_answer_generate.py
python 06b_internal_answer_generate.py

# to get the external feedback, you should first run 07a_external_inference.py to get a model's output on the difficulty 5 questions

# produce the external process feedback samples
# python 07a_external_inference.py --model your_model_name(default is Mistral-7B-Instruct-v0.1)
# python 07b_external_feedback.py

# by running all the above, you should have the final output in the ./conifer_data folder. You could call utils.get_all(input_dir, output_path) to get all the results in a single file, with chatml format.
