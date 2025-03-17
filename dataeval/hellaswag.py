import argparse
import functools
import os
import pathlib
import pickle
import json
# import config
import datasets
import ipdb
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _settings

training_sample = [
  {
    "question": {
      "stem": "People are dancing having fun at a party. A race starts and people are running. Cheerleaders are standing on the side of the road. people",
      "choices": [
        {"label": "A", "text": "are standing on a beach watching and celebrating.",},
        {"label": "B", "text": "are swimming in puddles."},
        {"label": "C", "text": "are dancing on the grass."},
        {"label": "D", "text": "play a game of volleyball on the side of the road."},
      ]
    },
    "answerKey": "C"
  },
  {
    "question": {
      "stem": "He eats pizza with a group of friends. He sprays his mouth with mouthwash. He blows his breath into the air. he",
      "choices": [
        {"label": "A", "text": "is laughing and enjoying himself as he hangs out with a group of friends."},
        {"label": "B", "text": "does several stunts on his skateboard."},
        {"label": "C", "text": "looks back at the camera, spraying talk and eating the pizza."},
        {"label": "D", "text": "sits and reads a book while driving."},
      ]
    },
    "answerKey": "A"
  },
]

num_choice_dict = {"0":"A","1":"B","2":"C","3":"D"}


def sample_to_prompt(sample,tokenizer, **kwargs):
    question_stem = f"{sample['ctx']}"
    # choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in sample['question']['choices']])
    answer_list = []
    for i, option in enumerate(sample['endings']):
        answer_list.append(f"({num_choice_dict[str(i)]}) {option}")
    choices = "\n".join(answer_list)
    # Add introductory prompt
    prompt = """Complete the sentence by answering the following multi-choice question with one of the options and output the option only.

Here is an example:
"""

    for i, example in enumerate(training_sample):
        example_question_stem = example['question']['stem']
        example_choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in example['question']['choices']])
        example_answer = example['answerKey']  # Assuming each example contains an answerKey
        prompt += f"Example {i+1}:\nQ: {example_question_stem}\nChoices:\n{example_choices}\nAns: ({example_answer})\n\n"

    # Add the actual question to be answered
    prompt += f"Now, answer this question:\nQ: {question_stem}\nChoices:\n{choices}\nAns: ("
    print(prompt)
    return prompt


def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.','A','B','C','D']]
        eos_token_id.extend([tokenizer.convert_tokens_to_ids(_) for _ in ['A','B','C','D']])
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    else:
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n', ',', '.','A','B','C','D']]
        eos_token_id.extend([tokenizer.convert_tokens_to_ids(_) for _ in ['A','B','C','D']])
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']]  # only "Q"
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)


def process_data_to_model_inputs(sample, tokenizer):
    # Extracting the correct answer
    correct_answer_label = num_choice_dict[str(sample['label'])]
    answers = correct_answer_label[0]
    # Generating the prompt
    prompt = sample_to_prompt(sample,tokenizer)
    # Tokenizing the inputs and labels
    inputs = tokenizer(prompt, padding=False, truncation=False)
    outputs = tokenizer.convert_tokens_to_ids(answers)
    if outputs is None:
        outputs = tokenizer.encode(answers)[0]
    # Preparing the batch dictionary
    batch = {}
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs
    # batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs
    #
    # # Adjusting the labels to ignore padding tokens
    # batch["labels"] = [
    #     [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    # ]
    batch['id'] = sample['ind']
    batch['prompt'] = prompt
    batch['answer'] = answers

    # Ensure 'question' is included in the batch to match script expectations
    batch['question'] = [sample['ctx']]

    return batch


@functools.lru_cache()
def get_dataset(tokenizer):
    file_path = "/home/local/ASURITE/tchen169/UQ-NLG/hellaswag_val.jsonl"

    with open(file_path, 'r') as file:
        data_list = [json.loads(line) for line in file]
    id_mem = set()
    # data_list = [data_list[0]]
    # data_list = data_list[:1000]
    def remove_dups(sample):
        if sample['ind'] in id_mem:
            return None  # Filter out duplicates
        id_mem.add(sample['ind'])
        return sample

    # Removing duplicates
    unique_data = list(filter(None, map(remove_dups, data_list)))

    # Processing each sample
    processed_data = [process_data_to_model_inputs(sample, tokenizer) for sample in unique_data]

    # Converting to a Dataset format
    data = datasets.Dataset.from_list(processed_data)

    # Setting format for PyTorch
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"],
        output_all_columns=True
    )

    return data

if __name__ == '__main__':
    import pandas as pd

    import models

    tokenizer = models.load_tokenizer('llama-7b-hf')
    data = get_dataset(tokenizer)
