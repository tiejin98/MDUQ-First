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
      "stem": "Where do people usually sit waiting for a train?",
      "choices": [
        {"label": "A", "text": "waiting room"},
        {"label": "B", "text": "garage"},
        {"label": "C", "text": "kitchen"},
        {"label": "D", "text": "train station"},
        {"label": "E", "text": "bus stop"}
      ]
    },
    "answerKey": "A"
  },
  {
    "question": {
      "stem": "What is an ingredient in a cake?",
      "choices": [
        {"label": "A", "text": "flour"},
        {"label": "B", "text": "apple"},
        {"label": "C", "text": "potato"},
        {"label": "D", "text": "lettuce"},
        {"label": "E", "text": "corn"}
      ]
    },
    "answerKey": "A"
  },
  {
    "question": {
      "stem": "What is used to clean teeth?",
      "choices": [
        {"label": "A", "text": "hammer"},
        {"label": "B", "text": "brush"},
        {"label": "C", "text": "vacuum"},
        {"label": "D", "text": "sponge"},
        {"label": "E", "text": "cloth"}
      ]
    },
    "answerKey": "B"
  },
  {
    "question": {
      "stem": "Where would you find books to read?",
      "choices": [
        {"label": "A", "text": "library"},
        {"label": "B", "text": "kitchen"},
        {"label": "C", "text": "bathroom"},
        {"label": "D", "text": "bedroom"},
        {"label": "E", "text": "closet"}
      ]
    },
    "answerKey": "A"
  },
  {
    "question": {
      "stem": "Where can you find desks and students?",
      "choices": [
        {"label": "A", "text": "classroom"},
        {"label": "B", "text": "restaurant"},
        {"label": "C", "text": "gym"},
        {"label": "D", "text": "park"},
        {"label": "E", "text": "garden"}
      ]
    },
    "answerKey": "A"
  },
  {
    "question": {
      "stem": "What can be used to eat soup?",
      "choices": [
        {"label": "A", "text": "fork"},
        {"label": "B", "text": "knife"},
        {"label": "C", "text": "spoon"},
        {"label": "D", "text": "chopsticks"},
        {"label": "E", "text": "plate"}
      ]
    },
    "answerKey": "C"
  },
  {
    "question": {
      "stem": "What is used to tell time?",
      "choices": [
        {"label": "A", "text": "phone"},
        {"label": "B", "text": "shoe"},
        {"label": "C", "text": "clock"},
        {"label": "D", "text": "book"},
        {"label": "E", "text": "cup"}
      ]
    },
    "answerKey": "C"
  }
]

#
# def sample_to_prompt(sample, examples, **kwargs):
#     question_stem = sample['question']['stem']
#     choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in sample['question']['choices']])
#
#     # Add introductory prompt
#     prompt = """Answer the following multi-choice question with one of the options and output the option only. You must output one option:
#
# Here are some examples:
# """
#
#     # Add examples from CommonsenseQA training set
#     for i, example in enumerate(examples[:3]):  # Include only the first 7 examples
#         example_question_stem = example['question']['stem']
#         example_choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in example['question']['choices']])
#         example_answer = example['answerKey']  # Assuming each example contains an answerKey
#         prompt += f"Example {i+1}:\nQ: {example_question_stem}\nChoices:\n{example_choices}\nA: ({example_answer})\n\n"
#
#     # Add the actual question to be answered
#     prompt += f"Now, answer this question:\nQ: {question_stem}\nChoices:\n{choices}\nA: ("
#
#     return prompt
#
def sample_to_prompt(sample, **kwargs):
    question_stem = sample['question']['stem']
    choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in sample['question']['choices']])
    prompt = f"""Answer the following multi-choice question with one of the options and output the option only. You must output one option:
Q: {question_stem}
Choices:
{choices}
A: ("""
    return prompt


def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ == 'LlamaTokenizer':
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.','A','B','C','D','E']]
        eos_token_id.extend([tokenizer.convert_tokens_to_ids(_) for _ in ['A','B','C','D','E']])
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    else:
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n', ',', '.','A','B','C','D','E']]
        eos_token_id.extend([tokenizer.convert_tokens_to_ids(_) for _ in ['A','B','C','D','E']])
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']]  # only "Q"
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)


def process_data_to_model_inputs(sample, tokenizer):
    # Extracting the correct answer
    correct_answer_label = sample['answerKey']
    answers = correct_answer_label[0]

    # Generating the prompt
    prompt = sample_to_prompt(sample)
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
    batch['id'] = sample['id']
    batch['prompt'] = prompt
    batch['answer'] = answers

    # Ensure 'question' is included in the batch to match script expectations
    batch['question'] = [sample['question']['stem']]

    return batch


@functools.lru_cache()
def get_dataset(tokenizer):
    file_path = "dev_rand_split.jsonl"

    with open(file_path, 'r') as file:
        data_list = [json.loads(line) for line in file]
    id_mem = set()

    def remove_dups(sample):
        if sample['id'] in id_mem:
            return None  # Filter out duplicates
        id_mem.add(sample['id'])
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
    #model, tokenizer = models.load_model_and_tokenizer('llama-7b-hf')
    data = get_dataset(tokenizer)
