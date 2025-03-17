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
      "stem": "In rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.\n The statements above, if true, most strongly support which one of the following conclusions?",
      "choices": [
        {"label": "A", "text": "Unlike aspirin and other medications that reduce pain and swelling and that are currently available, the new medication would repair existing cell damage that had been caused by rheumatoid arthritis."},
        {"label": "B", "text": "A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it."},
        {"label": "C", "text": "Joint diseases other than rheumatoid arthritis would not be affected by the new medication."},
        {"label": "D", "text": "The benefits to rheumatoid arthritis sufferers of the new medication would outweigh the medication's possible harmful side effects."},
      ]
    },
    "answerKey": "B"
  },
  {
    "question": {
      "stem": "The use of space-based satellites to study environmental conditions on Earth is an important development in the conservation movement' s history. Environmental problems may now be observed long before they otherwise would be noticed, allowing for intervention before they reach the crisis stage. It is no wonder that environmentalists fail to consider both that spacecraft may damage the ozone layer and that this damage could be serious enough to warrant discontinuing spaceflight.\n The reasoning above most closely conforms to which one of the following principles?",
      "choices": [
        {"label": "A", "text": "People tend to ignore possible objectionable consequences of actions that support their activities."},
        {"label": "B", "text": "Attempts to employ technology often have unforeseen consequences that may be negative."},
        {"label": "C", "text": "Technology usually has at least some negative impact on the environment, even if it is largely beneficial."},
        {"label": "D", "text": "A negative consequence of an activity may be outweighed by its great positive consequences."},
      ]
    },
    "answerKey": "A"
  },
]

num_choice_dict = {"0":"A","1":"B","2":"C","3":"D"}


def sample_to_prompt(sample,tokenizer, **kwargs):
    question_stem = f"{sample['context']}\n{sample['question']}"
    # choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in sample['question']['choices']])
    answer_list = []
    for i, option in enumerate(sample['answers']):
        answer_list.append(f"({num_choice_dict[str(i)]}) {option}")
    choices = "\n".join(answer_list)
    # Add introductory prompt
    prompt = """Answer the following multi-choice question with one of the options and output the option only.

Here is an example:
"""

    for i, example in enumerate(training_sample[:1]):
        example_question_stem = example['question']['stem']
        example_choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in example['question']['choices']])
        example_answer = example['answerKey']  # Assuming each example contains an answerKey
        prompt += f"Example {i+1}:\nQ: {example_question_stem}\nChoices:\n{example_choices}\nAns: ({example_answer})\n\n"

    # Add the actual question to be answered
    prompt += f"Now, answer this question with context:\nQ: {question_stem}\nChoices:\n{choices}\nAns: ("
    print(prompt)
    return prompt
#
# def sample_to_prompt(sample, **kwargs):
#     question_stem = sample['question']['stem']
#     choices = "\n".join([f"({choice['label']}) {choice['text']}" for choice in sample['question']['choices']])
#     prompt = f"""Answer the following multi-choice question with one of the options and output the option only. You must output one option:
# Q: {question_stem}
# Choices:
# {choices}
# A: ("""
#     return prompt


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
    batch['id'] = sample['id_string']
    batch['prompt'] = prompt
    batch['answer'] = answers

    # Ensure 'question' is included in the batch to match script expectations
    batch['question'] = [sample['question']]

    return batch


@functools.lru_cache()
def get_dataset(tokenizer):
    file_path = "/home/local/ASURITE/tchen169/UQ-NLG/reclor/reclor_data/val.json"

    with open(file_path, 'r') as file:
        data_list = [json.loads(line) for line in file]
    id_mem = set()
    data_list = data_list[0]
    # data_list = data_list[13:513]
    def remove_dups(sample):
        if sample['id_string'] in id_mem:
            return None  # Filter out duplicates
        id_mem.add(sample['id_string'])
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
