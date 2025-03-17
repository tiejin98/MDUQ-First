import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset

import _settings  # Where you define DATA_FOLDER, etc.



def _save_dataset_hotpot():
    """
    Loads the HotpotQA dev file (hotpot_dev_distractor_v1.json),
    combines contexts, and saves an HF Dataset with:
        story, question, answer, id
    """
    save_path = f"{_settings.DATA_FOLDER}/hotpot_dataset"
    if not os.path.exists(save_path):
        # 1. Load data from JSON
        with open(f"/home/local/ASURITE/tchen169/UQ-NLG/HotpotQA/hotpot_dev_distractor_v1.json", "r") as infile:
            data = json.load(infile)  # list of samples

        # 2. Prepare the dict for storing
        dataset_dict = {
            "story": [],
            "question": [],
            "answer": [],
            "id": [],
        }

        # 3. Iterate over samples
        for sample in data:
            # sample keys: '_id', 'answer', 'question', 'supporting_facts', 'context', ...
            # We ignore 'supporting_facts' as requested.

            # Combine context lines into one big string:
            context = sample["context"]  # e.g. [["Title", [sent1, sent2, ...]], ["Title2", [sent1, ...]]]
            context_parts = []
            for c in context:
                title = c[0]
                paragraphs = c[1]
                # Optionally keep the title
                context_parts.append(title)
                context_parts.extend(paragraphs)
            story = " ".join(context_parts)

            question = sample["question"]
            answer_text = sample["answer"]
            _id = sample["_id"]  # Unique ID

            # Fill the dataset dict
            dataset_dict["story"].append(story)
            dataset_dict["question"].append(question)
            dataset_dict["answer"].append({"text": answer_text})
            dataset_dict["id"].append(_id)

        # 4. Convert to DataFrame -> Dataset -> save
        df = pd.DataFrame.from_dict(dataset_dict)
        ds = Dataset.from_pandas(df)
        ds.save_to_disk(save_path)

    return save_path

@functools.lru_cache(1)
def read_all_contexts_hotpot():
    """
    Loads the saved dataset from disk (Hotpot dev) and returns a
    dict mapping ID -> story text. Similar to read_all_contexts() in CoQA code.
    """
    ds = datasets.load_from_disk(_save_dataset_hotpot())
    return {row["id"]: row["story"] for row in ds}


def get_dataset(tokenizer, split="validation"):
    """
    Equivalent to get_dataset() in CoQA code.
    1. Loads dataset from disk
    2. Creates a prompt: story + " Q: " + question + " A:"
    3. Tokenizes into input_ids, attention_mask
    4. Returns the dataset in PyTorch format
    """
    # 1. Load from disk
    ds = datasets.load_from_disk(_save_dataset_hotpot())

    # 2. Encode function
    def encode_hotpot(example):
        # Convert answer from dict {"text": ...} into plain string
        example["answer"] = example["answer"]["text"]
        example["prompt"] = example["story"] + " Q: " + example["question"] + " A:"
        prompt = example["prompt"]
        # 3. Tokenize
        if hasattr(tokenizer, 'apply_chat_template'):
            chat = [
                {"role": "user", "content": example["prompt"]}
                  ]
            prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False
            )
        tokenized = tokenizer(
            prompt,
            truncation=False,
            padding=False,  # or as needed
        )
        return tokenized

    # 4. Map the encode function
    ds = ds.map(encode_hotpot, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    return ds



def _generate_config(tokenizer):
    """
    Similar to your CoQA _generate_config:
    - Figures out eos_token_id and bad_words_ids for generation
    """
    if tokenizer.__class__.__name__ == "LlamaTokenizer":
        # Example: LLaMA uses 29889 for '.' in some merges
        eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [29889]
    elif tokenizer.__class__.__name__ == "GPT2Tokenizer":
        # GPT2 usually tokenizes with a leading space ID, so index [1] might be '.' or '\n'
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['.', '\n']]
    else:
        # Fallback for other tokenizers (e.g., T5, BERT-like)
        if tokenizer.eos_token_id is not None:
            eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']] + [tokenizer.eos_token_id]
        else:
            eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]

    # Example of "bad words" or special tokens to avoid generating
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [[tokenizer(x)['input_ids'][1]] for x in question_framing_ids]

    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)


# Sample usage:
if __name__ == "__main__":
    # import models  # if you have a "models" module with load_tokenizer
    # tokenizer = models.load_tokenizer()
    # ds_hotpot = get_dataset_hotpot(tokenizer)
    # print(ds_hotpot[0])
    # config = _generate_config(tokenizer)
    # print(config)
    pass
