import json
import pickle
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np

########################################
# 1) Initialize vllm with Qwen2.5-32B-Instruct
########################################
# Replace model path or name as needed for vllm
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct",gpu_memory_utilization=0.98)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

with open("0.pkl", "rb") as f:
    generation_data = pickle.load(f)





def build_prompt(prompt_text, original_ans, provided_ans):
    """
    We instruct the model to output only an integer from 0 to 100.
    """
    user_prompt = f"""
Given the following question, the original answer, and a provided answer, output an integer from 0 to 100 (no other text) indicating how correct the provided answer is.


Question:
{prompt_text}

Original Answer:
{original_ans}

Provided Answer:
{provided_ans}

correctness:
""".strip()
    return user_prompt


def parse_correctness_score(generated_text):
    """
    Find the first integer in the generated text, clamp it to 0..100.
    If none found, default to 0.
    """
    match = re.search(r"\d+", generated_text)
    if match:
        val = int(match.group(0))
        return max(0, min(100, val))
    return 0

def get_correctness_score_vllm(prompt_text, original_ans, provided_ans, max_new_tokens=10):
    """
    Use Qwen2.5-32B-Instruct with vllm to get a single correctness score.
    """
    # Build the user section of the prompt
    user_prompt = build_prompt(prompt_text, original_ans, provided_ans)

    # Combine system and user content into a single text prompt
    prompt_text = (
        "System: You are a helpful assistant. "
        "And you should only output numbers to present correctness.\n"
        f"User: {user_prompt}\n"
    )

    # Set up generation parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        top_p=1.0,
        n=1  # Generate only one response
    )

    # Generate using vllm
    outputs = llm.generate(prompt_text, sampling_params)

    # vllm returns a list of RequestOutput, and each RequestOutput has .outputs
    # which is a list of GenerationOutput. We take the first generation text.
    if len(outputs) > 0 and len(outputs[0].outputs) > 0:
        generated_text = outputs[0].outputs[0].text.strip()
        return parse_correctness_score(generated_text)
    else:
        return 0

results = {}

for item in tqdm(generation_data, desc="Processing samples"):
    # Decode the prompt text
    prompt_text = tokenizer.decode(item['prompt']) if "prompt" in item else ""
    # Ground truth
    original_answer = item['answer']
    # Provided answers
    generations = item['generations']
    # Get the index
    idx = item['id']

    sub_dict = {}
    for provided_answer in generations:
        score = get_correctness_score_vllm(prompt_text, original_answer, provided_answer)
        sub_dict[provided_answer] = score

    results[idx] = sub_dict


with open("correctness_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    


with open("correctness_results.json") as f:
    accuracy_data = json.load(f)

a = np.load("0.pkl",allow_pickle=True)

res_dict = {}
for sample in a:
    sample_acc_list = []
    id = sample["id"]
    generations = sample["generations"]
    accuracy_dict = accuracy_data[id]
    for gene in generations:
        try:
            if accuracy_dict[gene] > 90:
                sample_acc_list.append(1)
            else:
                sample_acc_list.append(0)
            # sample_acc_list.append(accuracy_dict[gene])
        except:
            sample_acc_list.append(0)
    res_accuacy = np.mean(sample_acc_list)
    res_dict[id] = res_accuacy
    
with open('accuracy.pkl',"wb") as f:
    pickle.dump(res_dict,f)

    



