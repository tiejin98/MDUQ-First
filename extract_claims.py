import _settings
from dataeval.load import read_cleaned_outputs_new
import json

from tqdm import tqdm
import ollama
import pickle as pkl

def queryFromWebLlama(prompt):

    import requests
    # URL from the provided image
    url = "https://www.llama2.ai/api"
    ip = '10.181.85.195'
    port = '10809'
    proxy = {'http': f'http://{ip}:{port}', 'https': f'http://{ip}:{port}'}

    payload = {
        "prompt": prompt,
        "model": "meta/llama-2-70b-chat",
        "systemPrompt": "You are a helpful assistant.",
        "temperature": 0.75,
        "topP": 0.9,
        "maxTokens": 8000,
        "image": None,
        "audio": None
    }

    # Headers from the provided image inside the red box
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PostmanRuntime/7.32.3",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    # Send the request
    response = requests.post(url, json=payload, headers=headers, proxies=proxy)

    # Print the response
    content = response.text
    # with open('./ans.txt', 'w', encoding='utf-8') as f:
    #     f.write(content)
    return content

def queryFromOllama(prompt):
    response = ollama.chat(model='qwen:32b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ],options={'main_gpu':0,})
    return response['message']['content']

def extract_claims(claims_file_path, qa_file, model_name='ollama'):
    if model_name == 'ollama':
        query = queryFromOllama
    else:
        query = queryFromWebLlama

    for input_text in tqdm(qa_file):
        repeat = 0
        while True:
            prompt = f'<<INST>><<SYS>>\nYou are given a piece of text that includes knowledge claims. A claim is a statement that asserts something as true or false, which can be verified by humans. [Task] Your task is to accurately identify and extract every claim stated in the provided text. Then, resolve any coreference (pronouns or other referring expressions) in the claim for clarity. Each claim should be concise (less than 15 words) and self-contained. Your response MUST be a list of dictionaries. Each dictionary should contains the key "claim", which correspond to the extracted claim (with all corefer-ences resolved). You MUST only respond in the for-mat as described below. DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTESTHATVIOLATETHERESPONSEFORMAT IS BANNED. START YOUR RESPONSE WITH ’[’. [Response Format] [{{"claim": "Ensure that the claim is fewer than 15 words and conveys a complete idea. Resolve any coref-erence (pronouns or other referring expressions) in the claim for clarity." }},... ] Here are two examples: [text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Sat-urday. The sixth-seed reaches Monte Carlo Masters final for the first time . Berdych will face either Rafael Nadal or Novak Djokovic in the final. [response]: [{{"claim": "Tomas Berdych defeated Gael Mon-fis 6-1, 6-4"}}, {{"claim": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday"}}, {{"claim": "Tomas Berdych reaches Monte Carlo Masters fi-nal"}}, {{"claim": "Tomas Berdych is the sixth-seed"}}, {{"claim": "Tomas Berdych reaches Monte Carlo Mas-ters final for the first time"}}, {{"claim": "Berdych will face either Rafael Nadal or Novak Djokovic"}}, {{"claim": "Berdych will face either Rafael Nadal or Novak Djokovic in the final"}}] [text]: Tinder only displays the last 34 photos - but users can easily see more. Firm also said it had improved its mutual friends feature. [response]: [{{"claim": "Tinder only displays the last photos"}}, {{"claim": "Tinder only displays the last 34 photos"}}, {{"claim": "Tinder users can easily see more photos"}}, {{"claim": "Tinder said it had improved its feature"}}, {{"claim": "Tinder said it had improved its mutual friends feature"}}] Now complete the following: [text]: {input_text} [response]: [/INST]'
            claims = query(prompt)
            try:
                json.loads(claims)
                claims = claims.replace('\n','')
                with open(claims_file_path, 'a') as f:
                    f.write(claims)
                    f.write('\n')
                repeat = 0
                break
            except:
                # print(f'pass, the claim is {claims}')
                repeat += 1
                if repeat > 5:
                    with open(claims_file_path, 'a') as f:
                        f.write('\n')
                    repeat = 0
                    break

def answer_true_false(claims_file_path, qa_file, model_name='ollama'):
    if model_name == 'ollama':
        query = queryFromOllama
    else:
        query = queryFromWebLlama
    
    few_shots = {
            'triviaqa': [{'question': 'In Scotland a bothy/bothie is a?',
                        'reference': 'House','answer': 'House',
                            },
                        {'question': 'Where in England was Dame Judi Dench born?',
                        'reference':  'York', 'answer': 'London'
                            }],
            'nq_open': [{'question': 'who makes up the state council in russia',
                            'reference': 'governors and presidents', 'answer': 'governors and presidents'
                            },
                            {'question': 'when does real time with bill maher come back',
                                'reference': 'November 9, 2018', 'answer': 'September 8, 2000'
                            }],
            'coqa': [{'question': 'When was the Vat formally opened?',
                    'reference': 'It was formally established in 1475', 'answer': 'In 1475',
                    },
                    {'question': 'what is the library for?',
                    'reference': 'research', 'answer': 'tourism'}]

        }['coqa']

    for row in tqdm(qa_file):
        for pred in row['generations']:
            repeat = 0
            while True:
                prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100.
Question: {few_shots[0]['question']}
Reference: {few_shots[0]['reference']}
Answer: {few_shots[0]['answer']}
Rating: 100.

Question: {few_shots[1]['question']}
Reference: {few_shots[1]['reference']}
Answer: {few_shots[1]['answer']}
Rating: 0.

Question: {row['question']}
Reference: {row['answer']}
Answer: {pred.strip()}
Rating:"""
                level = query(prompt)
                try:
                    float(level)
                    with open(claims_file_path, 'a') as f:
                        f.write(level)
                        f.write('\n')
                    repeat = 0
                    break
                except:
                    # print(f'pass, the claim is {claims}')
                    repeat += 1
                    if repeat > 5:
                        with open(claims_file_path, 'a') as f:
                            f.write('\n')
                        repeat = 0
                        break

def is_unwanted_answer(a):
    unwanted_answers = {"", " ", ".", "{}", "[]", "()", "null", "None", "undefined"}
    return a.strip() in unwanted_answers

def extract_combined_claims(claims_file_path, qa_file, model_name='ollama'):
    
    
    previous_q = None

    if model_name == 'ollama':
        query = queryFromOllama
    else:
        query = queryFromWebLlama

    for q, a in tqdm(qa_file):

        if previous_q is not None and q != previous_q:
            with open(claims_file_path, 'a') as f:
                f.write("----------\n")
            
        previous_q = q

        if is_unwanted_answer(a):
            a = a.replace('\n','')
            with open(claims_file_path, 'a') as f:
                # f.write(a)
                f.write("None")
                f.write('\n')
                continue
            
        repeat = 0
        while True:
            prompt = f'<<INST>><<SYS>>\nYou are given a piece of text that includes knowledge claims. A claim is a statement that asserts something as true or false, which can be verified by humans. [Task] Your task is to accurately identify and extract every claim stated in the provided text. Then, resolve any coreference (pronouns or other referring expressions) in the claim for clarity. Each claim should be concise (less than 15 words) and self-contained. Your response MUST be a list of dictionaries. Each dictionary should contains the key "claim", which correspond to the extracted claim (with all corefer-ences resolved). You MUST only respond in the for-mat as described below. DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA RESPONSE FORMAT IS BANNED INCLUDING EXPLANATION. START YOUR RESPONSE WITH ’[’. [Response Format] [{{"claim": "Ensure that the claim is fewer than 15 words and conveys a complete idea. Resolve any coref-erence (pronouns or other referring expressions) in the claim for clarity." }},... ] Here are two examples: [text]: Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Sat-urday. The sixth-seed reaches Monte Carlo Masters final for the first time . Berdych will face either Rafael Nadal or Novak Djokovic in the final. [response]: [{{"claim": "Tomas Berdych defeated Gael Mon-fis 6-1, 6-4"}}, {{"claim": "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday"}}, {{"claim": "Tomas Berdych reaches Monte Carlo Masters fi-nal"}}, {{"claim": "Tomas Berdych is the sixth-seed"}}, {{"claim": "Tomas Berdych reaches Monte Carlo Mas-ters final for the first time"}}, {{"claim": "Berdych will face either Rafael Nadal or Novak Djokovic"}}, {{"claim": "Berdych will face either Rafael Nadal or Novak Djokovic in the final"}}] [text]: Tinder only displays the last 34 photos - but users can easily see more. Firm also said it had improved its mutual friends feature. [response]: [{{"claim": "Tinder only displays the last photos"}}, {{"claim": "Tinder only displays the last 34 photos"}}, {{"claim": "Tinder users can easily see more photos"}}, {{"claim": "Tinder said it had improved its feature"}}, {{"claim": "Tinder said it had improved its mutual friends feature"}}] Now complete the following: [text]: Question:{q} Answer:{a} [response]: [/INST]'
            claims = query(prompt)
            try:
                claims = json.loads(claims)
            except:
                claims.split()
            # claims = claims.replace('\n','')
            with open(claims_file_path, 'a') as f:
                for c in claims:
                    f.write(c['claim'])
                    f.write('.')
                f.write('\n')
            repeat = 0
            break
 

if __name__ == '__main__':
    # Change to the real
    path = "0.pkl"
    
    with open(path, 'rb') as f:
        obj = pkl.load(f)

    data = []
    for ele in obj:
        print(ele)
        q = ele['question']
        for a in ele['generations']:
            data.append([q,a])
        break

    extract_combined_claims('0_claim.txt', data, model_name='ollama')


    # [1] 201516
