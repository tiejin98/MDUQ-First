# This script exists just to load models faster
import functools
import os
import torch.nn as nn

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)
from transformers import AutoModelForCausalLM, AutoConfig,LlamaForCausalLM,get_linear_schedule_with_warmup,GemmaForCausalLM,MistralForCausalLM,Phi3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from _settings import LLAMA_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device="cuda:0", torch_dtype=torch.float16):
    # Access = "XXXXXXX"
    if model_name.startswith('facebook/opt-'):
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")#, torch_dtype=torch_dtype)
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        #model = AutoModelForCausalLM.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch_dtype,token=Access)
    elif model_name == 'roberta-large-mnli':
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")#, torch_dtype=torch_dtype)
    elif model_name == "Phi3":
        model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",torch_dtype=torch.float16)
    elif model_name == "Phi4":
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", torch_dtype=torch.bfloat16,token=Access)
    elif model_name =="Deepseek":
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", torch_dtype=torch.bfloat16,token=Access)
    elif model_name == "Phi3-mini":
        model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct",torch_dtype=torch.float16)
    elif model_name == "Phi3-small":
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-small-128k-instruct",torch_dtype=torch.float16,trust_remote_code=True)
    elif model_name == "Phi3-medium":
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-medium-128k-instruct",torch_dtype=torch.float16,trust_remote_code=True)
    elif model_name == "Mistral":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",torch_dtype=torch.float16,token=Access)
    elif model_name == "Gemma":
        model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it",torch_dtype=torch.float16,token=Access)
    else:
        #model = AutoModelForCausalLM.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", torch_dtype=torch_dtype, token=Access)
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    Access = "xxxxx"
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token=Access)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "Phi3":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    elif model_name == "Phi4":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
    elif model_name == "Deepseek":
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    elif model_name == "Phi3-mini":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    elif model_name == "Phi3-small":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-small-128k-instruct",trust_remote_code=True)
    elif model_name == "Phi3-medium":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-128k-instruct",trust_remote_code=True)
    elif model_name == "Mistral":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        if tokenizer.pad_token_id == None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "Gemma":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", use_fast=use_fast,token=Access)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
