# MD-UQ
This repo uses python to generate and process the data and uses matlab to run tensor decomposition. <!-- Delete many repeated code, might cause bug. -->

## Simple Requirement
Python: Transformers, Pytorch.
Matlab: tensor_toolbox. 
Others: Ollama

## Setting
Before Running, please change the dir of ```_setting.py```.

## Semantic Generation

To generation semantic answers, first provide Huggingface Access Token in ```models/_load_model.py```. And then run ```generate.py``` by:

`python generate --model llama-13b-hf --dataset coqa --device cuda:0`. 

More parameters can be found in the code.

## Knowledge Generation

After semantic generation, it will output a ```0.pkl```, which contains all the semantic generations. After that, run ```extract_claims.py``` (ollama required), and the extracted claim will be saved to a ```0_claim.txt``` file. And then run ```getting_claim_pkl.py``` to get the pickle file of knowledge generation. Please change the path in the code.

## Tensor Decomposition

