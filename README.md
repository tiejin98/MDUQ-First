# MDUQ-First
This repo uses python to generate and process the data and uses matlab to run tensor decomposition. 
<details>
  <summary>Todo</summary>
  - Delete many repeated code, might cause bug.
  
  - Improve code quality.
</details>

## Simple Requirement
Python: Transformers, Pytorch, vllm.
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

## Accuracy Evaluation of Generation

Run ```correctness_qwen.py``` to get ```accuracy.pkl```, which is used to evaluate the uncetainty in the end. In the ```correctness_qwen.py```, users are also required to change the path of 0.pkl to the real path to semantic generation.

## Getting Similarity Matrix
First run ```sim_mat.py```  and then run ```transfer_matrix.py``` to get file for Matleb.

## Tensor Decomposition, Uncertainty Measures and Evaluation
Run ```cp_decom.m``` and ```tucker_decom.m``` to get the results of tensor decomposition with Matlab. Then, running ```get_matlab_res.py``` and ```inter_unq_ensemble.py``` to get the final uncertianty measures. Note that sometimes, you may need to use Matlab to transfer the ```cp_decom.m``` and ```tucker_decom.m``` to version7 of .mat. Finally, evaluate the uncertainty using ```evaluation.py```

