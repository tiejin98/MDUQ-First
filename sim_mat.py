import tqdm
import torch
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import models.nli as sc
from dataeval.load import read_cleaned_outputs_new

def sim_matrix(data, device, 
               judge_model='microsoft/deberta-large-mnli',
               text_key = 'text_cleaned'):
    sc_model = sc.ClassifyWrapper(judge_model, device=device)
    semantic_sims = {}
    for _ in tqdm.tqdm(data, desc="computing similarities"):
        _tres = sc_model.create_sim_mat_batched(_['question'], _['generations'][text_key])
        _tres['id'] = _['id']
        semantic_sims[_['id']] = _tres
    return semantic_sims


def recover_sim_mat(simmat):
    for id, sim in simmat.items():
        sim_mat = sim['sim_mat'].clone()
        sim_mat[torch.arange(sim_mat.shape[0]), torch.arange(sim_mat.shape[0]), :] = torch.tensor([-torch.inf, -torch.inf, 100])
        mapping = sim['mapping']
        # a len(ans) x len(ans) x 3 tensor
        ret = torch.zeros((len(mapping), len(mapping), 3))
        for i, ans_i in enumerate(mapping):
            for j, ans_j in enumerate(mapping):
                ret[i,j] = torch.tensor(sim_mat[mapping[i], mapping[j]])
        simmat[id]['sim_mat_renew']=ret
    return simmat


if __name__ == '__main__':
    seman_path = "0.pkl"
    data = read_cleaned_outputs_new(seman_path)
    matrix = sim_matrix(data,device="cuda:0")
    matrix = recover_sim_mat(matrix)
    np.save("seman_simmat.npy",matrix)
    know_path = "0_know.pkl"
    data = read_cleaned_outputs_new(seman_path)
    matrix = sim_matrix(data,device="cuda:0")
    matrix = recover_sim_mat(matrix)
    np.save("know_simmat.npy",matrix)
    

