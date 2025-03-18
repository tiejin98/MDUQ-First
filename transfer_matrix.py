import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle

def get_affinity_mat(logits, mode='disagreement', temp=None, symmetric=True):
    if mode == 'jaccard':
        return logits
    # can be weigheted
    if mode == 'disagreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) != 0
    if mode == 'disagreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 0]
        if symmetric:
            W = (W + W.permute(1,0))/2
        W = 1 - W
    if mode == 'agreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) == 2
    if mode == 'agreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 2]
        if symmetric:
            W = (W + W.permute(1,0))/2
    if mode == 'gal':
        W = logits.argmax(-1)
        _map = {i:i for i in range(len(W))}
        for i in range(len(W)):
            for j in range(i+1, len(W)):
                if min(W[i,j], W[j,i]) > 0:
                    _map[j] = _map[i]
        W = torch.zeros_like(W)
        for i in range(len(W)):
            W[i, _map[i]] = W[_map[i], i] = 1
    W = W.cpu().numpy()
    W[np.arange(len(W)), np.arange(len(W))] = 1
    W = W.astype(np.float32)
    return W

def get_D_mat(W):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    return D

def get_L_mat(W, symmetric=True):
    # compute the degreee matrix from the weighted adjacency matrix

    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError()
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
    return L.copy()

def get_laplacian(logits):
    W = get_affinity_mat(logits, mode="agreement_w", temp=1.0)
    L = get_L_mat(W, symmetric=True)
    return L

def compute_normalized_laplacian(similarity_matrix):
    # Apply softmax to the similarity matrix

    # Calculate the weighted adjacency matrix (W)
    W = (similarity_matrix + similarity_matrix.T) / 2

    # Calculate the degree matrix (D)
    D = np.diag(np.sum(W, axis=1))
    L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))

    return L.copy()


res = {}
seman = np.load("seman_simmat.npy",allow_pickle=True).item()

for key in tqdm(seman.keys()):
    sim_mat = c[key]['sim_mat_renew']
    sim_mat = torch.nn.functional.softmax(sim_mat,dim=2)
    sim_mat = sim_mat[:,:,2].numpy()
    for i in range(20):
        sim_mat[i,i] = 1
    res[key] = sim_mat

with open('seman_simmat.pkl', 'wb') as file:
    pickle.dump(res, file)


res = {}
seman = np.load("knowledge_simmat.npy",allow_pickle=True).item()

for key in tqdm(seman.keys()):
    sim_mat = c[key]['sim_mat_renew']
    sim_mat = torch.nn.functional.softmax(sim_mat,dim=2)
    sim_mat = sim_mat[:,:,2].numpy()
    for i in range(20):
        sim_mat[i,i] = 1
    res[key] = sim_mat

with open('knowledge_simmat.pkl', 'wb') as file:
    pickle.dump(res, file)

