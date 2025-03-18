import numpy as np


def compute_inter_unq(a):
    key_list = list(a.keys())
    res = {}
    for key in key_list:
        res[key] = np.sum(np.array(a[key]))
    return res

def compute_inter_unq_sum(a,b):
    key_list = list(a.keys())
    res = {}
    for key in key_list:
        res[key] = np.sum(np.array(a[key])) + np.sum(np.array(b[key]))
    return res

def compute_inter_unq_min(a,b):
    key_list = list(a.keys())
    res = {}
    for key in key_list:
        res[key] = np.min([np.sum(np.real(np.array(a[key]))),np.sum(np.real(np.array(b[key])))])
    return res


a = np.load("recon_res_matlab_tucker.npy",allow_pickle=True).item()
b = np.load("recon_res_matlab_cp.npy",allow_pickle=True).item()



inter_unq = compute_inter_unq_min(a,b)
np.save("unq_matlab_ensemble_min.npy",inter_unq)
