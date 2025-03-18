import numpy as np
import scipy.io

path_file = "results_total_tucker.mat"
mat_data = scipy.io.loadmat(path_file)


recon_res_res = {}
recon_res_keys = mat_data['recon_res'][0]
recon_res_values = mat_data['recon_res'][1]
for i in range(len(recon_res_keys)):
    recon_res_res[str(recon_res_keys[i][0])] = 1 - recon_res_values[i][0]
np.save("recon_res_matlab_tucker.npy",recon_res_res)


path_file = "results_total_cp.mat"
mat_data = scipy.io.loadmat(path_file)


recon_res_res = {}
recon_res_keys = mat_data['recon_res'][0]
recon_res_values = mat_data['recon_res'][1]
for i in range(len(recon_res_keys)):
    recon_res_res[str(recon_res_keys[i][0])] = 1 - recon_res_values[i][0]
np.save("recon_res_matlab_cp.npy",recon_res_res)








