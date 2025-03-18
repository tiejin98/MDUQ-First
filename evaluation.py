import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc

def compute_modified_auarc(y_true, uncertainty_scores):
    # Sort by uncertainty, highest first
    indices = np.argsort(uncertainty_scores)
    sorted_y_true = y_true[indices]


    cumulative_accuracy = np.cumsum(sorted_y_true) / np.arange(1, len(y_true) + 1)
    rejection_fraction = np.arange(1, len(y_true) + 1) / len(y_true)

    # Compute AUARC as the area under the curve, using the trapezoidal rule
    auarc = np.trapz(cumulative_accuracy[::-1], rejection_fraction)

    return auarc, cumulative_accuracy[::-1], rejection_fraction




accuracy = np.load("acc.npy",allow_pickle=True).item()
cca_res = np.load("unq_matlab_ensemble_min.npy",allow_pickle=True).item()

cca_res_seman = []
res_acc = []
total = 0
for id in cca_res.keys():
    total += 1
    cca_res_seman.append(np.real(cca_res[id]))
    res_acc.append(accuracy[id])
cca_res_seman = np.array(cca_res_seman)
res_acc = np.array(res_acc)
cca_res_seman = np.nan_to_num(cca_res_seman, nan=0.0)
final_acc = []
seman_cca = []
test = []
for i in range(len(cca_res_seman)):
    y_true = res_acc[i]
    y_predict = cca_res_seman[i]
    true_number = int(y_true*20)
    for i in range(20):
        if i < true_number:
            final_acc.append(1)
        else:
            final_acc.append(0)
        # If evaluating the confidence, change 1-y_predict to y_predict only.
        seman_cca.append(1-y_predict)
        test.append(y_true)

fpr, tpr, thresholds = roc_curve(final_acc,seman_cca,drop_intermediate=False)
auarc, cumulative_accuracy, rejection_fraction = compute_modified_auarc(res_acc,cca_res_seman)

print("AUC:", auc(fpr,tpr))
print(f"AUARC: {auarc}")
