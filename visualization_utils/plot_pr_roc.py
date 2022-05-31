import argparse
import os
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('eval_session_group_name', type=str)
parser.add_argument('visualizations_path', type=str)
args = parser.parse_args()

pr_roc_save_path = os.path.join(args.visualizations_path, args.eval_session_group_name +  "_pr_roc_points.pt")
pr_roc_data = torch.load(pr_roc_save_path)



for key in pr_roc_data:
    print(key)
    if "precisions_recalls" in key:
        precisions, recalls = pr_roc_data[key]
        plt.plot(recalls, precisions, label=key[:key.index("precisions_recalls") - 1])
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("precision vs recall " + args.eval_session_group_name)
plt.legend(loc="lower left", prop={'size': 6})
plt.savefig(os.path.join(args.visualizations_path, "precision_recall_" + args.eval_session_group_name))

plt.clf()
for key in pr_roc_data:
    print(key)
    if "fprs_tprs" in key:
        fprs, tprs = pr_roc_data[key]
        plt.plot(fprs, tprs, label=key[:key.index("fprs_tprs") - 1])

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC " + args.eval_session_group_name)
plt.legend(loc="lower right", prop={'size': 6})
plt.savefig(os.path.join(args.visualizations_path, "roc_" + args.eval_session_group_name))
