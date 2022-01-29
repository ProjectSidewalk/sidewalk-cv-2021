import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.pyplot import figure
 
label_types = {
    0: "null",
    1: "curb ramp",
    2: "missing curb ramp",
    3: "obstacle", 
    4: "surface problem"
}

VISUALIZATIONS_PATH = "../visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    os.makedirs(VISUALIZATIONS_PATH)

sizes = [1000, 2500, 5000, 10000, 20000, 37500]

SESSION_NAMES = [f"model_{size}" for size in sizes]

final_train_accuracies, final_val_accuracies = [], []

final_precisions = [[], [], [], [], []]
final_recalls = [[], [], [], [], []]


for name in SESSION_NAMES:
    precisions[name]
    TRAIN_SAVE_PATH = "../datasets/" + SESSION_NAME + ".pt"

    results = torch.load(TRAIN_SAVE_PATH)
    metrics = results['metrics']
    epochs = results['epoch'] + 1  # epochs are 0-indexed in checkpoint

    val_accuracy, train_accuracy = metrics['accuracy_validation'], metrics['accuracy_train']
    max_train_accuracies.append(train_accuracy[-1])
    max_val_accuracies.append(val_accuracy[-1])

    final_precisions = metrics['precision_validation'][-1]
    final_recalls = metrics['recall_validation'][-1]
    for precision, i in enumerate(final_precisions):
        precisions[i].append(precision)
    for recall, i in enumerate(final_recalls):
        recalls[i].append(recall)

figure(figsize=(16, 12))
plt.plot(sizes, final_train_accuracies, label="train")
plt.plot(sizes, final_val_accuracies, label="validation")
plt.title("accuracy vs dataset size")
plt.savefig(f"{VISUALIZATIONS_PATH}/accuracy_vs_size")

figure(figsize=(16, 12))
for i in range(5):
    plt.plot(sizes, final_precisions[i], label=label_types[i])
plt.title("precisions vs dataset size")
plt.savefig(f"{VISUALIZATIONS_PATH}/precision_vs_size")

figure(figsize=(16, 12))
for i in range(5):
    plt.plot(sizes, final_recalls[i], label=label_types[i])
plt.title("recalls vs dataset size")
plt.savefig(f"{VISUALIZATIONS_PATH}/recall_vs_size")



