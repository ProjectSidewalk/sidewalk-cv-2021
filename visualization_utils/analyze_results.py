import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.pyplot import figure
 

VISUALIZATIONS_PATH = "../visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    os.makedirs(VISUALIZATIONS_PATH)

SESSION_NAME = "with_nulls"
TRAIN_SAVE_PATH = "../models/" + SESSION_NAME + ".pt"
label_types = {
    0: "null",
    1: "curb ramp",
    2: "missing curb ramp",
    3: "obstacle", 
    4: "surface problem"
}

binary_labels = {
    0: "negative",
    1: "positive"
}

NUM_CLASSES = 5

results = torch.load(TRAIN_SAVE_PATH)
metrics = results['metrics']
epochs = results['epoch'] + 1  # epochs are 0-indexed in checkpoint

def plot_label_metric(metric_name, num_classes):
    figure(figsize=(16, 12))
    stacked = torch.stack(metrics[metric_name])
    flipped_metric = [stacked[:, i] for i in range(num_classes)]
    is_binary = (num_classes == 2)
    for i, metric in enumerate(flipped_metric):
        metric = metric.cpu()
        plt.plot(np.arange(epochs), metric, label = (binary_labels if is_binary else label_types)[i])
    plt.title(f'{metric_name} vs epoch', fontsize=20)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(VISUALIZATIONS_PATH + metric_name + "_" + SESSION_NAME)

plot_label_metric('precision_validation', NUM_CLASSES)
plot_label_metric('precision_train', NUM_CLASSES)
plot_label_metric('recall_validation', NUM_CLASSES)
plot_label_metric('recall_train', NUM_CLASSES)

print("accuracy: " + str(metrics['accuracy_validation']))
figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['accuracy_train'], label = 'train accuracy')
plt.plot(np.arange(epochs), metrics['accuracy_validation'], label = 'validation accuracy')
plt.title(f'accuracy vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("accuracy", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(VISUALIZATIONS_PATH + "accuracies_" + SESSION_NAME)

figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['loss_train'], label = 'train loss')
plt.plot(np.arange(epochs), metrics['loss_validation'], label = 'validation loss')
plt.title(f'loss vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(VISUALIZATIONS_PATH + "losses_" + SESSION_NAME)
