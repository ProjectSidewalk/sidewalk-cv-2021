import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.pyplot import figure
 

VISUALIZATIONS_PATH = "./visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    os.makedirs(VISUALIZATIONS_PATH)

SESSION_NAME = 'regnet_save.pt'
TRAIN_SAVE_PATH = "./datasets/" + SESSION_NAME
label_types = {
    0: "null",
    1: "curb ramp",
    2: "missing curb ramp",
    3: "obstacle", 
    4: "surface problem"
}

results = torch.load(TRAIN_SAVE_PATH)
metrics = results['metrics']
epochs = 50

def plot_label_metric(metric_name):
    figure(figsize=(16, 12))
    stacked = torch.stack(metrics[metric_name])
    flipped_metric = [stacked[:, i] for i in range(1, 5)]
    for i, metric in enumerate(flipped_metric):
        metric = metric.cpu()
        plt.plot(np.arange(epochs), metric, label = label_types[i+ 1])
    plt.title(f'{metric_name} vs epoch', fontsize=20)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(VISUALIZATIONS_PATH + metric_name)

plot_label_metric('precision_validation')
plot_label_metric('precision_train')
plot_label_metric('recall_validation')
plot_label_metric('recall_train')

figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['accuracy_train'], label = 'train accuracy')
plt.plot(np.arange(epochs), metrics['accuracy_validation'], label = 'validation accuracy')
plt.title(f'accuracy vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("accuracy", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(VISUALIZATIONS_PATH + "accuracies")

figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['loss_train'], label = 'train loss')
plt.plot(np.arange(epochs), metrics['loss_validation'], label = 'validation loss')
plt.title(f'loss vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(VISUALIZATIONS_PATH + "losses")
