import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

session_name = 'regnet_save.pt'
train_save_path = "./datasets/" + session_name
results = torch.load(train_save_path)
metrics = results['metrics']

def plot_label_metric(metric_name):
    figure(figsize=(16, 12))
    stacked = torch.stack(metrics[metric_name])
    flipped_metric = [stacked[:, i] for i in range(1, 5)]
    for i, metric in enumerate(flipped_metric):
        metric = metric.cpu()
        plt.plot(np.arange(20), metric, label = str(i+ 1))
    plt.title(f'{metric_name} vs epoch', fontsize=20)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(metric_name)

plot_label_metric('precision_validation')
plot_label_metric('precision_train')
plot_label_metric('recall_validation')
plot_label_metric('recall_train')

figure(figsize=(16, 12))
plt.plot(np.arange(20), metrics['accuracy_train'], label = 'train accuracy')
plt.plot(np.arange(20), metrics['accuracy_validation'], label = 'validation accuracy')
plt.title(f'accuracy vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("accuracy", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig("accuracies")