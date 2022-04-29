import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.pyplot import figure
import config
 
if not os.path.isdir(config.VISUALIZATIONS_PATH):
    os.makedirs(config.VISUALIZATIONS_PATH)

TRAIN_SAVE_PATH = os.path.join("../", config.MODEL_SAVE_FOLDER, config.SESSION_NAME + ".pt")
multiclass_labels = dict()
for i, c in enumerate(config.CLASSES):
    multiclass_labels[i] = c
binary_labels = dict()
for i, c in enumerate(config.BINARY_CLASSES):
    binary_labels[i] = c 

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
        plt.plot(np.arange(epochs), metric, label = (binary_labels if is_binary else multiclass_labels)[i])
    plt.title(f'{metric_name} vs epoch', fontsize=20)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(os.path.join(config.VISUALIZATIONS_PATH, metric_name + "_" + config.SESSION_NAME))

plot_label_metric('precision_validation', config.NUM_CLASSES)
plot_label_metric('precision_train', config.NUM_CLASSES)
plot_label_metric('recall_validation', config.NUM_CLASSES)
plot_label_metric('recall_train', config.NUM_CLASSES)

print("accuracy: " + str(metrics['accuracy_validation']))
figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['accuracy_train'], label = 'train accuracy')
plt.plot(np.arange(epochs), metrics['accuracy_validation'], label = 'validation accuracy')
plt.title(f'accuracy vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("accuracy", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(os.path.join(config.VISUALIZATIONS_PATH, "accuracies_" + config.SESSION_NAME))

figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['loss_train'], label = 'train loss')
plt.plot(np.arange(epochs), metrics['loss_validation'], label = 'validation loss')
plt.title(f'loss vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(os.path.join(config.VISUALIZATIONS_PATH, "losses_" + config.SESSION_NAME))
