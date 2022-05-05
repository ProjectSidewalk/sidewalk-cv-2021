import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.pyplot import figure

parser = argparse.ArgumentParser()
parser.add_argument('model_save_folder', type=str)
parser.add_argument('session_name', type=str)
parser.add_argument('visualizations_path', type=str)
args = parser.parse_args()

CLASSES = ["null", "curb ramp", "missing curb ramp", "obstacle", "surface problem"]
BINARY_CLASSES = ["negative", "positive"]
NUM_CLASSES = 2

if not os.path.isdir(args.visualizations_path):
    os.makedirs(args.visualizations_path)

TRAIN_SAVE_PATH = os.path.join(args.model_save_folder, args.session_name + ".pt")

multiclass_labels = dict()
for i, c in enumerate(CLASSES):
    multiclass_labels[i] = c
binary_labels = dict()
for i, c in enumerate(BINARY_CLASSES):
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
    plt.savefig(os.path.join(args.visualizations_path, metric_name + "_" + args.session_name))

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
plt.savefig(os.path.join(args.visualizations_path, "accuracies_" + args.session_name))

figure(figsize=(16, 12))
plt.plot(np.arange(epochs), metrics['loss_train'], label = 'train loss')
plt.plot(np.arange(epochs), metrics['loss_validation'], label = 'validation loss')
plt.title(f'loss vs epoch', fontsize=20)
plt.xlabel("epoch", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.legend(prop={'size': 16})
plt.savefig(os.path.join(args.visualizations_path, "losses_" + args.session_name))
