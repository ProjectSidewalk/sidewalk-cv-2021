import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

def plot_confusion_matrix(visualizations_path, model_name, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(visualizations_path + model_name + ".png", format="png")
    plt.show()
    plt.close()

def visualize_mistakes(model, is_inception, loss_func, dataset_loader, test, device):
    # put model into eval mode
    model.eval()

    # length of data set we are evaluating on.
    n = len(dataset_loader.dataset)

    #
    incorrect_predictions = []
    corresponding_ground_truths = []
    corresponding_image_names = []

    epoch_count = 0
    # correct predictions.
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels, paths in dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_inception:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            incorrect_indices = torch.nonzero(predictions != labels, as_tuple=True)[0]
            for index in incorrect_indices:
                incorrect_predictions.append(predictions[index])
                corresponding_ground_truths.append(labels[index])
                corresponding_image_names.append(paths[index])
