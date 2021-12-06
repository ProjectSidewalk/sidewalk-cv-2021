import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from datatypes.dataset import SidewalkCropsDataset
from PIL import Image
from utils.training_utils import get_pretrained_model, load_best_weights, evaluate
from utils.visualization_utils import visualize_mistakes
from torchvision import transforms


SESSION_NAME = "SESSION_NAME_HERE"
MISTAKES_SAVE_PATH = "./visualizations/" + SESSION_NAME + "-mistakes"

IMAGES_PER_ROW = 5
IMAGES_PER_COL = 3
IMAGE_SIZE = 5
IMAGES_PER_PLOT = IMAGES_PER_ROW * IMAGES_PER_COL
label_types = {
    0: "null",
    1: "curb ramp",
    2: "missing curb ramp",
    3: "obstacle", 
    4: "surface problem"
}

mistakes = pd.read_csv(f"{MISTAKES_SAVE_PATH}.csv")
num_plots = len(mistakes) // (IMAGES_PER_ROW * IMAGES_PER_COL)
for plot_idx in range(num_plots // 10): # only plot 10% of mistakes for now
    start_row = IMAGES_PER_PLOT * plot_idx
    end_row = start_row + IMAGES_PER_PLOT # exclusive
    plot_rows = mistakes.iloc[start_row:end_row]
    plot_rows.reset_index(drop=True, inplace=True)
    fig = plt.figure(num=1, figsize=(IMAGES_PER_ROW * IMAGE_SIZE, IMAGES_PER_COL * IMAGE_SIZE))
    fig.suptitle(f"{SESSION_NAME} misclassifications {plot_idx}", fontsize=30)
    for i, mistake in plot_rows.iterrows():
        image = Image.open(mistake['image path'])
        predicted = label_types[mistake['prediction']]
        actual = label_types[mistake['ground truth']]
        ax = plt.subplot(IMAGES_PER_COL, IMAGES_PER_ROW, i+1)
        plt.axis("off")
        ax.set_title(f"{mistake['image path'][22:]}\npred: {predicted}\n actual: {actual}", fontsize=15)
        plt.imshow(image)
    plt.savefig(f"{MISTAKES_SAVE_PATH}_{plot_idx}.png", bbox_inches="tight")
    plt.clf()