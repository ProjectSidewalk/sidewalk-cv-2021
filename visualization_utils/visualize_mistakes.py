import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('session_name', type=str)
parser.add_argument('image_base_path', type=str)
parser.add_argument('visualizations_path', type=str)
parser.add_argument('crop_size', type=int)
parser.add_argument('num_plots', type=int)
args = parser.parse_args()

MISTAKES_SAVE_PATH = os.path.join(args.visualizations_path, args.session_name + '_mistakes')
FALSE_POSITIVES_SAVE_PATH = os.path.join(args.visualizations_path, args.session_name + '_false_positives')
FALSE_NEGATIVES_SAVE_PATH = os.path.join(args.visualizations_path, args.session_name + '_false_negatives')

IMAGES_PER_ROW = 5
IMAGES_PER_COL = 3
IMAGE_SIZE = 5
IMAGES_PER_PLOT = IMAGES_PER_ROW * IMAGES_PER_COL
label_types = {
    0: 'negative',
    1: 'curb ramp',
    2: 'missing curb ramp',
    3: 'obstacle', 
    4: 'surface problem'
}

def add_border(image, mistake_type):
    width, height = image.size
    pixels = image.load()
    color = (255, 0, 0) if mistake_type == 'false positives' else (255, 215, 0)

    for y in range(0, 20):
        for x in range(0, width):
            pixels[x, y] = color
    for y in range(height-20, height):
        for x in range(0, width):
            pixels[x, y] = color

    for x in range(0, 20):
        for y in range(0, height):
            pixels[x, y] = color

    for x in range(width-20, width):
        for y in range(0, height):
            pixels[x, y] = color

def crop(image):
    width, height = image.size   # Get dimensions

    left = (width - args.crop_size)/2
    top = (height - args.crop_size)/2
    right = (width + args.crop_size)/2
    bottom = (height + args.crop_size)/2

    # Crop the center of the image
    return image.crop((left, top, right, bottom))

def make_plots(mistakes, num_plots, mistake_type):
    for plot_idx in range(num_plots):
        start_row = IMAGES_PER_PLOT * plot_idx
        end_row = start_row + IMAGES_PER_PLOT # exclusive
        plot_rows = mistakes.iloc[start_row:end_row]
        plot_rows.reset_index(drop=True, inplace=True)
        fig = plt.figure(num=1, figsize=(IMAGES_PER_ROW * IMAGE_SIZE, IMAGES_PER_COL * IMAGE_SIZE))
        fig.suptitle(f'{args.session_name} {mistake_type} {plot_idx}', fontsize=30)
        for i, mistake in plot_rows.iterrows():
            image = Image.open(f'{mistake["image path"]}')

            image = crop(image)
            add_border(image, mistake_type)

            path = mistake["image path"][len(args.image_base_path):]
            confidence = mistake['confidence']
            ax = plt.subplot(IMAGES_PER_COL, IMAGES_PER_ROW, i+1)
            plt.axis('off')
            ax.set_title(f'{path}\n confidence: {confidence:.4f}', fontsize=15)
            ax.spines['bottom'].set_color('0.5')
            plt.imshow(image)
        save_path = FALSE_POSITIVES_SAVE_PATH if mistake_type == 'false positives' else FALSE_NEGATIVES_SAVE_PATH
        plt.savefig(f'{save_path}{plot_idx}.png', bbox_inches='tight')
        plt.clf()

if __name__ == '__main__':
    all_mistakes = pd.read_csv(f'{MISTAKES_SAVE_PATH}.csv')
    all_mistakes = all_mistakes.sample(frac=1, random_state=1).reset_index(drop=True)

    print(len(all_mistakes))

    false_negatives = all_mistakes[all_mistakes['prediction'] == 0]
    false_positives = all_mistakes[all_mistakes['prediction'] != 0]

    make_plots(false_negatives, args.num_plots, 'false negatives (predicted negative, actual positive)')
    make_plots(false_positives, args.num_plots, 'false positives (predicted positive, actual negative)')
