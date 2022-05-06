import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import citysurfaces.network.hrnetv2 as hrnetv2
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import get_pretrained_model, load_training_checkpoint, train
from torch.optim import lr_scheduler
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('session_name', type=str)
parser.add_argument('image_base_path', type=str)
parser.add_argument('csv_base_path', type=str)
parser.add_argument('train_set_csv', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('model_save_folder', type=str)
parser.add_argument('crop_size', type=int)
args = parser.parse_args()

NUM_CLASSES = 2

if not os.path.isdir(args.model_save_folder):
  os.makedirs(args.model_save_folder)

# save path for model
CHECKPOINT_SAVE_PATH = os.path.join(args.model_save_folder, args.session_name + ".pt")

if __name__ == "__main__":
  # check for GPU
  if torch.cuda.is_available():  
    dev = "cuda" 
  else:  
    dev = "cpu"
  device = torch.device(dev) 
  print("device:", device)

  # =================================================================================================
  # setup model for fine tuning
  model, input_size = get_pretrained_model(args.model_name, NUM_CLASSES)
  model.to(device)

  lr = 0.01
  momentum = 0.9
  weight_decay = 1e-6

  # weight using inverse of each sample size
  # acquire label sample sizes from train csv
  # samples_per_class = np.array([10000, 11187, 8788, 2678, 7204])
  # weights = 1.0 / samples_per_class
  # norm = np.linalg.norm(weights)
  # normalized_weights = weights / norm
  # normalized_weights_tensor = torch.from_numpy(normalized_weights).float().to(device)

  # add normalized_weights_tensor as input to loss_func if weighted loss is desired
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) # torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
  scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.5) #lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, step_size_up = 2500, mode='triangular2')

  # =================================================================================================
  # load train datasets
  image_transform = transforms.Compose([
    transforms.CenterCrop(args.crop_size),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # having issues with CUDA running out of memory, so lowering batch size
  batch_size = 8

  train_labels_csv_path = os.path.join(args.csv_base_path, args.train_set_csv)
  train_img_dir = args.image_base_path

  # load our custom train/val sidewalk crops dataset
  train_val_dataset = SidewalkCropsDataset(train_labels_csv_path, train_img_dir, image_transform, eval=False)

  # partition train dataset into 80/20 split for train/validation
  k = .8
  train_val_dataset_size = len(train_val_dataset)
  train_size = int(k * train_val_dataset_size)
  val_size = train_val_dataset_size - train_size
  print("train size:", train_size)
  print("validation size:", val_size)

  torch.manual_seed(0)
  train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

  # =================================================================================================
  # train for n epochs
  epochs = 25
  dataLoaders = {
    "training": train_dataloader,
    "validation": val_dataloader
  }
  metrics, last_epoch = load_training_checkpoint(model, CHECKPOINT_SAVE_PATH, optimizer, scheduler)
  print("next epoch:", last_epoch + 1)
  print("resuming training...\n")

  train(model, NUM_CLASSES, (args.model_name == "inception"), optimizer, scheduler, loss_func, epochs, dataLoaders,
        CHECKPOINT_SAVE_PATH, metrics, last_epoch + 1, device)
