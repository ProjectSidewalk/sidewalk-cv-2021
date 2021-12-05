import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import get_pretrained_model, load_training_checkpoint, train
from torch.optim import lr_scheduler
from torchvision import transforms

# set base path to training/test data folder
BASE_PATH = "./datasets/"

# name of model architecture
MODEL_NAME = "MODEL NAME HERE"

# number of output classes
NUM_CLASSES = 5  # (1,2,3,4) for label types, 0 for null crops

# name of training session for saving purposes
TRAIN_SESSION_NAME = "TRAIN SESSION NAME HERE"

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# =================================================================================================
# setup model for fine tuning
model, input_size = get_pretrained_model(MODEL_NAME, NUM_CLASSES, False)
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
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=lr, step_size_up = 2500, mode='triangular2') # lr_scheduler.StepLR(optimizer, 10, gamma=0.3)
checkpoint_save_path = BASE_PATH + TRAIN_SESSION_NAME + ".pt"

# =================================================================================================
# load train datasets
image_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# having issues with CUDA running out of memory, so lowering batch size
batch_size = 32

train_labels_csv_path = BASE_PATH + "train_crop_info.csv"
train_img_dir = BASE_PATH + "train_crops/"

# load our custom train/val sidewalk crops dataset
train_val_dataset = SidewalkCropsDataset(train_labels_csv_path, train_img_dir, image_transform)

# partition train dataset into 80/20 split for train/validation
k = .8
train_val_dataset_size = len(train_val_dataset)
train_size = int(k * train_val_dataset_size)
val_size = train_val_dataset_size - train_size
print(train_size)
print(val_size)

torch.manual_seed(0)
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

print(len(train_dataset))
print(len(val_dataset))

# =================================================================================================
# train for n epochs
epochs = 50
dataLoaders = {
  "training": train_dataloader,
  "validation": val_dataloader
}
metrics, last_epoch = load_training_checkpoint(model, checkpoint_save_path, optimizer, scheduler)
print("next epoch: " + str(last_epoch + 1))
print("resuming training...\n")

train(model, NUM_CLASSES, (MODEL_NAME == "inception"), optimizer, scheduler, loss_func, epochs, dataLoaders,
      checkpoint_save_path, metrics, last_epoch + 1, device)
