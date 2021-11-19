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

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# load train datasets
image_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# having issues with CUDA running out of memory, so lowering batch size
batch_size = 16

train_labels_csv_path = BASE_PATH + "train_non_null_crop_info.csv"
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

train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

print(len(train_dataset))
print(len(val_dataset))

# get resnet50 for fine tuning
model = get_pretrained_model()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5) # (1,2,3,4) for label types, 0 for null crops 
model.to(device)

lr = 0.01

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
checkpoint_save_path = BASE_PATH + "regnet_save.pt"

# train for 20 epochs
epochs = 20
dataLoaders = {
  "training": train_dataloader,
  "validation": val_dataloader
}
metrics, last_epoch = load_training_checkpoint(model, checkpoint_save_path, optimizer, scheduler)
print("next epoch: " + str(last_epoch + 1))
print("resuming training...\n")

train(model, optimizer, scheduler, loss_func, epochs, dataLoaders, checkpoint_save_path, metrics, last_epoch + 1, device)
# print("Best validation accuracy: ", best_validation_accuracy)

# visualization of training and validation loss over epochs
plt.plot(np.arange(epochs), metrics['loss_train'], label="training loss")
plt.plot(np.arange(epochs), metrics['loss_validation'], label="validation loss")
plt.title("Training/Validation loss for FT model")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
