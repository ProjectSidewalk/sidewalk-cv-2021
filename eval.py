import os
import torch
import torch.nn as nn
import torchvision
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import get_pretrained_model, load_best_weights, evaluate
from visualization_utils.confusion_matrix import plot_confusion_matrix
from torchvision import transforms

VISUALIZATIONS_PATH = "./visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    print("made visualization folder")
    os.makedirs(VISUALIZATIONS_PATH)

# set base path to test data folder
BASE_PATH = "./datasets/"

# name of model architecture
MODEL_NAME = "efficientnet"

# number of output classes
NUM_CLASSES = 5  # (1,2,3,4) for label types, 0 for null crops

# the actual classes
CLASSES = ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"]

# name of training session for loading purposes
SESSION_NAME = "large-efficientnet"

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# =================================================================================================
# load model for evaluation
model, input_size = get_pretrained_model(MODEL_NAME, NUM_CLASSES, False)
model.to(device)

pretrained_save_path = BASE_PATH + SESSION_NAME + ".pt"
load_best_weights(model, pretrained_save_path)

loss_func = nn.CrossEntropyLoss()

# =================================================================================================
# load our custom test sidewalk crops dataset
image_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_labels_csv_path = BASE_PATH + "train_large_crop_info.csv"
test_img_dir = BASE_PATH + "train_crops/"
test_dataset = SidewalkCropsDataset(test_labels_csv_path, test_img_dir, transform=image_transform, eval=True)

batch_size = 32

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# =================================================================================================
# evaluate loaded model on test set
MISTAKES_SAVE_PATH = "./visualizations/" + SESSION_NAME + "-mistakes.csv"
test_accuracy, test_loss, cm = evaluate(model, (MODEL_NAME == "inception"), loss_func, test_dataloader, True, MISTAKES_SAVE_PATH,device)
print("Test accuracy for {} as FT: ".format(MODEL_NAME), test_accuracy)
print("Test loss for {} as FT: ".format(MODEL_NAME), test_loss)
if cm is not None:
  plot_confusion_matrix(VISUALIZATIONS_PATH, SESSION_NAME, cm, CLASSES, normalize=True)
