import os
import torch
import torch.nn as nn
import torchvision
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import load_best_weights, evaluate
from utils.visualization_utils import plot_confusion_matrix
from torchvision import transforms

VISUALIZATIONS_PATH = "./visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    print("made visualization folder")
    os.makedirs(VISUALIZATIONS_PATH)

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# load our custom test sidewalk crops dataset
image_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

BASE_PATH = "./datasets/"
test_labels_csv_path = BASE_PATH + "test_crop_info.csv"
test_img_dir = BASE_PATH + "test_crops/"
test_dataset = SidewalkCropsDataset(test_labels_csv_path, test_img_dir, image_transform)

batch_size = 32

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# load model for evaluation
efficientnetb3 = torchvision.models.resnet50(pretrained = True)
num_ftrs = efficientnetb3.fc.in_features
efficientnetb3.fc = nn.Linear(num_ftrs, 5) # (1,2,3,4) for label types, 0 for null crops 
efficientnetb3.to(device)
loss_func = nn.CrossEntropyLoss()

pretrained_save_path = BASE_PATH + "resnet50_weighted_loss.pt"
load_best_weights(efficientnetb3, pretrained_save_path)

# evaluate loaded model on test set
test_accuracy, test_loss, cm = evaluate(efficientnetb3, loss_func, test_dataloader, True, device)
print("Test accuracy for ResNet50 as FT: ", test_accuracy)
print("Test loss for ResNet50 as FT: ", test_loss)
if cm is not None:
  plot_confusion_matrix(VISUALIZATIONS_PATH, "resnet50-weighted-loss", cm, ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"], normalize=True)
