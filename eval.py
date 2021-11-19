import torch
import torch.nn as nn
import torchvision
from .datatypes.dataset import SidewalkCropsDataset
from .utils.training_utils import load_training_checkpoint, evaluate
from .utils.visualization_utils import plot_confusion_matrix
from torchvision import transforms

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

batch_size = 128

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# load model for evaluation
resnet50 = torchvision.models.resnet50(pretrained = True).to(device)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 5) # (1,2,3,4) for label types, 0 for null crops 
resnet50.to(device)
loss_func = nn.CrossEntropyLoss()

pretrained_save_path = BASE_PATH + "training_test_saves"
load_training_checkpoint(resnet50, pretrained_save_path)

# evaluate loaded model on test set
test_accuracy, test_loss, cm = evaluate(resnet50, loss_func, test_dataloader, True)
print("Test accuracy for ResNet as FT: ", test_accuracy)
print("Test loss for ResNet as FT: ", test_loss)
if cm is not None:
  plot_confusion_matrix(cm, ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"])