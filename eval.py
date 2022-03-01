import os
import torch
import torch.nn as nn
import torchvision
import citysurfaces.network.hrnetv2 as hrnetv2
import matplotlib.pyplot as plt
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import get_pretrained_model, load_best_weights, evaluate
from visualization_utils.confusion_matrix import plot_confusion_matrix
from torchvision import transforms

def get_precision_recall(output_probabilities, corresponding_ground_truths, prob_cutoff=.5):
  classifications = torch.where(output_probabilities > prob_cutoff, 1, 0)
  correct_classifications = torch.where(classifications == corresponding_ground_truths, classifications, -1)
  
  true_positive_count = torch.count_nonzero(correct_classifications == 1)
  actual_positive_count = torch.count_nonzero(corresponding_ground_truths == 1)
  predicted_positive_count = torch.count_nonzero(classifications == 1)

  return true_positive_count / predicted_positive_count, true_positive_count / actual_positive_count

VISUALIZATIONS_PATH = "./visualizations/"
if not os.path.isdir(VISUALIZATIONS_PATH):
    print("made visualization folder")
    os.makedirs(VISUALIZATIONS_PATH)

# set base path to training/test data folder
IMAGE_BASE_PATH = "/tmp/datasets/crops/"

# set different base path for CSVs in case /tmp gets deleted
CSV_BASE_PATH = "./datasets/"

# save path for model weights
MODEL_SAVE_FOLDER = "./models/"

# name of model architecture
MODEL_NAME = "MODEL NAME HERE"

# number of output classes
NUM_CLASSES = "NUM CLASSES HERE"

# the actual classes
CLASSES = ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"]

# name of training session for loading purposes
SESSION_NAME = "SESSION NAME HERE"
PRETRAINED_SAVE_PATH = MODEL_SAVE_FOLDER + SESSION_NAME + ".pt"

# for zoom testing
CROP_SIZE = "CROP SIZE HERE"

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# =================================================================================================
# load model for evaluation
# setup model for fine tuning
model, input_size = get_pretrained_model(MODEL_NAME, NUM_CLASSES)
model.to(device)

load_best_weights(model, PRETRAINED_SAVE_PATH)

loss_func = nn.CrossEntropyLoss()

# =================================================================================================
# load our custom test sidewalk crops dataset
image_transform = transforms.Compose([
  transforms.CenterCrop(CROP_SIZE),
  transforms.Resize(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_labels_csv_path = CSV_BASE_PATH + "CSV NAME HERE"
test_dataset = SidewalkCropsDataset(test_labels_csv_path, IMAGE_BASE_PATH, transform=image_transform, eval=True)

batch_size = 12

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# =================================================================================================
# evaluate loaded model on test set
MISTAKES_SAVE_PATH = "./visualizations/" + SESSION_NAME + "_mistakes.csv"
test_accuracy, test_loss, cm, output_probabilities, corresponding_ground_truths = evaluate(model, (MODEL_NAME == "inception"), loss_func, test_dataloader, True, MISTAKES_SAVE_PATH,device)
print("Test accuracy for {} as FT: ".format(MODEL_NAME), test_accuracy)
print("Test loss for {} as FT: ".format(MODEL_NAME), test_loss)
if cm is not None:
  plot_confusion_matrix(VISUALIZATIONS_PATH, SESSION_NAME, cm, CLASSES, normalize=True)

precisions = []
recalls = []
for prob_cutoff in torch.linspace(.001, .999, 1000):
  precision, recall = get_precision_recall(output_probabilities, corresponding_ground_truths, prob_cutoff)
  precisions.append(precision)
  recalls.append(recall)
plt.scatter(recalls, precisions)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("precision vs recall " + SESSION_NAME)
plt.savefig(VISUALIZATIONS_PATH + "precision_recall_" + SESSION_NAME)
