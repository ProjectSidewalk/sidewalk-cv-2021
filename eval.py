import argparse
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
from sklearn.metrics import precision_recall_curve, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument('image_base_path', type=str)
parser.add_argument('csv_base_path', type=str)
parser.add_argument('test_set_csv', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('model_save_folder', type=str)
parser.add_argument('session_name', type=str)
parser.add_argument('crop_size', type=int)
parser.add_argument('visualizations_path', type=str)
args = parser.parse_args()

CLASSES = ["null", "curb ramp", "missing curb ramp", "obstacle", "surface problem"]
NUM_CLASSES = 2

def get_precision_recall(output_probabilities, corresponding_ground_truths, prob_cutoff=.5):
  classifications = torch.where(output_probabilities > prob_cutoff, 1, 0)
  correct_classifications = torch.where(classifications == corresponding_ground_truths, classifications, -1)
  
  true_positive_count = torch.count_nonzero(correct_classifications == 1)
  actual_positive_count = torch.count_nonzero(corresponding_ground_truths == 1)
  predicted_positive_count = torch.count_nonzero(classifications == 1)

  return true_positive_count / predicted_positive_count, true_positive_count / actual_positive_count

if not os.path.isdir(args.visualizations_path):
    print("made visualization folder")

# save path for model
CHECKPOINT_SAVE_PATH = os.path.join(args.model_save_folder, args.session_name + ".pt")

# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print("device:", device)

# =================================================================================================
# load model for evaluation
# setup model for fine tuning
model, input_size = get_pretrained_model(args.model_name, NUM_CLASSES)
model.to(device)

load_best_weights(model, CHECKPOINT_SAVE_PATH)

loss_func = nn.CrossEntropyLoss()

# =================================================================================================
# load our custom test sidewalk crops dataset
image_transform = transforms.Compose([
  transforms.CenterCrop(args.crop_size),
  transforms.Resize(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_labels_csv_path = os.path.join(args.csv_base_path, args.test_set_csv)
test_dataset = SidewalkCropsDataset(test_labels_csv_path, args.image_base_path, transform=image_transform, eval=True)

batch_size = 12
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# =================================================================================================
# evaluate loaded model on test set
MISTAKES_SAVE_PATH = os.path.join(args.visualizations_path, args.session_name + "_mistakes.csv")
cm, output_probabilities, corresponding_ground_truths = evaluate(model, (args.model_name == "inception"), loss_func, test_dataloader, True, MISTAKES_SAVE_PATH,device)
if cm is not None:
  plot_confusion_matrix(args.visualizations_path, args.session_name, cm, CLASSES, normalize=True)

precisions, recalls, _ = precision_recall_curve(corresponding_ground_truths, output_probabilities)
plt.plot(recalls, precisions)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("precision vs recall " + args.session_name)
plt.savefig(os.path.join(args.visualizations_path, "precision_recall_" + args.session_name))
plt.clf()

false_positive_rates, true_positive_rates, _ = roc_curve(corresponding_ground_truths, output_probabilities)
plt.plot(false_positive_rates, true_positive_rates)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC " + SESSION_NAME)
plt.savefig(os.path.join(args.visualizations_path, "roc_" + args.session_name))

precision_default_cutoff, recall_default_cutoff = get_precision_recall(output_probabilities, corresponding_ground_truths, .5)
print("precision at default cutoff: " + str(precision_default_cutoff))
print("recall at default cutoff: " + str(recall_default_cutoff))
