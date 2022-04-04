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
import config

def get_precision_recall(output_probabilities, corresponding_ground_truths, prob_cutoff=.5):
  classifications = torch.where(output_probabilities > prob_cutoff, 1, 0)
  correct_classifications = torch.where(classifications == corresponding_ground_truths, classifications, -1)
  
  true_positive_count = torch.count_nonzero(correct_classifications == 1)
  actual_positive_count = torch.count_nonzero(corresponding_ground_truths == 1)
  predicted_positive_count = torch.count_nonzero(classifications == 1)

  return true_positive_count / predicted_positive_count, true_positive_count / actual_positive_count

if not os.path.isdir(config.VISUALIZATIONS_PATH):
    print("made visualization folder")
    os.makedirs(config.VISUALIZATIONS_PATH)

# the actual classes
CLASSES = ["null", "curb ramp", "missing curb ramp", "obstacle", "surface problem"]

# save path for model
CHECKPOINT_SAVE_PATH = os.path.join(config.MODEL_SAVE_FOLDER, config.SESSION_NAME + ".pt")

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
model, input_size = get_pretrained_model(config.MODEL_NAME, config.NUM_CLASSES)
model.to(device)

load_best_weights(model, CHECKPOINT_SAVE_PATH)

loss_func = nn.CrossEntropyLoss()

# =================================================================================================
# load our custom test sidewalk crops dataset
image_transform = transforms.Compose([
  transforms.CenterCrop(config.CROP_SIZE),
  transforms.Resize(input_size),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_labels_csv_path = os.path.join(config.CSV_BASE_PATH, TEST_SET_CSV)
test_dataset = SidewalkCropsDataset(test_labels_csv_path, config.IMAGE_BASE_PATH, transform=image_transform, eval=True)

batch_size = 12

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# =================================================================================================
# evaluate loaded model on test set
MISTAKES_SAVE_PATH = os.path.join(VISUALIZATIONS_PATH, config.SESSION_NAME + "_mistakes.csv")
cm, output_probabilities, corresponding_ground_truths = evaluate(model, (config.MODEL_NAME == "inception"), loss_func, test_dataloader, True, MISTAKES_SAVE_PATH,device)
if cm is not None:
  plot_confusion_matrix(VISUALIZATIONS_PATH, config.SESSION_NAME, cm, config.CLASSES, normalize=True)

precisions = []
recalls = []
for prob_cutoff in torch.linspace(0, 1, 100000):
  precision, recall = get_precision_recall(output_probabilities, corresponding_ground_truths, prob_cutoff)
  precisions.append(precision)
  recalls.append(recall)
plt.scatter(recalls, precisions)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("precision vs recall " + config.SESSION_NAME)
plt.savefig(os.path.join(VISUALIZATIONS_PATH, "precision_recall_" + config.SESSION_NAME))