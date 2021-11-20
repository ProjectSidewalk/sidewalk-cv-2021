import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from datatypes.dataset import SidewalkCropsDataset
from utils.training_utils import get_pretrained_model, load_training_checkpoint, train
from torch.optim import lr_scheduler
from torchvision import transforms

files = glob.glob('./fake_folder/*')
print(files)