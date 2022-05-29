import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SidewalkCropsDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, eval=False):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.targets = self.img_labels['label_type']
    self.transform = transform
    self.target_transform = target_transform
    self.eval = eval

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['image_name'])
    image = Image.open(img_path)
    label = self.targets[idx]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    if self.eval:
      return image, label, img_path
    else:
      return image, label
