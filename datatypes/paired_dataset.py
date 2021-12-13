import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PairedCropsDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, eval=False):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
    self.eval = eval

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    label = self.img_labels.iloc[idx, 1]
    image_pair = None
    if label == 0:
        # we'll send the null crops through both models, disregarding crop size
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        image_pair = [image, image]
    else:
        small_img_path = img_path + "_0.jpg"
        large_img_path = img_path + "_1.jpg"
        small_img = Image.open(small_img_path)
        large_img = Image.open(large_img_path)
        if self.transform:
            small_img = self.transform(small_img)
            large_img = self.transform(large_img)

        image_pair = [small_img, large_img]

    if self.target_transform:
      label = self.target_transform(label)
    if self.eval:
      return image_pair, label, img_path
    else:
      return image_pair, label
