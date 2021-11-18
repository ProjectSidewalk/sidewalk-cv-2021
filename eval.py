import torch
from .datatypes.dataset import SidewalkCropsDataset
from .utils.training_utils import load_training_checkpoint
from torchvision import transforms

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

# make sure to evaluate on *_pretrained if loading pretrained model
# test_accuracy, test_loss, cm = evaluate(resnet50_pretrained, loss_func, test_dataloader, True)
# print("Test accuracy for ResNet as FT: ", test_accuracy)
# print("Test loss for ResNet as FT: ", test_loss)
# # print(train_set.img_labels[:50])
# if cm is not None:
#   plot_confusion_matrix(cm, ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"])