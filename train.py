# check for GPU
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
device = torch.device(dev) 
print(device)

# load train/test datasets
image_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 96

labels_csv_path = "/content/drive/MyDrive/sidewalk_cv/crop_info.csv"
img_dir = "/content/drive/MyDrive/sidewalk_cv/crops"
# train_labels_csv_path = "/content/drive/MyDrive/sidewalk_cv/train_crop_info_1.csv"
# train_img_dir = "/content/drive/MyDrive/sidewalk_cv/train_crops_1000"

# test_labels_csv_path = "/content/drive/MyDrive/sidewalk_cv/test_crop_info.csv"
# test_img_dir = "/content/drive/MyDrive/sidewalk_cv/test_crops"

# load our custom sidewalk crop dataset
# train_set = SidewalkCropsDataset(train_labels_csv_path, train_img_dir, image_transform)
dataset = SidewalkCropsDataset(labels_csv_path, img_dir, image_transform)

# partition dataset into 80/10/10 split for train/validation/test
k = .8
dataset_size = len(dataset)
train_size = int(k * dataset_size)
val_size = int((dataset_size - train_size) / 2)
test_size = dataset_size - train_size - val_size
print(dataset_size)
print(train_size)
print(val_size)
print(test_size)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

# Do we want a dedicated test set?
# test_dataset = SidewalkCropsDataset(test_labels_csv_path, test_img_dir, image_transform)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# get resnet50 for fine tuning
resnet50 = torchvision.models.resnet50(pretrained = True).to(device)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 5) # (1,2,3,4) for label types, 0 for null crops 

lr = 0.01

resnet50.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
checkpoint_save_path = BASE_PATH + "training_test_saves"

# train for 20 epochs
epochs = 20
dataLoaders = {
  "training": train_dataloader,
  "validation": val_dataloader
}
loss_train, loss_validation, last_epoch = load_training_checkpoint(resnet50, optimizer, scheduler, checkpoint_save_path)
print("training losses: " + str(loss_train))
print("validation losses: " + str(loss_validation))
print("next epoch: " + str(last_epoch + 1))
print("resuming training...\n")

resnet50, best_validation_accuracy, loss_train, loss_validation = train(resnet50, optimizer, scheduler, loss_func, epochs, dataLoaders, checkpoint_save_path, loss_train, loss_validation, last_epoch + 1)
print("Best validation accuracy: ", best_validation_accuracy)

# visualization of training and validation loss over epochs
plt.plot(np.arange(epochs), loss_train, label="training loss")
plt.plot(np.arange(epochs), loss_validation, label="validation loss")
plt.title("Training/Validation loss for FT model")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()