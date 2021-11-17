# # load our custom test sidewalk crops dataset
# test_labels_csv_path = BASE_PATH + "test_crop_info.csv"
# test_img_dir = BASE_PATH + "test_crops/"
# test_dataset = SidewalkCropsDataset(test_labels_csv_path, test_img_dir, image_transform)

# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# make sure to evaluate on *_pretrained if loading pretrained model
# test_accuracy, test_loss, cm = evaluate(resnet50_pretrained, loss_func, test_dataloader, True)
# print("Test accuracy for ResNet as FT: ", test_accuracy)
# print("Test loss for ResNet as FT: ", test_loss)
# # print(train_set.img_labels[:50])
# if cm is not None:
#   plot_confusion_matrix(cm, ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"])