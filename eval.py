# make sure to evaluate on *_pretrained if loading pretrained model
test_accuracy, test_loss, cm = evaluate(resnet50_pretrained, loss_func, test_dataloader, True)
print("Test accuracy for ResNet as FT: ", test_accuracy)
print("Test loss for ResNet as FT: ", test_loss)
# print(train_set.img_labels[:50])
if cm is not None:
  plot_confusion_matrix(cm, ["null", "curb ramp", "missing ramp", "obstruction", "sfc problem"])