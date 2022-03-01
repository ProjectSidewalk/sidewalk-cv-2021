import citysurfaces.network.hrnetv2 as hrnetv2
import copy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm

from architectures.two_model_ensemble import TwoModelEnsembleNet
from architectures.coatnet import coatnet_0
from sklearn.metrics import confusion_matrix
from time import perf_counter
from torch.optim import lr_scheduler
from tqdm import tqdm


CITYSURFACES_PRETRAINED_MODEL_PATH = "./models/block_c_10classes.pth"


def get_pretrained_model(model_name, num_classes, use_pretrained=True):

  if model_name == "resnet":
      """ Resnet50
      """
      model = torchvision.models.resnet50(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features
      model.fc = nn.Linear(num_ftrs, num_classes)
      input_size = 224

  elif model_name == "inception":
      """ Inception v3
      Be careful, expects (299,299) sized images and has auxiliary output
      """
      model = torchvision.models.inception_v3(pretrained=use_pretrained)
      # Handle the auxilary net
      num_ftrs = model_ft.AuxLogits.fc.in_features
      model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
      # Handle the primary net
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs,num_classes)
      input_size = 299

  elif model_name == "efficientnet":
    """ EfficientNetB3
    """
    model = torchvision.models.efficientnet_b3(pretrained=use_pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    input_size = 224
  
  elif model_name == "regnet":
    """ RegNet-y, 8gF
    """
    model = torchvision.models.regnet_y_8gf(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
  elif model_name == "hrnet":
    """ Maryam's custom architecture trained on her citysurfaces dataset
    """
    model = hrnetv2.get_hrnet_with_citysurface_weights(CITYSURFACES_PRETRAINED_MODEL_PATH, num_classes)
    input_size = 224
  elif model_name == "coatnet":
    """ CoAtNet 0
    """
    model = coatnet_0(num_classes)
    input_size = 224
  else:
    print("Invalid model name, exiting...")
    exit()


  return model, input_size
  

# # Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

def save_training_checkpoint(training_states, best_model_state, metrics, epoch, path):
  # add things like TPR, FPR later when we start evaluating them
  torch.save({'model_state': training_states['model'].state_dict(),
              'best_model_state': best_model_state,
              'optimizer_state': training_states['optimizer'].state_dict(),
              'scheduler_state': training_states['scheduler'].state_dict(),
              'metrics': metrics,
              'epoch': epoch
              }, path)
  print("saved")
  

def load_training_checkpoint(model, path, optimizer=None, scheduler=None):
  if not os.path.isfile(path):
    return {'loss_train': [],
    'loss_validation': [], 
    'accuracy_train': [], 
    'accuracy_validation': [],
    'precision_train': [],
    'recall_train': [],
    'precision_validation': [],
    'recall_validation': []}, -1  # training starts at last epoch + 1
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state'])

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state'])
  
  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state'])
  
  return checkpoint['metrics'], checkpoint['epoch']
  

def load_best_weights(model, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['best_model_state'])


def train(model, num_classes, is_inception, optimizer, scheduler, loss_func, epochs, datasetLoaders, save_path, metrics, start_epoch, device):
  t_start = perf_counter()

  best_model_state = copy.deepcopy(model.state_dict())

  is_cyclic_lr = isinstance(scheduler, lr_scheduler.CyclicLR)

  is_two_model_ensemble = isinstance(model, TwoModelEnsembleNet)
  print("two model ensemble ", is_two_model_ensemble)

  for epoch in range(start_epoch, epochs):
    epoch_t_start = perf_counter()
    print("Epoch " + str(epoch) + " out of " + str(epochs))
    for mode in ['training', 'validation']:
      if mode == "training":
        # set model to training mode.
        model.train()
      else:
        # set model to evaluation mode for validation set.
        model.eval()

      # get length of dataset for current mode
      n = len(datasetLoaders[mode].dataset)

      # sum of losses over batches.
      total_loss = 0

      # number of correct predictions.
      total_correct = 0

      epoch_count = 0

      pred_positive_counts = torch.zeros(num_classes).to(device)
      actual_positive_counts = torch.zeros(num_classes).to(device)
      true_positive_counts = torch.zeros(num_classes).to(device)
      for inputs, labels in tqdm(datasetLoaders[mode]):
        if is_two_model_ensemble:
          inputs_small, inputs_large, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
        else:
          inputs, labels = inputs.to(device), labels.to(device)
        epoch_count += inputs_large.size(0) if is_two_model_ensemble else inputs.size(0)
        # For code brevity, we'll set the reset gradients for model params
        # with the intention of using it for training.
        # We'll use the set_grad_enabled to toggle whether we actually use the
        # gradient for training/validation.
        # https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html
        optimizer.zero_grad()
        use_grad = (mode == 'training')
        with torch.set_grad_enabled(use_grad):
          if is_inception and use_grad:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = model(inputs)
            loss1 = loss_func(outputs, labels)
            loss2 = loss_func(aux_outputs, labels)
            loss = loss1 + 0.4*loss2
          elif is_two_model_ensemble:
            outputs = model(inputs_small, inputs_large)
            loss = loss_func(outputs, labels)
          else:
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

          _, preds = torch.max(outputs, 1)
          preds.to(device)

          if use_grad:
            # We are training, so make sure to actually
            # train by using loss/stepping.
            loss.backward()
            optimizer.step()

            if is_cyclic_lr:
              scheduler.step()
        
        correct_preds = torch.where(preds == labels, preds, torch.tensor([-1 for i in preds]).to(device))
        for i in range(num_classes):
          true_positive_counts[i] += torch.count_nonzero(correct_preds == i)
          pred_positive_counts[i] += torch.count_nonzero(preds == i)
          actual_positive_counts[i] += torch.count_nonzero(labels == i)
        total_loss += loss.item() * inputs_large.size(0) if is_two_model_ensemble else loss.item() * inputs.size(0) # Averages over batch              
        total_correct += (preds == labels).sum().item() # what is .data for?

      # calculate average loss over batches
      loss_avg = total_loss / n

      # calculate accuracy
      accuracy = total_correct / n
      
      # calculate precisions and recalls
      precisions = torch.div(true_positive_counts, pred_positive_counts)
      recalls = torch.div(true_positive_counts, actual_positive_counts)

      print("mode: " + mode + ", accuracy: " + str(accuracy) + ", loss: " + str(loss_avg))

      if mode == 'validation':
        if not metrics['accuracy_validation'] or accuracy > np.max(metrics['accuracy_validation']):
          # we found a better accuracy with these new weights, so save them
          best_model_state = copy.deepcopy(model.state_dict())
        metrics['precision_validation'].append(precisions)
        metrics['recall_validation'].append(recalls)
        metrics['loss_validation'].append(loss_avg)
        metrics['accuracy_validation'].append(accuracy)
      else:
        # record training accuracy
        metrics['precision_train'].append(precisions)
        metrics['recall_train'].append(recalls)
        metrics['loss_train'].append(loss_avg)
        metrics['accuracy_train'].append(accuracy)
        # make sure to step through lr update schedule (if not using cyclic)
        if not is_cyclic_lr:
          print("epoch LR scheduler step")
          scheduler.step()

    training_states = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    save_training_checkpoint(training_states, best_model_state, metrics, epoch, save_path)
    epoch_t_stop = perf_counter()
    print("Elapsed time during epoch {}".format(epoch),
                                        epoch_t_stop-epoch_t_start)
    print("\n")

  t_stop = perf_counter()
  print("Elapsed time during training in seconds",
                                        t_stop-t_start)

def evaluate(model, is_inception, loss_func, dataset_loader, test, mistakes_save_path, device):
  # put model into eval mode
  model.eval()

  is_two_model_ensemble = isinstance(model, TwoModelEnsembleNet)
  print("two model ensemble ", is_two_model_ensemble)
  
  # length of data set we are evaluating on.
  n = len(dataset_loader.dataset)

  # initialize the prediction and label lists(tensors) for our confusion matrix
  predlist=torch.zeros(0, dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0, dtype=torch.long, device='cpu')
  conf_mat = None
  
  incorrect_predictions = []
  corresponding_ground_truths = []
  corresponding_image_names = []

  epoch_count = 0

  # correct predictions.
  correct = 0
  total_loss = 0
  with torch.no_grad():
    for inputs, labels, paths in dataset_loader:
      if is_two_model_ensemble:
        inputs_small, inputs_large, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
      else:
        inputs, labels = inputs.to(device), labels.to(device)
      epoch_count += inputs.size(0)
      print("percent {}".format(epoch_count / n))
      if is_inception:
        outputs, _ = model(inputs)
      elif is_two_model_ensemble:
        outputs = model(inputs_small, inputs_large)
      else:
        outputs = model(inputs)

      # we ignore aux output in test loss calculation
      # since we aren't updating weights
      loss = loss_func(outputs, labels)
      _, predictions = torch.max(outputs, 1)

      # append batch prediction results and labels
      predlist=torch.cat([predlist, predictions.view(-1).cpu()])
      lbllist=torch.cat([lbllist, labels.view(-1).cpu()])

      total_loss += loss.item() * inputs_large.size(0) if is_two_model_ensemble else loss.item() * inputs.size(0) # Averages over batch              
      correct += (predictions == labels).sum().item()  # what is labels.data

      incorrect_indices = torch.nonzero(predictions != labels, as_tuple=True)[0]
      for index in incorrect_indices:
        incorrect_predictions.append(predictions[index].item())
        corresponding_ground_truths.append(labels[index].item())
        corresponding_image_names.append(paths[index])

  print(n - correct)
  print(len(incorrect_predictions))
  mistakes = pd.DataFrame(
    {
      'prediction': incorrect_predictions,
      'ground truth': corresponding_ground_truths,
      'image path': corresponding_image_names
    }
  )

  mistakes.to_csv(mistakes_save_path, index=False)

  if test:
    # display a Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

  return  correct / n, total_loss / n, conf_mat
