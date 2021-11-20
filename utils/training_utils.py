import copy
import glob
import numpy as np
import os
import torch
import torchvision
from sklearn.metrics import confusion_matrix
from time import perf_counter

def get_pretrained_model():
  return torchvision.models.regnet_y_8gf(pretrained = True)

def save_training_checkpoint(training_states, best_model_state, metrics, epoch, folder):
  # add things like TPR, FPR later when we start evaluating them
  if not os.path.exists(folder):
    os.makedirs(folder)
  torch.save({'model_state': training_states['model'].state_dict(),
              'best_model_state': best_model_state,
              'optimizer_state': training_states['optimizer'].state_dict(),
              'scheduler_state': training_states['scheduler'].state_dict(),
              'metrics': metrics,
              'epoch': epoch
              }, f'{folder}epoch_{str(epoch)}.pt')
  print("saved")
  

def load_training_checkpoint(model, folder, optimizer=None, scheduler=None):
  checkpoint_files = glob.glob(f'{folder}*.pt')
  if not checkpoint_files:
    return {'loss_train': [],
    'loss_validation': [], 
    'accuracy_train': [], 
    'accuracy_validation': [],
    'precision_train': [],
    'recall_train': [],
    'precision_validation': [],
    'recall_validation': []}, -1  # training starts at last epoch + 1
  most_recent_save = max(checkpoint_files, key=os.path.getmtime)
  checkpoint = torch.load(most_recent_save)

  model.load_state_dict(checkpoint['model_state'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state'])
  
  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state'])
  
  return checkpoint['metrics'], checkpoint['epoch']
  

def load_best_weights(model, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['best_model_state'])


def train(model, optimizer, scheduler, loss_func, epochs, datasetLoaders, save_folder, metrics, start_epoch, device):
  t_start = perf_counter()

  best_model_state = copy.deepcopy(model.state_dict())

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

      pred_positive_counts = torch.zeros(5).to(device)
      actual_positive_counts = torch.zeros(5).to(device)
      true_positive_counts = torch.zeros(5).to(device)

      for inputs, labels in datasetLoaders[mode]:
        inputs, labels = inputs.to(device), labels.to(device)
        epoch_count += inputs.size(0)
        print("percent {}".format(epoch_count / n))
        # For code brevity, we'll set the reset gradients for model params
        # with the intention of using it for training.
        # We'll use the set_grad_enabled to toggle whether we actually use the
        # gradient for training/validation.
        # https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html
        optimizer.zero_grad()
        use_grad = (mode == 'training')
        with torch.set_grad_enabled(use_grad):
          outputs = model(inputs)
          loss = loss_func(outputs, labels)
          _, preds = torch.max(outputs, 1)
          preds.to(device)
          if use_grad:
            # We are training, so make sure to actually
            # train by using loss/stepping.
            loss.backward()
            optimizer.step()
        
        correct_preds = torch.where(preds == labels.data, preds, torch.tensor([-1 for i in preds]).to(device))
        for i in range(5):
          true_positive_counts[i] += torch.count_nonzero(correct_preds == i)
          pred_positive_counts[i] += torch.count_nonzero(preds == i)
          actual_positive_counts[i] += torch.count_nonzero(labels == i)
        total_loss += loss.item() * inputs.size(0) # Averages over batch              
        total_correct += (preds == labels.data).sum().item() # what is .data for?

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
        # make sure to step through lr update schedule
        #scheduler.step()

    training_states = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
    save_training_checkpoint(training_states, best_model_state, metrics, epoch, save_folder)
    epoch_t_stop = perf_counter()
    print("Elapsed time during epoch {}".format(epoch),
                                        epoch_t_stop-epoch_t_start)
    print("\n")

  t_stop = perf_counter()
  print("Elapsed time during training in seconds",
                                        t_stop-t_start)

def evaluate(model, loss_func, dataset_loader, test, device):
  # put model into eval mode
  model.eval()
  
  # length of data set we are evaluating on.
  n = len(dataset_loader.dataset)

  # initialize the prediction and label lists(tensors) for our confusion matrix
  predlist=torch.zeros(0, dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0, dtype=torch.long, device='cpu')
  conf_mat = None

  # correct predictions.
  correct = 0
  total_loss = 0
  with torch.no_grad():
    for inputs, labels in dataset_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = loss_func(outputs, labels)
      _, predictions = torch.max(outputs, 1)

      # append batch prediction results and labels
      predlist=torch.cat([predlist, predictions.view(-1).cpu()])
      lbllist=torch.cat([lbllist, labels.view(-1).cpu()])

      total_loss += loss.item() * inputs.size(0)  # weighted average with size?
      correct += (predictions == labels).sum().item()  # what is labels.data

  if test:
    # display a Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

  return  correct / n, total_loss / n, conf_mat